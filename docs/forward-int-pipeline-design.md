# forward.c 定点路径：norm/attention 输出 int 贯通设计

接 `code-structure-design.md` 的 #4。这是「选项 A」的最后一块：消除
norm 定点档输出 float → 下游 `lz_quantize_q8` 再 float→int8 的重复转换。

## 现状（已核实，行号是当前 HEAD）

`forward_ssm` 的 sgdn norm 链（`forward.c:1831-1860`）：

1. `lz_rmsnorm_gated`（`ops_norm.c:121`）→ `lz_rmsnorm_gated_out_fixed`
   （`ops_norm.c:37`）用 Q15 int 算完整 silu，最后一处
   `o[i] = (float)t * pow2f(30-ex-ew-eg-ei)` **dequant 回 float**，写进
   `s->ssm_out`。
2. `lz_quantize_q8`（`ops.c:381`）读 `s->ssm_out`（float），amax → scale
   → round 量化到 int8（`s->xq`）。
3. 重复：Q15 int → float（dequant）→ int8（量化）。

`ssm_out` 的 float 消费者只有三个：`lz_quantize_q8`（主）、
`lz_matmul_xq_nt` 的 `x` 参数（`ops.h:712`，f32 fallback，只在量化不可用时
走）、`LZ_TAP`（`forward.c:28`，编译期空宏）。

## 两个关键发现

### 1. `lz_quantize_q8_silu` 是死代码

`ops.c:471` 有定义、`ops.h:569` 有声明，但**零调用**。它是「融合量化」模板
（把 Q15 的 2^-15 折叠进 scale，避免 sigmoid 的 Q15→float 往返），但实际路线
选了「norm 内部算完整 silu」（`ops_norm.c:35` 注释明说），所以 forward.c 走
普通 `lz_quantize_q8`。它不是 bug，是两条设计路线里没走的那条——但它的
「2^-15 folds into the scale」机制正是本设计步 1 要复用的。

### 2. 量化结果与 dequant scale 无关（bit-identical 证明）

norm 的 dequant 是 `o[i] = t * 2^k`，`k = 30-ex-ew-eg-ei`，`2^k` 是 2 的幂。
`lz_quantize_q8` 的 round 是 `q[i] = round(o[i] * 127/amax)`，其中
`amax = max|o[i]| = 2^k * max|t|`（2^k 缩放精确）。

代入：

```
q[i] = round(t*2^k * 127/(t_amax*2^k)) = round(t * 127/t_amax)
```

**2^k 消掉**。只要 `|t*127| < 2^22`（t 是 Q15 ±32767，`32767*127 = 4,161,409
< 2^22 = 4,194,304`，成立），`t*127` 在 float 里精确，于是 int 路径和 float
路径的 `q[i]` **逐位相同**。唯一可能分歧是存储的 `s[grp]` 在 deq denormal 时
差 1 ULP。

**结论：步 1 是 bit-identical，不是 value-changing。** 这比
`code-structure-design.md` #4 原估计（「不 bit-identical，和选项 A 一致」）
风险低一个量级——可以用 `--dump-logits` 逐字节验证，而不是容差。

## 方案：三步走

### 步 1：norm 输出 int + `lz_quantize_q8_int`（bit-identical，低风险）

- `lz_rmsnorm_gated_out_fixed` 增加 int 输出：写 Q15 `t` 到 int 缓冲，另给
  deq = `pow2f(k)` 一个标量（不是逐元素 dequant）。
- `lz_quantize_q8_int` 变体（接受 int32 Q15 + deq）：int 域 amax（`q8_amax`
  对位模式）、`s[grp] = t_amax * deq * (1/127)`、`q[i] = round(t*127/t_amax)`。
  deq 只进 scale，不进元素循环——正是 `lz_quantize_q8_silu` 的机制。
- f32 fallback 用 `t * deq` 重建 float（只在量化不可用时走，`lz_matmul_xq_nt`
  的 `x` 参数）。
- 旋钮：`--norm-int`（默认 off），或复用 `--norm-write fixed` 的开关形状。
- 验证：`--dump-logits` 逐字节不变（bit-identity，比容差强）。

### 步 2：attention 输出 int

- `attn_out`（`forward.c:1670` 上游）也整数化。
- 这里开始碰**求和**（attention 加权和），结合顺序/精度累积，**从这里起
  value-changing**。
- 验证：precision-correct（gcc==Watcom 对齐 + 容差对噪声底）。

### 步 3：逐层贯通

- 23 个 `lz_quantize_q8` 调用点（原设计文档写「12」，已过时——见下），逐个
  把 float 上游换 int。
- Q15 精度（15 位）< float（24 位），逐层累积误差。
- 每步旋钮隔离，precision-correct。

## 风险与顺序

| 步 | 风险 | 验证 |
|---|---|---|
| 1 norm int | 低（bit-identical） | `--dump-logits` 逐字节 |
| 2 attention int | 中（value-changing 起点） | 容差 + gcc==Watcom |
| 3 逐层贯通 | 高（Q15 累积 + 23 调用点接口） | 每步旋钮 + 容差 |

步 1 独立可交付、独立可验证，且因为 bit-identical，它是「选项 A 里唯一
不改变数值的一步」——先做步 1 可以立刻收回 norms 位点的重复转换，且不用
为 precision-correct 的验收标准付代价。

## 遗留的文档债

- `code-structure-design.md` #4 写「12 个调用点」，实际 23 处（forward.c 20
  + ops.c 1 + ops_gdn.c 2），且写「不 bit-identical」，步 1 实际 bit-identical。
  本设计落定后要回填那两条。
