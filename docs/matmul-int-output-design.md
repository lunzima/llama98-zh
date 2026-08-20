# matmul 输出 int 化设计

新立项：让 matmul 内核输出定点 int 而非 float，消除「matmul float 输出 →
量化 int8」的中转（步 3 审计发现的 ~15 个调用点）。

## 现状

matmul（`lz_matmul_xq_nt` → `matmul_q8_impl` 等，ops_matmul.c）的累加器是
**per-32-input-group 的 int32**（`g_acc32`，nb = in_dim/32 个）。epilogue
`epi_q8`（ops.c:985）做跨 per-group scale 的加权 dequant：

```
o[i] = sum_g accb[i][g] * (xsb[g] * ws[i][g])
```

- `xsb[g]` = activation scale（per-input-group，来自 `lz_quantize_q8` 的
  `xqs`，`in_dim/gs` 个）。
- `ws[i][g]` = weight scale（per-row per-group，静态）。
- `accb[i][g]` = int32 累加（int8×int8 → int32，per-32-input-group）。

下游 `lz_quantize_q8` 再 float → int8。

## 核心困难：scale 维度错位（和步 2 的本质区别）

步 2（attention wsum）能 int 化，因为它的 `sscale = cmax*LZ_Q15_INV` 是
**单一 per-32-group 值**，且量化也是 per-32-group——scale 维度和量化粒度
对齐，`deq` 折进 scale 即可。

matmul 的 scale 是 **per-input-group 的 `xsb[g]*ws[g]`（ng 个不同值）**，
而下游量化是 **per-output-group 的 `gs`**。两者维度错位：

- int 域 amax 需要「跨 input-group 的 scale 加权和的 max」——
  `max_i |sum_g accb[i][g] * scale[i][g]|`，在 scale per-group 不同时无法
  像步 2 那样把单一 `deq` 提出去。
- 不能简单复用 `lz_quantize_q8_int64`（它假设单一 `deq`）。

## 定性：格式级重构，不是「加 int 变体」

这是**量化格式**的改动，不是「像步 2 加一个 int 变体」的轻量工程。任何
int 输出方案都会改变「量化粒度」（per-output-group → 别的东西），下游
所有 matmul 调用点 + 量化消费者都要适配。

## 可行方案（待验证，按可行性排序）

### 方案 B：预乘 scale（若 activation scale 全局统一）

若 `xsb` 全局统一（不 per-group），则 `scale = sx * ws[i][g]`，`ws` 静态
可预乘进 int 域，工程大幅简化。**当前 `xsb` 是 per-group 的**（`lz_quantize_q8`
输出 `in_dim/gs` 个 scale），方案 B 需先改「activation 量化为全局 scale」，
那是另一个精度权衡。

### 方案 A：per-input-group int 量化（改变量化格式）

- matmul 输出 `accb[g]`（int32）+ `scale[g]`（float）。
- 下游「per-input-group 量化」：`amax_g = max|accb[g]|`（int 域精确）、
  `q_g = round(accb[g]*127/amax_g)`、`s_g = amax_g*scale[g]*(1/127)`。
- 量化粒度从「per-output-group」变「per-input-group」，value-changing，
  下游消费者（量化格式）全适配。int 域 amax 精确，但量化粒度改变影响精度。

## 分阶段（若做）

1. 先定「量化粒度」：方案 A 的 per-input-group 量化是否可接受（精度影响
   需实测，量化粒度改变会动下游 matmul 的 scale 语义）。
2. Q8_0 dtype 先行（最常用），验证精度 + 收益。
3. 后续 dtype（q41/q61/t2/q16）。
4. 铁律二：每个新内核 gcc intrinsics + Watcom `#pragma aux` 两份 bit-identical。

## 实测（2026-08-20）：接线位点上 int 路径不省 f32

`-DLZ_F32COUNT=1` 构建，kmr20 16 token，`--matmul-int off` vs `on`：

| | dequant epilogue（每 token） |
|---|---|
| off | 1,020,192 |
| on | **1,021,664**（+0.14%） |

**第一次量出的「-9.9%」是仪器假象**：`matmul_q8_int_consume` 当时零记账，
而 `epi_q8` 记 10 处，于是开关只是让那个位点停止记账。逐位验算证实差值
（mul 34,816 / add 33,792 / cvt 32,768，除以 1024 得 34/33/32）恰好是
`epi_q8` 在 ng=32、ng/r=1 时的记账公式。补上记账后数字翻转。

**结论：`w2==NULL` 位点（o_proj → 残差加）结构上不可能赢。** 每组
int 路径花 1 cvt + 1 mul + 1 add（量化）对 `epi_q8` 的 1 cvt + 1 mul +
1 add（反量化），而 int 路径还多出 pass 1 的 per-group amax/invb/A 折叠
（3 mul + 1 add + 1 div + 2 cvt）。省下的浮点乘加被量化本身吃掉——
**量化到 int8 只有在下游真有 int8×int8 matmul 消费它时才划算**。

真正的收益只存在于 `w2!=NULL`（真实的 producer→consumer matmul 对），
那里同时省掉 producer 的浮点反量化和 consumer 的浮点再量化。该路径已由
`.prof` 探针验过数值正确，但**没有接线**，因为本模型的 o_proj 输出进的是
残差加，不是 matmul。

## 风险

- **高**：格式级重构 + 5 dtype + ~28 调用点 + 铁律二。
- 精度：value-changing，且量化粒度改变（方案 A）的影响需实测。
- 收益：census 里「dequant epilogue」占 11.0%（136000 计数），int 化省
  的是 `lz_i32f`（int32→float）+ float 乘 + float→int8 量化。
