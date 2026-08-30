# 代码结构改进设计

接 `code-structure-analysis.md` 的五个改进机会，每个落到「目标 / 现状 / 改动 /
验证 / 风险」。按优先级排序——前三个是闸门增强（小、立即做），后两个是结构
重构（大、需独立设计）。

---

## 1. `#endif` 注释守卫（最高优先，最小）

**目标**：阻止裸 `#endif` 回归（`acef096` 刚补完 49 个，回退最易）。

**改动**：`build/dead_ifndef_gate.sh` 同款脚本 `build/bare_endif_gate.sh`：
`awk '/^[[:space:]]*#endif[[:space:]]*$/{c++} END{print c}' src/*.c common/*.c`
断言为 0，接进 `make check`（和 dead-ifndef 一样是最前面，不依赖编译）。

**验证**：删掉一个 `#endif` 注释，闸门变红；恢复后变绿。

**风险**：无。纯文档闸门，和 `acef096` 的注释改动零冲突。

---

## 2. 守卫矩阵交叉组合（高优先）

**现状**：`986917f` 的 `switch_matrix_gate.sh` 覆盖 15 个单开关，但没有交叉。
已知会出问题的交叉：
- `LZ_SSE2_GROUP=0` × `LZ_GDN_FIXED=2`（#32 的伴生约束，已配 `2PLANE=0`）
- `LZ_NO_FPU=1` × `LZ_SCALAR`（no-FPU 时 scalar 必然 fixed，二者语义重叠）
- `LZ_GDN_STATE_2PLANE=0` × `LZ_GDN_FIXED=0`（两平面关 + 定点关）

**改动**：`switch_matrix_gate.sh` 加一个 `CROSS` 列表，每个交叉组合各 `check_one`
一次（编译 + `-dM` 断言两个宏都生效）。

**验证**：矩阵 PASS，交叉组合各编译通过。

**风险**：低。只是多几个 `check_one` 条目。

---

## 3. `fldt` 判据增强（中优先）

**现状**：`test_excess_precision.py`（`13aae8f`）用 `round_trips_float32()` 判据，
但 `forward.c` 那个「`static const float` 后仍被调度器提升为 `fldt`」的案例
覆盖不足——判据验证的是「常量的值」，不是「常量的位置」。

**改动**：判据额外断言「每个 `fldt` 加载的常量，其 ops.c 行号对应的源码是
`static const float` 定义」，这样调度器挪动加载位置（同一个值、不同位置）不会
触发假红，而「新写一个 inline 字面量」会触发真红。

**验证**：故意把一个 `static const float` 改回 inline 字面量，判据红；恢复后绿。

**风险**：低。纯测试增强，不改引擎。

---

## 4. forward.c 定点路径审计（最高价值，需独立设计）

**现状**：norm gate 的 Q15 整数乘积（`6f9b3d1`）已经消除了 sigmoid 的 int→float
往返，但 norms 位点还剩 6 次 float mul/元素（3×scale + 2×sigmoid 插值 + 1×dequant），
占 fixed 档 35.7%。`lz_quantize_q8` 有 12 个调用点，全是 `float→int8`。

**核心问题**：norm 的输出是 float，下游 `lz_quantize_q8` 再 float→int8。如果 norm
的进/出域都整数化（预先量化 x/w/g 到 Q15，循环纯 int，输出保持 int），norms 位点
的 f32 可归零。但这需要**整个上游链条整数化**——`lz_quantize_q8` 的 12 个调用点，
有 4 个（1248/1713/1923/2213 的 `xb`、`attn_out`、`ssm_out`）的输入来自 norm 或
attention 的 float 输出，要让它们 int 贯通，需要 norm/attention 的输出也保持 int。

**改动路径**（分三步，每步独立可验证）：

1. **norm 输出保持 int**：`lz_rmsnorm_gated_out_fixed` 的 Q15 乘积结果，不再
   dequant 回 float，而是直接写 int 到输出缓冲，供 `lz_quantize_q8` 的 int 变体
   消费。`lz_quantize_q8` 加一个 `lz_quantize_q8_int` 变体（接受 Q15 int 输入）。
2. **attention 输出保持 int**：`attn_out`（1670 的 `lz_quantize_q8` 上游）也整数化。
3. **逐层贯通**：1248/1713/1923/2213 的量化链，逐步把上游 float 换成 int。

**验证**：每步 precision-correct（不 bit-identical，和选项 A 一致），gcc==Watcom，
census 的 norms 位点 mul/add 逐级下降（最终趋近零）。

**风险**：**高**。这是选项 A 的最深度重构，改变 `lz_quantize_q8` 的接口（影响 12 个
调用点），且 Q15 域的精度（15 位）低于 float（24 位），逐层贯通时误差会累积。
必须每步独立验证 precision-correct，且做成旋钮（`--norm-int` 或复用 `--norm-write
fixed`）让 float 路径可退回。

---

## 5. DMA 隔离验证（中优先）

**现状**：forward.c 2954 行，约 800 行纯数学（GDN/KDA），约 700 行状态管理
（cache/ring/slot）。纯数学部分理论上可 `test_ops.c` 隔离验证，但当前只能端到端
验证（`--dump-logits`）。

**改动**：把 forward.c 的纯数学函数（`lz_gdn_step` 的数学核、`lz_attn_score` 的
打分核）的签名抽成「无状态输入 → 无状态输出」，`test_ops.c` 加针对性契约。

**验证**：`test_ops.c` 直接调这些纯函数，对标量参考对拍。

**风险**：中。抽取签名会动 forward.c 的函数边界，但纯数学函数本就不该碰
cache/ring/slot 状态，抽出后隔离验证更可靠。

---

## 执行顺序建议

1. **立即做**：#1（`#endif` 守卫）、#2（矩阵交叉）、#3（`fldt` 判据）——三个闸门
   增强，小、独立、纯测试/脚本，不碰引擎功能代码。
2. **独立设计后做**：#4（forward.c 定点化）——选项 A 的延续，价值最高但风险最高，
   需三步走 + 每步旋钮隔离 + precision-correct 验证。
3. **视需要做**：#5（DMA 隔离）——提升可测性，不急。
