# 代码结构分析

基于 `feat/attn-conv-fixed-tiers` 分支（截至 `acef096`，含选项 A 全部提交
和去意面注释）。只读分析，不改任何代码。

---

## 一、模块边界

```
src/ops.c          6082 行   21 个 #include    所有算子/内核/量化/归一化
src/forward.c      2954 行    8 个 #include    推理 pass + 状态传递
src/generate.c     2685 行                      生成循环（贪心/top-p）
src/cli_main.c     2279 行                      CLI 参数 + 档位选择
src/model.c        1502 行                      safetensors 加载
src/ops_mmx.c      2910 行                      后缀 TU：MMX 内核（#30 搬进）
src/ops_sse2.c     1794 行                      后缀 TU：SSE2 内核
src/ops_mmx_sse.c   565 行                      后缀 TU：SSE1-on-MMX 内核
common/session.c    551 行                      会话/状态管理
```

**单体文件**：`ops.c` + 它 `#include` 的六个 `ops_kernel_*.h` + `ops_mmx.h` +
`mmx_compat.h` + `ops_kernel_shared.h` ≈ 11000 行编译进同一个 TU，21 个
`#include`（高耦合）。这是引擎最大、最复杂的模块。

---

## 二、ops.c 内部结构（6082 行 + 136 个 `#if`/`#ifdef`）

守卫密度 **每 42 行一个守卫**——这是结构债务的核心数量指标。

| 行范围 | 功能区 | 守卫性质 |
|---|---|---|
| 1-225 | forward 声明、类型、`#include` | 几乎无守卫 |
| 225-600 | norm 定点化（`norm_ss_fixed`）、sigmoid Q15 表（`sig_build`） | `LZ_GDN_STATE_2PLANE`、`LZ_GDN_FIXED`、`__WATCOMC__` |
| 629-800 | `q8_round`、rsqrt 定点化（`lz_rsqrt_fixed`） | 无守卫 |
| 823-930 | exp 定点化（`lz_exp_fixed`） | 无守卫 |
| 997-1427 | CPU 检测（`q8r_have_simd`、`lz_cpu_has_mmx/sse`） | `__WATCOMC__`、`LZ_CPUID_NOCACHE` |
| 1430-1740 | prefetch 检测、档位控制（六个 `g_` 静态变量） | `LZ_GDN_FIXED`、`LZ_ATTN_FIXED`、`LZ_CONV_FIXED` 等 |
| 1854-2040 | epilogue MAC、`lz_q8r_tier` | `LZ_HAVE_Q8R_SIMD/SSE`、`LZ_Q8R_FORCE_SSE` |
| 2040-2400 | 标量参考内核（`matmul_scalar_ref`）、`lz_row_pick` | `LZ_DOT_MMX_EXTERN`、`LZ_ROW_SSE2_EXTERN` 等 |
| 2400-5700 | 五个 matmul 行内核、dot 内核、amax/norm、**GDN/KDA pass**、attention wsum | `__WATCOMC__`、`LZ_GDN1_MMX_EXTERN`、`LZ_GDN_FIXED`、`LZ_GDN_STATE_2PLANE`、`LZ_ATTN_FIXED`、`LZ_SSE2_GROUP`、`LZ_HAVE_P2_MMX` 等 |

**内核调度机制**：五层嵌套——

```
lz_row_pick()             ← 表：LZ_ROW_Q8/Q41/Q61/T2/Q16
  → _mmx/_sse2/_intrin   ← gcc intrinsics
  → _asm                  ← Watcom #pragma aux（#30 后跨 TU extern）
  → LZ_G128_GCC / LZ_G128_*   ← 128 组内核（选项 A 后 gcc 也有）
```

每个表有 6-7 槽（mmx-intrin / sse-intrin / sse2-intrin / mmx-asm / sse-asm /
sse2-asm / g128），由 `#if defined(LZ_*_EXTERN)` + `#if defined(__WATCOMC__)` 守卫
分别填充。`lz_row_pick()` 在运行时根据 `g_kernel` 选择。

---

## 三、forward.c 数据流

### 整数域（int8/int16 贯通，无 int→float 往返）

| 存储 | 类型 | 位置 | 说明 |
|---|---|---|---|
| `kq8`/`kq8_lo` | int8 | 587 | K cache，attention 直接读 int8 |
| `vq8`/`vq8_lo` | int8 | 603 | V cache，attention 直接读 int8 |
| `ssm_state_q8` | int8 | 731 | GDN 状态，`lz_gdn_step` int8 贯通 |
| `wsum_cq` | int16 | 507 | attention wsum 定点累积 |
| `xq` | int8 | 829 | 量化激活 |

跨 token 的状态都是 int8/int16，无 `lz_i32f` 的 int→float 解量化往返。

### int↔float 边界（必要的量化）

4 处 `lz_quantize_q8` + 2 处 sigmoid float 乘，总 `float→int` 输出：

| 位置 | 操作 | 说明 |
|---|---|---|
| 1248 | `lz_quantize_q8(s->xb...)` | 激活量化（float→int8） |
| 1307 | `lz_quantize_q8(ktt...)` | K cache 写入（float→int8） |
| 1309 | `lz_quantize_q8(vtt...)` | V cache 写入（float→int8） |
| 1659 | `lz_sigmoid(gate[d])` | sigmoid float 乘（不可消除，不是量化） |
| 1670 | `lz_quantize_q8(s->attn_out...)` | attention 输出量化 |
| 2130+ | `lz_quantize_q8` / `lz_quantize_q8_silu` | norm gate 输出量化 |

**sigmoid 的 int→float 往返已消除**（`6f9b3d1`，norm 路径的 Q15 sigmoid 融进 `lz_quantize_q8_silu`）。

---

## 四、结构债务

### 4.1 单体文件

`ops.c` + 六个 `ops_kernel_*.h` + `ops_mmx.h` + `mmx_compat.h` ≈ 11000 行
编进一个 TU，守卫 136 个（1/42 行），这是可读性和可维护性的主要瓶颈。

### 4.2 `__WATCOMC__` 守卫

`ops.c` 有 **61 处 `__WATCOMC__`**（经过 #30 清理后），大部分是「编译器身份」
（`#pragma aux`、`_m_*` intrinsics），少数是「架构代理」（#21 已修的典型，
余下 0 处残留，阶段 2 的 audit 已确认）。

`LZ_HAVE_*` 命名仍然分散（阶段 3 的 `||ASM||EXTERN` 合并 + 跨文件去重，
`8a23c16`），但仍可被更统一的 `LZ_*_EXTERN` 完全替代——`#29` 的阶段 3 已
迈出第一步，但后续的 `#if __WATCOMC__` 守卫→ `__WATCOMC__` 区分（阶段 2）
和 `LZ_HAVE_*` → `LZ_*_EXTERN` 迁移（阶段 3 的剩余）尚未完成。

### 4.3 内核调度机制

五层派发表 + 12 个守卫宏（`LZ_*_EXTERN` / `LZ_HAVE_*` / `LZ_*_ASM`），加上
六个档位控制（`--recur-write fixed` 等）、`lz_soft_float_target()` 运行时选择、
三个 SYMMETRIC ISA 闸门（`test_ops_isa_floor.py`、`test_excess_precision.py`、
`test_cpuid_guard.py`）——这套系统的正确性建立在「每个守卫组合至少被建过一次」
的假设上，而 `986917f` 的开关矩阵只覆盖了 gcc + Watcom 的当前配置。

### 4.4 两个可能矛盾的约束

- **铁律二**要求 bit-identical（跨编译器一致），而**选项 A** 要求 precision-correct
  （输出改变）。两者共存时正确做法是「旋钮隔离」（`--scalar-fixed auto/float/fixed`，
  `508e015`），但选项 B（bit-identical）仍然是浮点档的约束——区分这两条路线的是
  档位，不是全局开关。
- **铁律四**和**铁律六第 6 条**同时在场：铁律四说「判据能被垃圾骗过」，而铁律六
  第 6 条（`kernels don't emit, callers do`）现在已经事实上违反（选项 A 的 norm
  循环在内核 TU 内部 emms）——两件事都对，冲突要在上下文中裁量，不能靠修辞解决。

---

## 五、改进机会

1. **守卫矩阵**（CI 优先）：当前 `986917f` 覆盖了 15 个开关组合，但
   `LZ_SSE2_GROUP=0` × `LZ_NO_FPU=1` 这类交叉尚未跑过。矩阵应补上这几个
   已知的交叉点。
2. **flagt 检测闸门**：`test_excess_precision.py`（`13aae8f`）取代了棘轮，但它的
   `round_trips_float32()` 判据只验证浮点字面量本身——对 `static const float`
   赋值后**仍然被调度器提升为 `fldt`** 的情况（forward.c 仍在讨论的那个案例）
   覆盖不足。可以让它额外断言「每个 `fldt` 常量的 ops.c 行号」，这样调度器
   挪动加载时不会触发假红。
3. **`#endif` 注释守卫**：`144` 个裸 `#endif`（`acef096` 已补完）回归最易，用
   `git diff --stat` 或一个 `#endif`-unannotated 闸门拦住。
4. **forward.c 的定点路径审计**：norm 选项 A 后，fixed 档里 norm 的 mul+add
   已大幅降，但剩余 6 次 float mul（3×scale + sigmoid 插值 + 1×dequant）仍占
   norms 位点 35.7%。这些是否可以「在进/出域完全定点化」（保持 int 输出给
   `lz_quantize_q8`，省去 dequant 回 float），是下一步可以审计的方向。
5. **DMA 分析**：ops.c 调用 forward.c 时有 21 个 `#include`，forward.c 自身 8 个；
   两者间的函数调用线超过 20 条（`lz_gdn_step`、`lz_rmsnorm_gated`、
   `lz_quantize_q8`、`lz_silu`、`lz_sigmoid`、`lz_exp`、`lz_rsqrt`、
   `lz_i32f`、`lz_q8_amax`、`q8_round` 等）。forward.c 的 2954 行中
   约 800 行是纯数学（GDN/KDA），可以通过 `test_ops.c` 隔离验证，但
   约 700 行是状态管理（cache、ring、slot），目前只能端到端验证。
