# 架构预处理宏清理方案

目标三件事，按依赖排序：

1. **清除宏的屎山** —— 让每个 `#if` 一眼能看出它在问什么。
2. **预留 x86 no-387** —— 给「x86 无 387 FPU」留显式扩展点。
3. **保留现有 armv5te 无 FPU 方案** —— 清理不破坏 ARM 的 `-mfloat-abi=soft`。

---

## 一、现状盘点：六轴宏混在一个名字空间里

引擎里同一类 `#if` 现在问了六种不同的问题，而名字不区分：

| 轴 | 现在用的宏 | 混淆点 |
|---|---|---|
| (a) 编译器身份 | `__WATCOMC__`(~120) `__GNUC__` `__clang__` | 有时被当 (c) 用 |
| (b) 目标架构 | `__i386__` `__x86_64__` `__arm__` `_M_IX86` | — |
| (c) TU 能发什么 ISA | `__MMX__`(31) `__SSE__` `__SSE2__` `__SSE_MATH__` | 有时被当 (d) 用 |
| (d) 内核在构建里 | `LZ_MMX_TU` `LZ_SSE2_TU` `LZ_*_EXTERN`(7) `LZ_HAVE_*`(12) | 三套命名并存 |
| (e) 运行时 CPU | `lz_cpu_has_mmx/sse()`（CPUID） | — |
| (f) 用户选的档 | `LZ_KERNEL_*` `LZ_GDN_FIXED` `LZ_ATTN_FIXED` `LZ_CONV_FIXED` | — |

**已确认的误用实例（就是这次会话挖出来的）：**

- `ops.c` 的 `lz_fpu_float_begin` 曾用 `#if defined(__WATCOMC__)` 判 PC=24，
  正确判据是 `__i386__ && !__SSE_MATH__ && !__SSE2_MATH__`（#21 修的）。
  这是 (a) 冒充 (c) 的典型。
- `ops.c` 十处 `#if defined(__MMX__) && defined(__WATCOMC__)`，其实是
  「Watcom 的 asm 内核在构建里」，正确判据是 `__WATCOMC__`（拆内核后，
  内核已跨 TU extern，#30 修的）。这是 (c) 冒充 (d)。
- 两个 dead `#ifndef`（`LZ_ATTN_FIXED=0` 撒谎、`LZ_GDN_FIXED` 重复），
  #29 已修并加了 `build/dead_ifndef_gate.sh` 闸门。

**根源**：`__WATCOMC__` 和 `__MMX__` 被当成了多个轴的速记。名字不表达轴，
于是「读的人以为它在问 A，其实在问 B」。

---

## 二、float 可用性：这是 no-387 预留的落点

现状是「float 能不能用」散在三处，没有统一抽象：

```
lz_soft_float_target()          = q8r_have_simd()==0     ops.c:1314  （运行时，CPUID SSE）
--fixed auto / --recur-write 等 = lz_soft_float_target()  各档位解析
ARM 构建                        = -mfloat-abi=soft（编译期）+ 无 SIMD
x86 构建                        = 隐式「假定 387 存在」（Makefile 注释里的一句话）
```

**关键区别**：`lz_soft_float_target()` 判的是「无 **SSE**」，不是「无 **FPU**」。
486DX / 早期 Pentium 有 387 无 SSE，它们落进「定点」是「软浮点」的语义；
386 / 486SX 连 387 都没有，落进「定点」是「无浮点」的语义。两者现在被
同一个 `q8r_have_simd()==0` 兜住，没有显式区分。

**预留 x86 no-387 的正确落点**：加一个编译期宏 `LZ_NO_FPU`，语义是
「目标没有 FPU，float 运算不可用」，然后把 `lz_soft_float_target()`
扩成：

```c
/* 编译期无 FPU（x86 no-387 / ARM 软浮点）必然定点；
   运行时无 SSE 是软浮点，也走定点。 */
static int lz_soft_float_target(void) {
#if defined(LZ_NO_FPU)
    return 1;
#else
    return q8r_have_simd() == 0;
#endif
}
```

这样：

- **x86 no-387** 构建定义 `LZ_NO_FPU=1`（编译期声明「无 387」），未来接入时
  一行构建旗标即可，不用再翻运行时判断。
- **ARM 软浮点** 保持现状 —— 它已经靠「无 SIMD → 定点」走得对，`LZ_NO_FPU`
  不强制它（或者 ARM 构建显式定义 `LZ_NO_FPU=1` 作为文档化，两种都可，方案
  选后者，见 §四）。

`LZ_NO_FPU` 的边界要说清：它**不是**「软浮点 ABI」（`-mfloat-abi=soft`），
那个是编译器的事；它**不是**「无 SSE」（`q8r_have_simd()`），那个是运行时。
它是「引擎自己的定点-only」这个编译期事实。

---

## 三、目标设计：命名约定让轴可见

**不强制重命名存量宏**（重命名 120 处 `__WATCOMC__` 的收益小于风险）。
约定作用于**新代码**和**逐处修误用时**：

- (a) 编译器身份：继续用 `__WATCOMC__` / `__GNUC__`，但**只**在
  `#pragma aux`、intrinsic 拼写、C89 怪癖这些真·编译器差异处用。
- (b) 目标架构：继续用 `__i386__` / `__x86_64__` / `__arm__`。
- (c) TU ISA：继续用 `__MMX__` / `__SSE__` / `__SSE2__`，但只表示
  「这个 TU 被允许发什么」，**不**表示「内核在不在构建里」。
- (d) 内核在构建里：**统一到 `LZ_*_EXTERN`**（`LZ_DOT_MMX_EXTERN` 等），
  由 `LZ_*_TU`（构建级）推导。`LZ_HAVE_*` 是 #18 前的遗留，逐个迁走或删。
- (e) 运行时 CPU：保留 `lz_cpu_has_*`。
- (f) 用户档：保留 `LZ_*_FIXED` 等。

**修误用的规则**（这是「清屎山」的实质动作，不是改名）：

> 看到一个 `#if defined(X)`，先问「X 是哪一轴」，再问「这个分支要的是哪一轴」。
> 两个答案不一致，就是误用，改成正确的轴。

---

## 四、迁移路径（分阶段，每阶段用矩阵构建验证）

矩阵构建（impl-matrixbuild 在做）是这一切的验证网：每个阶段的改动，
靠「每个控制开关的非默认值至少建一次」保证不漏。

**阶段 1 —— float 可用性统一（本方案的核心，独立可做）**
- 引入 `LZ_NO_FPU` + 扩展 `lz_soft_float_target()`。
- ARM 构建显式定义 `LZ_NO_FPU=1`（文档化现状，不改变行为）。
- 加一个 `-DLZ_NO_FPU=1` 的控制构建到矩阵（x86 上模拟 no-387，
  断言「定点档被强选」）。
- **保留验证**：ARM `build/arm/build_engine.sh` 构建结果逐位不变。

**阶段 2 —— 误用消除（#21 同型，逐处修）**
- 扫 `__WATCOMC__` 120 处，分类「编译器身份 vs 架构代理」。
- 「架构代理」的改成 `__i386__ && !__SSE_MATH__`（或正确的轴）。
- 每处修完，矩阵构建 + 逐位对拍验证。

**阶段 3 —— 命名统一（LZ_HAVE_* → LZ_*_EXTERN）**
- 逐个迁 `LZ_HAVE_*`（11 个，`LZ_HAVE_SETMODE` 不是内核，另议）。
- 注意 `LZ_HAVE_*_ASM`（Watcom 手写汇编）与 `LZ_*_EXTERN`（gcc extern）
  语义不同，不能机械替换——先理清每个的 gcc/Watcom 对应关系。

**阶段 4 —— dead #ifndef 清扫（已部分完成）**
- `bbc554d` 删两个、`d848ce1` 加闸门。矩阵构建会扫出剩余。

---

## 六、算子排查三线（goal 新增：缺格 / x87 减负 / int→float 去重）

算子正确性已验过，这里只查「缺格」「欠优化（x87 减负）」「重复量化」。

### 线 1：缺格（能实现却没实现）

- **已确认**：gcc 的 128 组内核（`g128` 列）对 Q8/Q41/Q61/Q16 是「not written」，
  `ops.c:5568` 明说「Whether a hand-written gcc version would differ has NOT
  been measured」。这是「能实现却没实现」的明确候选，不是「nothing to offer」
  （T2 的 g128 才是核对过的 nothing-to-offer）。
- **排查**：跑两个构建的 `--kernel-matrix`，对每个「空」格判定
  「指令集不可能（已核对）」vs「not written（能实现）」vs「nothing to offer（核对过）」。
- **产出**：缺格清单，每个带判定和依据。

### 线 2：x87 减负（标量浮点算子的定点/SIMD 化）

- 六个定点档（`--recur/conv/attn/sig/norm/epi-write`）已覆盖主要算子，
  剩 SOFTMAX / EXP / RSQRT 是 x87 标量。
- **排查**：`-DLZ_F32COUNT=1` 构建跑 `--tokens`，读 per-token 的 f32 census
  （site × mul/add/div/cvt/cmp），找「计数高且可定点/SIMD」的 site。
- **产出**：x87 减负清单，每个算子的定点/SIMD 可行性 + 目标机（弱 x87）收益估计。

### 线 3：int→float 去重（定点档位联动的重复量化）

- 72 处 `lz_i32f`/`q8_round` 是 int↔float 边界。
- **排查**：forward.c 的状态流（GDN 状态 → attention 的传递），找「档位 A 的
  定点 int 输出 → 解量化 float → 档位 B 再量化 int」的重复转换。
- **产出**：重复量化清单，每个的「保持 int 贯通」方案。

---

## 七、闸门（不变）

- `build/dead_ifndef_gate.sh`（已接 `make check`）。
- `build/switch_matrix_gate.sh`（impl-matrixbuild 在做）。
- 每个阶段的「逐位对拍」：改前后 `--dump-logits` / `cmp` 不变。
- `LZ_NO_FPU` 的控制构建：定点档被强选 + 无 FPU 路径不崩溃。
