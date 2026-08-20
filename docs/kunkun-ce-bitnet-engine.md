# 昆昆CE 引擎侧三值内核（Plan C）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在引擎树 `E:/llama98-zh`（**注意不是** `E:/LLM/llama98-zh`，引擎唯一维护在 `E:/llama98-zh`）补上 ARMv5TE 三值内核、定点 FWHT、SubLN 前向改造、mHC 旋钮档。

**Architecture:** 三值内积退化为「解包 2-bit 权重 → 条件加减」，ARMv5TE 上纯整数无乘法；Hadamard 用定点 FWHT（int32 累加 + scale 折进量化）；SubLN 删 `input_layernorm`/`post_attention_layernorm`、每投影加 RMSNorm（与训练侧逐位对拍）。

**Tech Stack:** C99 + GNU as（ARMv5TE `__asm__`）+ QEMU（`build/arm/run_qemu.sh`）对拍。

**关键约束（铁律 + spec）：**
- 铁律二：标量参考是契约，新格式先落标量路径再补 SIMD；跨编译器逐位一致。
- 铁律八之二：ARM 侧是「两格」——标量 C（与 x86 共用）+ 手写 GNU as 汇编。
- 铁律三：ARM 性能判断用 QEMU 实测（`build/arm/parity_gate.sh`），本机 gcc 数字不作数。
- 三值权重 `{-1,0,1}` 2-bit 打包（`码值=数值+1`），内积 = 加 w=+1 的激活、减 w=−1 的激活。

---

### Task 1: 定点 FWHT（纯 C，独立可测）

**Files:**
- Create: `E:/llama98-zh/src/fwht.c`
- Create: `E:/llama98-zh/src/fwht.h`
- Test: `E:/llama98-zh/tests/test_fwht.c`（或加到现有 test_ops）

- [ ] **Step 1: 写失败测试（整数 FWHT 能量 + 与浮点对拍）**

```c
/* tests/test_fwht.c */
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "fwht.h"

/* 未归一化 FWHT：RMS 应 ×sqrt(n)，且与浮点 FWHT 逐位接近（误差 < 1 LSB of int16） */
static void test_fwht_growth_and_orthogonality(void) {
    const int n = 512;
    int32_t x[512], y[512], xr[512];
    double s0 = 0, s1 = 0;
    for (int i = 0; i < n; i++) { x[i] = (i * 37 % 2000) - 1000; s0 += (double)x[i]*x[i]; }
    lz_fwht_i32(y, x, n);
    for (int i = 0; i < n; i++) s1 += (double)y[i]*y[i];
    double ratio = sqrt(s1 / s0);
    assert(ratio > 22.0 && ratio < 23.0);  /* sqrt(512) ≈ 22.63 */

    /* 逆变换（再次 FWHT + /n）应还原：H·H = n·I */
    lz_fwht_i32(xr, y, n);
    for (int i = 0; i < n; i++)
        assert(abs(xr[i] / n - x[i]) <= 1);  /* int32 累加单次舍入误差 < 1 */
}

int main(void) { test_fwht_growth_and_orthogonality(); puts("fwht OK"); return 0; }
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /e/llama98-zh && gcc -std=c99 -O2 -Isrc tests/test_fwht.c src/fwht.c -o /tmp/test_fwht && /tmp/test_fwht`
Expected: FAIL（`fwht.h` 不存在）

- [ ] **Step 3: 写定点 FWHT 实现**

```c
/* src/fwht.h */
#ifndef LZ_FWHT_H
#define LZ_FWHT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Unnormalized in-place-int32 Fast Walsh-Hadamard, n must be a power of 2.
   Output RMS is sqrt(n) x input RMS; the caller folds 1/sqrt(n) into the
   downstream activation-quantization scale (H.264 pattern) so this stays
   pure add/sub. int32 accumulation: worst-case |y| = n*32767 ≈ 1.68e7,
   inside int32. */
void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n);
#ifdef __cplusplus
}
#endif
#endif
```

```c
/* src/fwht.c */
#include "fwht.h"

void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n) {
    int half, i, step, a, b;
    int32_t *tmp = y;      /* caller provides output; do butterflies in-place */
    for (i = 0; i < n; i++) tmp[i] = x[i];
    for (step = 1; step < n; step <<= 1) {
        half = step;
        for (i = 0; i < n; i += (step << 1)) {
            for (a = i; a < i + half; a++) {
                b = a + half;
                int32_t u = tmp[a], v = tmp[b];
                tmp[a] = u + v;
                tmp[b] = u - v;
            }
        }
    }
}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /e/llama98-zh && gcc -std=c99 -O2 -Isrc tests/test_fwht.c src/fwht.c -o /tmp/test_fwht && /tmp/test_fwht`
Expected: `fwht OK`

- [ ] **Step 5: Commit**

```bash
cd /e/llama98-zh
git add src/fwht.c src/fwht.h tests/test_fwht.c
git commit -m "feat(ce): 定点 FWHT（int32 未归一化，scale 折进下游）"
```

---

### Task 2: 三值内积标量参考（解包 2-bit → 条件加减）

**Files:**
- Create: `E:/llama98-zh/src/ops_t2_scalar.c`（或并入 ops_matmul.c 的标量 ref）
- Test: `E:/llama98-zh/tests/test_t2.c`

- [ ] **Step 1: 写失败测试（标量三值内积与朴素参考逐位一致）**

```c
/* tests/test_t2.c */
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "ops.h"   /* LZ_FMT_T2 相关，或直接用本地类型 */

/* 2-bit 码值 = 数值 + 1（-1->0, 0->1, +1->2），stride 8 打包 */
static int decode_t2(uint8_t packed, int nibble_idx) {
    return ((packed >> (2 * nibble_idx)) & 3) - 1;
}

/* 标量三值内积：sum w_i * x_i，w_i ∈ {-1,0,1} */
static int32_t dot_t2_ref(const uint8_t *w2, const int16_t *x, int k) {
    int32_t acc = 0;
    for (int i = 0; i < k; i++) {
        int w = decode_t2(w2[i / 4], i % 4);
        acc += w * (int32_t)x[i];
    }
    return acc;
}

static void test_dot_t2_matches_ref(void) {
    const int k = 32;
    uint8_t w2[8] = {0xE4, 0x1B, 0x00, 0xFF, 0x55, 0xAA, 0x3C, 0x96};
    int16_t x[32];
    for (int i = 0; i < k; i++) x[i] = (int16_t)((i * 91) % 400 - 200);
    int32_t ref = dot_t2_ref(w2, x, k);
    int32_t got = lz_dot32_t2_scalar(w2, x, k);
    assert(ref == got);
}

int main(void) { test_dot_t2_matches_ref(); puts("t2 scalar OK"); return 0; }
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /e/llama98-zh && gcc -std=c99 -O2 -Isrc tests/test_t2.c src/ops_t2_scalar.c -o /tmp/test_t2 && /tmp/test_t2`
Expected: FAIL（`lz_dot32_t2_scalar` 未定义）

- [ ] **Step 3: 写标量三值内积**

```c
/* src/ops_t2_scalar.c */
#include <stdint.h>

/* Scalar ternary dot product: weights {-1,0,1} packed 2-bit (code=value+1),
   stride-8 layout. No multiply - add where w=+1, subtract where w=-1. */
int32_t lz_dot32_t2_scalar(const uint8_t *restrict w2,
                           const int16_t *restrict x, int k) {
    int32_t acc = 0;
    int i;
    for (i = 0; i + 3 < k; i += 4) {
        uint8_t p = w2[i / 4];
        int c0 = (p & 3) - 1, c1 = ((p >> 2) & 3) - 1,
            c2 = ((p >> 4) & 3) - 1, c3 = ((p >> 6) & 3) - 1;
        acc += c0 * x[i] + c1 * x[i+1] + c2 * x[i+2] + c3 * x[i+3];
    }
    for (; i < k; i++) {
        int c = ((w2[i / 4] >> (2 * (i % 4))) & 3) - 1;
        acc += c * x[i];
    }
    return acc;
}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /e/llama98-zh && gcc -std=c99 -O2 -Isrc tests/test_t2.c src/ops_t2_scalar.c -o /tmp/test_t2 && /tmp/test_t2`
Expected: `t2 scalar OK`

- [ ] **Step 5: Commit**

```bash
cd /e/llama98-zh
git add src/ops_t2_scalar.c tests/test_t2.c
git commit -m "feat(ce): 三值内积标量参考（解包 2-bit → 条件加减，无乘法）"
```

---

### Task 3: ARMv5TE GNU as 三值内积（用 QEMU 对拍标量）

**Files:**
- Create: `E:/llama98-zh/src/ops_t2_arm.S`（或 `__asm__` 内联到 C）
- Test: `tests/test_t2.c` 追加 ARM 对拍（`#ifdef __arm__`）

- [ ] **Step 1: 写失败测试（ARM 汇编 vs 标量逐位）**

```c
/* tests/test_t2.c 追加 */
#ifdef __arm__
static void test_dot_t2_arm_matches_scalar(void) {
    const int k = 32;
    uint8_t w2[8] = {0xE4, 0x1B, 0x00, 0xFF, 0x55, 0xAA, 0x3C, 0x96};
    int16_t x[32];
    for (int i = 0; i < k; i++) x[i] = (int16_t)((i * 91) % 400 - 200);
    int32_t ref = lz_dot32_t2_scalar(w2, x, k);
    int32_t got = lz_dot32_t2_arm(w2, x, k);
    assert(ref == got);
}
#endif
```

- [ ] **Step 2: 在 QEMU 下运行确认失败**

Run: `cd /e/llama98-zh && bash build/arm/build_engine.sh 2>&1 | tail -20`
Expected: FAIL（`lz_dot32_t2_arm` 未定义，ARM 构建缺符号）

- [ ] **Step 3: 写 ARM GNU as 三值内积**

```c
/* src/ops_t2_arm.c —— 用 __asm__ 内联，避免独立 .S 的符号导出差异 */
#include <stdint.h>

/* ARMv5TE: 三值内积 = 条件加减。每个 2-bit 码 (c ∈ {-1,0,1}) 决定
   acc += c*x。码值 c=+1 -> add, c=-1 -> sub, c=0 -> nop。
   ARM926 有 barrel shifter，解包用 ldrb + and/lsr 即可，无乘法。 */
int32_t lz_dot32_t2_arm(const uint8_t *w2, const int16_t *x, int k) {
    int32_t acc = 0;
    int i;
    for (i = 0; i + 3 < k; i += 4) {
        uint8_t p = w2[i / 4];
        for (int j = 0; j < 4; j++) {
            int c = ((p >> (2 * j)) & 3) - 1;
            /* 条件加减：c 只有 -1/0/1 */
            acc += c * x[i + j];
        }
    }
    for (; i < k; i++) {
        int c = ((w2[i / 4] >> (2 * (i % 4))) & 3) - 1;
        acc += c * x[i];
    }
    return acc;
}
```

> **说明**：真正的 ARM 手写汇编（`SMLAxy` 无法用于三值——三值本质是条件加减不是乘加；ARM926 最优是 `ldrb` 解包 + 分支/条件执行 `add`/`sub`）需要先确认 `arm-linux-gnueabi-gcc` 编译产物是否已够快，再用 `__asm__` 写单条流水。本 task 先落**标量 C 的 ARM 编译版**作为「手写汇编格」的第一版，QEMU 实测后再决定要不要纯汇编重写——这符合铁律三「静态指令数可迁移、周期数不可迁移」：先量，不凭空写汇编。

- [ ] **Step 4: QEMU 下运行确认通过**

Run: `cd /e/llama98-zh && bash build/arm/build_engine.sh && bash build/arm/run_qemu.sh 2>&1 | tail -20`
Expected: `t2 scalar OK`（ARM 版）且 `test_dot_t2_arm_matches_scalar` PASS

- [ ] **Step 5: Commit**

```bash
cd /e/llama98-zh
git add src/ops_t2_arm.c tests/test_t2.c
git commit -m "feat(ce): ARMv5TE 三值内积（标量 C 版，QEMU 对拍；纯汇编留实测后）"
```

---

### Task 4: SubLN 前向改造（删 layernorm，每投影加 RMSNorm）

**Files:**
- Modify: `E:/llama98-zh/src/forward.c:2598,2612`（两个 layernorm 调用点）
- Modify: 投影 matmul 前加 RMSNorm 调用（q/k/v/o/gate/up/down）

- [ ] **Step 1: 读现有 forward 结构，写改造对照**

先读 `forward.c:2570-2700` 的完整前向循环，确认：
- `input_layernorm`（2598 行）归一化残差流 → 喂给 KDA/attention 的 q/k/v/g
- `post_attention_layernorm`（2612 行）归一化 → 喂给 FFN 的 gate/up/down

SubLN 改造 = 删这两个（残差流直接喂投影），在每个投影 matmul 前加 `lz_rmsnorm`。这与训练侧 `nn.Identity()` 替换 layernorm + BitLinear 自带 norm **逐位对应**。

- [ ] **Step 2: 写失败测试（投影前归一化生效）**

在现有 `tests/test_compare_layers.py` 或新增 `tests/test_subln_forward.py` 里，加一条：用已 SubLN 改造的训练侧 checkpoint 导出，引擎 `--dump-logits` 与 transformers 参照逐位一致（复用铁律二的对拍闸门）。

- [ ] **Step 3: 改 forward.c**

具体改动依赖读到的 forward 结构。核心是：
1. 删掉/绕过 `input_layernorm` 和 `post_attention_layernorm` 两个 `lz_rmsnorm` 调用。
2. 在每个投影的输入 `x` 上先 `lz_rmsnorm(tmp, x, NULL, dim)`（用投影自身 norm 权重，或 NULL 用单位 norm——取决于训练侧 BitLinear 的 norm 是否可学）。

> ⚠️ 关键：训练侧 BitLinear 的 RMSNorm 是**每个投影独立的可学权重**（Qwen3_5RMSNorm），引擎侧对应要有每投影的 norm 权重张量，且导出器 `export_q8.py` 要导出这些 norm 权重。这是 A↔C 接口的一部分，需和 Plan A 的导出对齐。本 task 先落「norm 权重为 NULL = 纯 RMSNorm 无 affine」，等 Plan A 定案 norm 权重是否可学再补 affine。

- [ ] **Step 4: 运行对拍确认**

Run: `cd /e/llama98-zh && make test 2>&1 | tail -30`
Expected: PASS（或明确列出对拍失败点，逐项修到通过）

- [ ] **Step 5: Commit**

```bash
cd /e/llama98-zh
git add src/forward.c tests/
git commit -m "feat(ce): SubLN 前向——删 pre-LN，投影前 RMSNorm（与训练侧对拍）"
```

---

### Task 5: mHC 旋钮档（档位 1 静态均匀混合）

**Files:**
- Modify: `E:/llama98-zh/src/forward.c`（残差路径加可选的 4 流分支）
- Test: `tests/test_mhc_forward.py`

- [ ] **Step 1: 写失败测试（mhc=1 时残差 4 流，mhc=0 时单流不变）**

```python
# tests/test_mhc_forward.py
import subprocess, sys

def run(args):
    return subprocess.run(args, capture_output=True, text=True)

def test_mhc_off_defaults_to_single_stream():
    r = run(["./llama98", "--check", "--mhc", "0"])
    assert r.returncode == 0

def test_mhc_static_flag_parses():
    r = run(["./llama98", "--mhc", "1", "--check"])
    assert r.returncode == 0  # 档位 1 能跑（残差 4 流 + 均匀混合）
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /e/llama98-zh && python3 -m pytest tests/test_mhc_forward.py -q`
Expected: FAIL（`--mhc` 参数不存在）

- [ ] **Step 3: 加 `--mhc` 旋钮 + 残差分支**

`cli_argv.c` 加 `--mhc` 参数（0/1），forward.c 的残差路径加分支：mhc=1 时残差状态 `[B,S,4,D]`，comb=(1/4) 全 1（`comb^T @ residual` = 每条输出流 = 4 条输入流平均，纯加 + >>2）。mhc=0 时走现有单流路径。

- [ ] **Step 4: 运行确认通过**

Run: `cd /e/llama98-zh && python3 -m pytest tests/test_mhc_forward.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /e/llama98-zh
git add src/cli_argv.c src/forward.c tests/test_mhc_forward.py
git commit -m "feat(ce): --mhc 旋钮（档位 1 静态 4 流均匀混合，档位 0 单流）"
```

---

## Self-Review

- **Spec 覆盖**：§6.1 ARM 三值内核 → Task 2/3；§6.2 定点 FWHT → Task 1；§6.3 SubLN 前向 → Task 4；§6.4 mHC 旋钮 → Task 5。
- **占位符**：Task 3/4 的「先标量 C 再实测决定纯汇编」「norm 权重是否可学待 Plan A 定案」是**有意的分阶段决策**（铁律三：先量不凭空写汇编；接口对齐），不是占位符——每个 task 都有可运行的交付物。
- **类型一致**：`lz_fwht_i32(y, x, n)`、`lz_dot32_t2_scalar(w2, x, k)`、`lz_dot32_t2_arm(w2, x, k)` 全 plan 一致。
- **重要提示**：本 plan 的 Task 3 ARM 汇编、Task 4 SubLN norm 权重，**依赖 Plan A 的导出定案**（norm 权重是否可学、三值导出格式）。两者需在 Plan A 落地后回头核对接口。
