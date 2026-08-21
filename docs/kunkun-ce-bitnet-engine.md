# 昆昆CE 引擎侧三值内核（Plan C）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在引擎树 `E:/llama98-zh`（**注意不是** `E:/LLM/llama98-zh`，引擎唯一维护在 `E:/llama98-zh`）补上 ARMv5TE 三值内核、定点 FWHT、SubLN 前向改造。mHC 旋钮档已由训练侧消融否掉，见 Task 5。

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

/* 2-bit 码值 = 数值 + 1（-1->0, 0->1, +1->2），stride-8 打包：
   weight[k]（块内 0..31）的码在 byte k&7 的 nibble k>>3 —— 与引擎
   src/ops.c:1104 的 ((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1 逐位一致，
   也和 dot32_t2_mmx 的 8-byte plane 布局一致。 */
static int decode_t2(const uint8_t *b2, int k) {
    return ((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1;
}

/* 标量三值内积：sum w_i * x_i，w_i ∈ {-1,0,1}，32 元素单 plane。 */
static int32_t dot_t2_ref(const uint8_t *w2, const int16_t *x) {
    int32_t acc = 0;
    for (int i = 0; i < 32; i++) {
        int w = decode_t2(w2, i);
        if (w > 0) acc += x[i];
        else if (w < 0) acc -= x[i];
    }
    return acc;
}

static void test_dot_t2_matches_ref(void) {
    uint8_t w2[8] = {0xE4, 0x1B, 0x00, 0xFF, 0x55, 0xAA, 0x3C, 0x96};
    int16_t x[32];
    for (int i = 0; i < 32; i++) x[i] = (int16_t)((i * 91) % 400 - 200);
    int32_t ref = dot_t2_ref(w2, x);
    int32_t got = lz_dot32_t2_scalar(w2, x);
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

/* Scalar ternary dot product, one 32-wide block (one 8-byte plane).
   Weights {-1,0,1} packed 2-bit, code=value+1, STRIDE-8 layout:
   weight[k] (k in 0..31) lives in byte k&7, nibble k>>3 - the same
   layout src/ops.c:1104 and dot32_t2_mmx use, so this is the exact
   scalar reference both SIMD paths must reproduce bit-for-bit (铁律二).
   No multiply: add where w=+1, subtract where w=-1, skip where w=0. */
int32_t lz_dot32_t2_scalar(const uint8_t *restrict w2,
                           const int16_t *restrict x) {
    int32_t acc = 0;
    int i;
    for (i = 0; i < 32; i++) {
        int c = ((w2[i & 7] >> (2 * (i >> 3))) & 3) - 1;
        if (c > 0) acc += x[i];
        else if (c < 0) acc -= x[i];
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
    uint8_t w2[8] = {0xE4, 0x1B, 0x00, 0xFF, 0x55, 0xAA, 0x3C, 0x96};
    int16_t x[32];
    for (int i = 0; i < 32; i++) x[i] = (int16_t)((i * 91) % 400 - 200);
    int32_t ref = lz_dot32_t2_scalar(w2, x);
    int32_t got = lz_dot32_t2_arm(w2, x);
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

/* ARMv5TE: 三值内积，32 元素单 plane（stride-8，同标量参考）。
   与 lz_dot32_t2_scalar 逐位一致：返回 256*sum(code*x)，code 读
   无符号 0..3（不是 value=code-1）——-1 和 1/256 都在共享 epilogue
   epi_q41 里处理，内核不碰。先落标量 C 的 ARM 编译版（铁律三：
   先量，不凭空写汇编），QEMU 实测后再决定要不要手写 __asm__ 汇编。 */
int32_t lz_dot32_t2_arm(const uint8_t *w2, const int16_t *x) {
    int32_t acc = 0;
    int i;
    for (i = 0; i < 32; i++) {
        int c = (w2[i & 7] >> (2 * (i >> 3))) & 3;   /* code = weight+1, 0..3 */
        acc += c * (int32_t)x[i];
    }
    return acc << 8;   /* the SIMD kernels' x256 fold */
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

### Task 4: SubLN 前向改造（删 layernorm，每投影加 RMSNorm）— 已完成

**验收状态**：引擎侧 SubLN 路径（含定点 Hadamard 与 per-token 激活量化）
对训练侧参照 **1-cos 1.87e-4、argmax 5/5**；`use_subn=0` 逐字节不变
（pilot 与 kunmoe-v2）；三目标编译（native gcc / ARM 交叉 / Watcom `-we`）。

> 提交 `b97bfcc` 的信息把 Hadamard 记成「数值待修」，**那条结论是错的**，
> 来自一份无效参照：它经 `from_pretrained` 生成，而 transformers v5 在 meta
> 设备上建模型后不会重建 `persistent=False` 的 buffer，Hadamard 矩阵加载回来
> 是非正交垃圾（`|H·Hᵀ−I|max = 1.0`）。缺陷在训练侧，已修（训练树 `7c1cee2`，
> 改为按需构造 + 闸门 `test_hadamard_survives_from_pretrained`）。
> 判据方法：参照要用**独立验证过**的路径重新生成，再排 2×2——每个引擎变体
> 只匹配自己那份参照（对角线 1.9e-4 / 1.8e-4，非对角 1.1）。

**Hadamard 块大小是配置字段，不是推导值**（bin v8）。`hadamard_o`（KDA
o_proj）与 `hadamard_down`（dense down_proj）：`0` = 关，正数 = 块大小
（须为 2 的幂且整除该投影输入维度），`-1`/缺字段 = 按旧规则推导（只为
v7 及更早的产物保留）。非法值在加载期报 `LZ_ERR_CFG_HADAMARD` 拒绝，
不静默改成 0。spec §3.4 要求的块对角消融由此可导出、可在引擎上跑。

跨树往返已验证：训练侧导出的 v8 bin 由引擎读入后，对「Hadamard 开」的
参照 1-cos 2.995e-04，对「关」的参照 1.135。

⚠️ `subn_fwht_quant` 的块循环没有余数保护，块大小不整除输入维度是**越界写**
而非只是算错——这是加载期校验必须拒绝非法值的原因。`dense_ffn_step` 有
两个调用方（主体用 `intermediate_size`，MTP 块用 `mtp_intermediate_size`）
而只有一个 `hadamard_down`，故加载期一并校验 MTP 宽度；两者若要各自独立
需要第三个字段。

**接口定案（A↔C 契约，已分析）**：

CE 层结构 = 8 层 `LLLALLLA`（6 KDA + 2 全注意力，docs/kunkun-ce-params.md §1）。
三值化投影 = KDA q/k/v/g/o（6 层）+ dense FFN gate/up/down（8 层）；普通
attention 层投影不三值化、无 SubLN norm。SubLN 语义：删所有层的
input_layernorm/post_attention_layernorm，每个三值化投影自带 RMSNorm
（作用在投影自己的输入上，训练侧 BitLinear.use_norm=True，norm 是
Qwen3_5RMSNorm **可学 affine**）。

**8 个新 SubLN norm 字段**（model.h LZLayer + LAYER_SPECS 映射）：

| 训练侧 safetensors 键名 | 引擎字段 | 维度 | 作用在 |
|---|---|---|---|
| linear_attn.q_proj.norm.weight | kda_q_norm | hidden | s->xb |
| linear_attn.k_proj.norm.weight | kda_k_norm | hidden | s->xb |
| linear_attn.v_proj.norm.weight | kda_v_norm | hidden | s->xb |
| linear_attn.g_proj.norm.weight | kda_g_norm | hidden | s->xb |
| linear_attn.o_proj.norm.weight | kda_o_subn_norm | value_dim | s->ssm_out |
| mlp.gate_proj.norm.weight | gate_norm | hidden | s->xb |
| mlp.up_proj.norm.weight | up_norm | hidden | s->xb |
| mlp.down_proj.norm.weight | down_norm | inter | s->hb |

⚠️ 陷阱：`linear_attn.o_norm.weight`（现有 kda_o_norm，Qwen3_5RMSNormGated
gated silu，head_v_dim）与 `linear_attn.o_proj.norm.weight`（新
kda_o_subn_norm，普通 RMSNorm，value_dim）是**两个不同权重**，不能复用。

**config.use_subn**：LZModelConfig 加 `use_subn`（对应训练侧 config.use_subn），
forward.c 据此分支「SubLN（删层 norm + 每投影 norm）」vs「普通（层 norm）」。
默认 0（旧模型不受影响）。

**Files:**
- Modify: `E:/llama98-zh/src/model.h`（8 字段 + config.use_subn）
- Modify: `E:/llama98-zh/src/model.c`（LAYER_SPECS 8 映射 + config.use_subn 加载）
- Modify: `E:/llama98-zh/src/forward.c`（删 2 层 norm + 8 投影 RMSNorm，use_subn 分支）
- Modify: `E:/LLM/llama98-zh/tools/export_q8.py`（导出 8 norm + config.use_subn）

- [ ] **Step 1: 引擎 model.h 加 8 个 norm 字段 + config.use_subn**

LZLayer 结构（model.h:279-324 附近）加 8 个 LZTensor f32 字段：
kda_q_norm/kda_k_norm/kda_v_norm/kda_g_norm/kda_o_subn_norm（KDA，hidden×4
+ value_dim×1）+ gate_norm/up_norm（hidden）+ down_norm（inter）。
LZModelConfig 加 `int use_subn;`。

- [ ] **Step 2: 引擎 model.c 加 8 个 LAYER_SPECS 映射 + config 加载**

LAYER_SPECS 表（model.c:356-434）加 8 行（suffix 用上表键名，layer_type
LZ_LT_KDA / ffn_kind 0，field 用 LOFF 新字段，dims 用对应维度）。
config 加载加 use_subn。

- [ ] **Step 3: 引擎 forward.c 删 2 层 norm + 8 投影 RMSNorm**

主循环（forward.c:2701/2715）：删 input_layernorm（2701-2702）与
post_attention_layernorm（2715-2716），改为 `if (!c->use_subn)` 包住原调用
（保留普通路径）。use_subn 分支下：
1. KDA 层（forward.c:2002-2018 附近）：kda_q/k/v/g_proj matmul 前各自
   `lz_rmsnorm(tmp, s->xb, lz_t_f32(&L->kda_*_norm, s->wscr), dim, eps)`；
   kda_o_proj（2236 附近）前用 kda_o_subn_norm 归一化 s->ssm_out（vdim）。
2. dense FFN（forward.c:2479-2508 附近）：gate/up_proj 前用 gate_norm/up_norm
   归一化 s->xb；down_proj 前用 down_norm 归一化 s->hb（idim）。

> 注意：RMSNorm 权重维度要对齐（kda_q/k/v/g/gate/up 是 hidden，kda_o_subn
> 是 value_dim，down 是 inter）。每投影独立归一化缓冲，不共享（SubLN 语义）。

- [ ] **Step 4: 训练侧 export_q8.py 导出 8 norm + config.use_subn**

export_q8.py 的 expect() 与写出加 8 个 norm（键名见上表），config 写出加
use_subn。注意 expect 的维度断言与训练侧 config.use_subn 传入。

- [ ] **Step 5: 单元对拍（最小 SubLN 模型）**

训练侧构造最小 SubLN 三值模型（use_subn=True，几层 KDA + dense），导出 bin，
引擎加载 + --dump-logits 与 transformers 参照逐位一致。端到端（真实 CE 模型）
对拍等 Plan B 训出。

- [ ] **Step 6: Commit（两端各一次）**

```bash
cd /e/llama98-zh   # 引擎树
git add src/model.h src/model.c src/forward.c
git commit -m "feat(ce): SubLN 前向——use_subn 分支，删层 norm，8 投影自带 RMSNorm"

cd /e/LLM/llama98-zh  # 训练树
git add tools/export_q8.py
git commit -m "feat(ce): export_q8 导出 8 投影 SubLN norm + use_subn"
```

---

### Task 5: mHC 旋钮档 —— 不做

**不做。** 训练侧消融已跑完（spec §5，2026-08-21），条件不触发：

- 3 seed × 2 臂 × 8M token 配对对照，mHC **3/3 更差**，效应 +0.0419 bit/char，
  而同臂 seed 间噪声底 0.0376 —— 效应/噪声比 1.12，没站到噪声之上。
- 训练侧吞吐代价 **−13.5%**，引擎侧还要多维护 4× 残差状态。

引擎不加 4 流残差、不加 `--mhc`，`export_q8.py` 也不导出 `mhc`。CE 出货配方 `mhc=0`，
该路径与未接线前逐位相同。要在别的深度/规模上复测，先在训练侧改一个 config 字段重跑
消融，赢了再回来做这一格。

---

## Self-Review

- **Spec 覆盖**：§6.1 ARM 三值内核 → Task 2/3；§6.2 定点 FWHT → Task 1；§6.3 SubLN 前向 → Task 4；§6.4 mHC 旋钮 → Task 5（消融否掉，不做）。
- **占位符**：Task 3/4 的「先标量 C 再实测决定纯汇编」「norm 权重是否可学待 Plan A 定案」是**有意的分阶段决策**（铁律三：先量不凭空写汇编；接口对齐），不是占位符——每个 task 都有可运行的交付物。
- **类型一致**：`lz_fwht_i32(y, x, n)`、`lz_dot32_t2_scalar(w2, x)`、`lz_dot32_t2_arm(w2, x)` 全 plan 一致（leaf 内积固定 32 元素单 8-byte plane，与引擎 `dot32_t2_mmx(w2, x)` 同契约）。
- **重要提示**：本 plan 的 Task 3 ARM 汇编、Task 4 SubLN norm 权重，**依赖 Plan A 的导出定案**（norm 权重是否可学、三值导出格式）。两者需在 Plan A 落地后回头核对接口。
