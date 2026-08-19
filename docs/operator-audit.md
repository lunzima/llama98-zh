# 算子排查报告（缺格 / x87 减负 / int→float 去重）

前提：算子正确性已验过，这里只查「缺格」「欠优化（x87 减负）」「重复量化」。
数据来自 `-DLZ_F32COUNT=1` 构建的 `--tokens` census，单 token，ctx=1。

## 线 1：缺格（能实现却没实现）

**核心发现：gcc 的 128 组内核是「not written」。**

`--kernel-matrix` 两个构建的并集：

| 格 | gcc | Watcom | 判定 |
|---|---|---|---|
| mmx / sse2 | x | x | 有 |
| sse | . | . | 指令集不可能（#30 逐条核对过） |
| g128（Q8/Q41/Q61/Q16） | G | G | **已补写（2026-08-20，任务 #12）**，见 `dot128_*_mmx` |
| g128（T2） | . | . | nothing to offer（核对过：发出同四条分开调用一样的序列） |

`ops.c` 原注释所说 *"Whether a hand-written gcc version would differ has NOT
been measured, so this is 'not written', not 'nothing to offer'"* 已经兑现：
gcc 的 128 组内核（`dot128_*_mmx`，`src/ops_kernel_dot_mmx.h`）现已写并入编，
默认走组路径（`LZ_G128_GCC` 旋钮，默认 1 与 Watcom 同形；置 0 退回逐 32 路径）。

g128 的减负机制是「把掩码构造（0x0F/0x03 + 零寄存器）提升到四个子块之外」，
Watcom 的手写汇编做了，gcc 的 intrinsics 版本这次也做了（`z`/`m0f`/`m03` 横跨
四个子块存活）。**逐位一致已验**（`chk=` 与逐 32 路径、与 Watcom 构建全等），
但**收益未量**——铁律三：目标机实测，本机数字不作数。

## 线 2：x87 减负（已定点化，`--scalar-fixed` 旋钮）

float 档 census 里，六个定点档覆盖的算子之外，还剩两个 x87 标量：

| site | float 档 | fixed 档 | 说明 |
|---|---|---|---|
| exp | 624780（11.5%） | 8905（1.2%） | 已定点化：Q20 表 + 残差二阶泰勒（f32 操作 13→7） |
| rsqrt | 4248（0.1%） | 4248（0.6%） | 已定点化：Q23 种子表 + 整数牛顿（f32 操作 9→2~3） |

**已修复（2026-08-20，`113fa90`）**：旋钮 `--scalar-fixed auto|float|fixed`，默认 auto
（开发机落 float=默认关；无 SIMD 弱 x87 目标落 fixed=达成 offload）。逐位一致在
结构上不可达（float 路径的舍入序列整数域复现不了），所以是 Outcome 2（旋钮），
非 bit-identical。census：exp 5213→3269、rsqrt 4248→1606（`--fixed all` 下）。

**关键发现**：q8 模型上 `--scalar-fixed fixed` 单独把 logits rms 动 0.87%，看似远超
算子 ~1e-6 的误差——这是量化管线的边界翻转放大。量过六个档单开的影响：conv 3.37%、
norm 0.92%、attn 0.44%、sig 0.29%、recur 0.36%、epi 0.18%，scalar 在族内不是异常。
「先比同量级再下结论」。

## 线 3：int→float 去重（定点档位联动的重复量化）

总 f32 从 5419392 降到 757821（**7.2 倍**）——这是定点档在无 SSE 目标机上的
价值。但 fixed 档暴露一个反直觉点：

| site | float 档 cvt | fixed 档 cvt | 判定 |
|---|---|---|---|
| recurrence write | 1769472 | **0** | 定点档把 int→float 全消了（这是定点档的价值） |
| sigmoid | 47760 | 47760，但 mul 46080→189360 | **Q15 表解量化的 int→float 重复** |
| conv1d | 0 | 27648 | float→int 转换 |

**核心发现：sigmoid 的 fixed 档反而升到 37.6%（284880 f32）。**

机制是「Q15 表查表（int）→ 乘 1/32767 解量化（int→float）→ 下游再量化
（float→int）」——中间那步解量化是重复的，因为下游（norm/attention 的定点档）
要的也是 int。**修复方向：sigmoid 的 Q15 输出保持 int（Q15 域），下游直接消费，
不经过 float 解量化**。这个去重能再砍掉 fixed 档里最大的一块 f32。

---

## 汇总

| 线 | 发现 | 优先级 |
|---|---|---|
| 1 | gcc 的 g128 缺格（not written，hoist mask 减负）→ **已补写（2026-08-20）**，`chk=` 逐位一致；收益未量 | 中（需目标机实测 gcc 是否更快） |
| 2 | exp / rsqrt 已定点化（`--scalar-fixed` 旋钮，Outcome 2） | 中（目标机弱 x87 上收益大） |
| 3 | sigmoid fixed 的解量化重复 → **已去重（bit-identical，`fb26f31`）** | 高 |

三线均已修复。线 1 见本文件开头「线 1」一节；收益与旋钮（`LZ_G128_GCC`）见
`src/ops_kernel_shared.h` 的注释。
