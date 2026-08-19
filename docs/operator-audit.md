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
| g128（Q8/Q41/Q61/Q16） | . | G | **not written（gcc 缺）** |
| g128（T2） | . | . | nothing to offer（核对过：发出同四条分开调用一样的序列） |

`ops.c:5568` 的注释自己写明了：*"Whether a hand-written gcc version would
differ has NOT been measured, so this is 'not written', not 'nothing to offer'"*。

g128 的减负机制是「把掩码构造（0x0F/0x03 + 零寄存器）提升到四个子块之外」，
Watcom 的手写汇编做了，gcc 的 intrinsics 交给编译器去 hoist。**这是「能实现
却没实现」的明确候选**，修复方向是手写 gcc 的 128 组内核并实测（铁律三：目标机
实测，本机数字不作数）。

## 线 2：x87 减负（未定点化的标量浮点算子）

float 档 census 里，六个定点档覆盖的算子之外，还剩两个 x87 标量：

| site | float 档 | fixed 档 | 说明 |
|---|---|---|---|
| exp | 624780（11.5%） | 8905（1.2%） | fixed 档下 sigmoid 的 exp 被 Q15 表替代，剩 softmax 的 exp 还是查表 x87 |
| rsqrt | 4248（0.1%） | 4248（0.6%） | lz_rsqrt 魔数+Newton，两档都未定点化 |

**修复方向**：exp 和 rsqrt 的定点化。exp 已有 Q15 表的先例（`sig_build`），
可推广到 softmax 的 exp；rsqrt 可用整数牛顿迭代。目标机（Cyrix 6x86MX /
IDT WinChip 的 x87 弱）上收益最大。

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
| 1 | gcc 的 g128 缺格（not written，hoist mask 减负） | 中（需实测手写 gcc 是否更快） |
| 2 | exp / rsqrt 未定点化（x87 标量） | 中（目标机弱 x87 上收益大） |
| 3 | sigmoid fixed 的解量化重复（37.6% f32 可去重） | **高**（去重能再砍 fixed 档最大块） |

三线都是「排查产出」，修复是后续（各自需独立设计 + 矩阵构建验证）。
