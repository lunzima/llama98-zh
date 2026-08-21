# 定点档管线边界图(2026-08-21,前提变更后重做)

分支 `feat/attn-conv-fixed-tiers`。v1(commit `e0215a1`)按"哪条边界因为
tier 可独立选择而是 float"分类;本次前提变更(用户已定:出货无条件定点,
不是速度判据)之后,读法整体倒过来,重做。

## 0. 分类框架

v1 的核心结论——"没有一条边界仅因为相邻 tier 设置不一致才物化成
float"——在代码层面仍然成立,但现在的含义反过来了:既然物化成 float
从不是"因为组合不一致才需要",那它就只有另一个可能的理由:**这一步的
下游消费者(邻居 tier)过去可以被独立选成 float**,所以生产者"以防万一"
把结果吐成 float。前提变更本身就删掉了这唯一理由——出货配置里再也没有
谁的邻居是 float 了。于是对每条边界要问的问题变了:

1. **数据强制(data-forced)**:float 是因为一个具体、可指认的**数据
   事实**——权重 dtype、残差流跨层的动态范围、某个算法实现只接受 float
   输入——而不是"下游可能是 float"这种可能性。这类是真正的硬约束,
   拿掉它需要一次结构性改动(下面逐条给出改动需要什么)。
2. **可选强制(optionality-forced)**:float 只是因为旧设计允许邻居
   tier 独立选择,所以这一步保留了一个"两边都能接"的公共接口。前提
   变更之后,这类物化没有存在理由了——但**删掉它、把 int 值贯通过去
   是否值得,是下一阶段的工程题,本文档只负责标出清单,不在本次任务里
   实现**(团队负责人原话:"I am not asking you to implement the
   int-carrying pipeline in this task")。

## 1. 边界逐条重分类

| 边界 | v1 分类 | 重分类 | 依据 |
|---|---|---|---|
| B1 残差流 `x`(进 norm) | structurally float | **数据强制** | 见 §2,RMS 跨 8 层实测增长约 3.9 倍,norm 输出本身被 RMSNorm 重新拉回稳定量级 |
| B2 norm 输出→quantize | 两档都有 | **可选强制(fixed 档),已证明可行(int 档)** | fixed 档的 float 物化只是"以防某个 sibling 是 F32"——见 §3,该假设在出货配方上不成立 |
| B2a SubLN Hadamard(`subn_fwht_quant`) | 数据强制反例 | **仍是数据强制** | 见 §4,FWHT 实现本身只接受 float 输入,是算法层面的硬约束,不是 tier 可选性的产物 |
| B4 matmul 权重分派(F32 fallback) | 唯一硬分支 | **权重 dtype 决定,与 write-tier 正交,但在出货配方上不生效** | 见 §3,`tools/quant_recipe.py` 的 ROLES 表没有任何投影角色落在 F32 |
| B5 matmul epilogue(`epi_fixed`) | structurally float | **可选强制,且已知怎么拆** | 见 §5,`epi_fixed` 已经在整数域算出 `(s, e)`,转 float 只是因为调用方不知道下一步是谁 |
| B6 epilogue→{conv\|recur\|attn} | structurally float,与 epi 取值无关 | **可选强制** | conv/recur/attn 的 fixed 档已经在内部维持定点表示,只是没有对外export——一旦下游不再可能是 float,没有理由回落 |
| B7/B8 sig/scalar 内部定点 + norm/attn 的 int 融合 | 部分 structurally float | **可选强制** | `lz_sigmoid`/`lz_exp`/`lz_rsqrt` 的 fixed 档本身是查表+整数算术,`float -> float` 签名只是历史接口约定 |
| B9 matmul 输出→残差加 | structurally float | **数据强制(通过 B1)** | 残差加本身数据强制,epilogue 侧沿用 B5 的可选强制结论 |
| SwiGLU 融合(`lz_quantize_q8_silu`) | 未在 v1 讨论 | **可选强制,且已知两个具体实现坑** | 见 §6 |

## 2. 残差流:实测数据强制

工具:`.prof/wresidgrow.c`(本地探针,`.prof/` 按仓库既有约定不进 git),
tap "an"/"pn"(两个 trunk norm 的输出,即 `s->xb`)与 "res"(残差加之后
的 `s->x`,`forward.c:3201`/`3219` 的 `LZ_TAP("res", l, s->x, dim)`)。
kmr20,`build/norm_int_gate_tokens_zh.txt`(603 token),8 层:

```
layer  an(absmax/rms)         pn(absmax/rms)         res(absmax/rms)
  0       8.6048/  1.2588      6.8548/  0.8505      1.0168/  0.0543
  1      18.2024/  1.3753      8.9709/  0.9970      1.2188/  0.0654
  2      20.1495/  1.4525     10.6427/  1.0726      2.3680/  0.0826
  3      18.3973/  1.3557      9.5382/  1.1292      2.5544/  0.1037
  4      15.1922/  1.6454      9.4988/  1.2283      2.5366/  0.1471
  5      16.9998/  1.7078      9.2508/  1.2561      2.9891/  0.1831
  6      16.7263/  1.4815     12.7704/  1.2409      3.3409/  0.2104
  7      12.2877/  1.3815     10.1953/  1.2534      2.3268/  0.2142
```

`an`/`pn`(norm 的输出)全程停在一个约 2 倍的窄带内(rms 0.85–1.71)——
RMSNorm 每次都把它拉回单位量级乘学习到的权重,这是它的定义,不是巧合。
`res`(残差流本身,norm 的输入)的 rms 从 0.0543 单调涨到 0.2142,**8 层
涨了约 3.9 倍**,且没有任何机制让它停下来——这条模型只有 8 层,层数更多
的模型(30+ 层是常见规模)按同一趋势会需要远大于 8 位的动态范围。

**这就是残差加(`s->x[i] += s->xb2[i]`)必须留在 float 的数据理由**:
一个静态的定点 Q 格式要么为晚期层的量级留够空间(早期层精度浪费),
要么为早期层校准(晚期层溢出)。RMSNorm 把它下游的一切都重新拉回稳定
区间,所以"下游"不需要宽动态范围;残差流本身**结构性地**绕过了 norm,
这正是它是"数据强制"而不是"可选强制"的原因。

**去掉这条残留需要的结构性改动**(未实现,只是指出形状):对残差流做
周期性的指数重新标定——类似 GDN 状态已经在做的"每 token 用 int8+per-
group scale 重新量化",但作用对象是残差流本身而不是 SSM 状态,可能需要
每 N 层(而不是每 token)重新选一次共享指数,避免逐 token 重新量化的
开销吞掉收益。这是一个块浮点(block-floating-point)方案,不是本次任务
范围。

## 3. 出货配方上没有任何投影角色是 F32(推翻一个假设)

`9569da9`(revert "ops: wire --norm-write int into the trunk norms")的
理由原文:"an F32-dtype [projection] consumer reads s->xb directly, so
the float write can never be skipped"——即 trunk norm 的几个下游消费者
(q/k/v_proj 或 in_proj_qkv/z/a/b,或 gate/up_proj)里,只要有一个是
F32 dtype,上游那次 float `lz_rmsnorm` 调用就不能省,`lz_rmsnorm_int`
只能作为**额外**开销叠加运行——这正是那次改动被量出 +5.4% f32 ops 而
回退的直接原因。

**这个假设本身是否在出货配方上成立,此前没有人验证过**——现在验证了:

- 训练树 `tools/quant_recipe.py`(THE quantization recipe of record,
  "p0")的 `ROLES` 表给每一个投影角色(`out_proj`/`in_proj_a`/
  `in_proj_b`/`v_proj`/`gate_proj`/`up_proj`/`o_proj`/`in_proj_qkv`/
  `in_proj_z`/`q_proj`/`down_proj`/`k_proj`/`b_proj`……)都分配了一个
  真实的量化格式(Q4_1/Q6_1/Q8_0/Q16_0 之一),**没有一个落在 F32**。
- 该文件自己在 `gate`(router)角色的按语里说得更直接:"the engine
  has no F32 weight format and Q16_0 is its ceiling, so 'do not
  quantize the router' resolves to Q16_0 by construction rather than
  by choice"——即便参照实现(K3)想要 F32 的精度,配方框架本身也从不
  把任何具名投影角色导出成 F32。
- 导出器 `tools/export_q8.py` 的 `plan()` 函数证实 F32 只在四类
  **结构性**例外上出现,而且是在任何配方规则生效之前就决定的:
  conv1d(gs 约束不可量化)、`A_log`/`dt_bias`(直接喂 `lz_exp`/
  `lz_softplus` 的小向量)、以及 **norm 自己的权重**(`o = x*inv*(1+w)`
  杠杆论证:`w` 接近零,量化误差相对它自己很大、相对 `1+w` 很小)。
  这四类里没有一类是 trunk norm 下游的投影权重。

**结论**:`9569da9` 的回退理由描述的可能性,在出货配方上不成立——
trunk norm 的 q/k/v(或 in_proj 家族、或 gate/up)从不是 F32。**回退
本身测出的 +5.4% 回归是真的**(代码按写法总是额外多算一次 float
sum-of-squares),但它的根因是**实现选择**(始终无条件跑 float 版,再
"加算"一次 int 版,而不是先判断"这个 norm 的所有兄弟消费者是否都
int8-可用,是则整条跳过 float"),不是数据强制的硬约束。这条边界
应该重分类为"可选强制,且当前实现连它自己允许的优化都没做到"——
在出货配方上,这条 float 计算是**死代码路径上多付的钱**,不是
不可避免的残留。

(norm 自己的权重 `w` 本身——不是投影权重——仍然永远是 F32,但那是
一条独立的、关于量化 norm 权重本身值不值得的**权重量化策略**决定,
文件里的杠杆论证与「这条 pipeline 边界能不能带 int」无关,不受本次
前提变更影响。)

## 4. SubLN Hadamard:仍是数据强制

`forward.c:2944-2947` 原话:"`--norm-write int` only applies when
dblk<=1: the dblk>1 branch feeds `subn_fwht_quant`, which needs the
FLOAT row to run the Hadamard transform before it quantizes - there is
no int-input twin of it, so that branch stays exactly as before
regardless of the tier."`subn_fwht_quant`(`forward.c:2053`)本身接
`const float *x`,内部再把它转成定点做 FWHT。这与 §3 的机制不同:不是
"consumer 可能是 F32"这种可选性,是"这个具体的变换实现只有一个 float
输入的版本"——一个真正的算法实现缺口。

去掉这条残留需要的结构性改动:写一个 int32 输入的 FWHT 变体(直接从
上游已经是定点的量取输入,不经过 float),铁律二仍然适用(gcc
intrinsics + Watcom `#pragma aux` 两份逐位一致)。未实现,不在本次
任务范围。

## 5. matmul epilogue:已知怎么拆,只差一个未解问题

`epi_fixed`(`ops.c:1117`)已经在整数域做完了这一步:int64 累加器 `s`
加一个纯整数指数 `e = -g_epi_xe[tk] - (int)w->sexp[row]`,只在返回
点转一次 float(`(float)s * pow2f(e) * post`,`91c0820` 把 `post` 折进
了指数,进一步减少浮点乘法)。转 float **只是因为调用方不知道下一步是
谁**——`o`(matmul 输出缓冲)被 30+ 个语义各异的调用点复用(q/k/v proj
接 RoPE、`o_proj`/`out_proj` 接残差加、gate/up 接 SiLU、router_logits
接 softmax、`embed_tokens` 接采样……),`epi_fixed` 不该也不能替每个
调用点决定下一步要不要 float。

`docs/matmul-int-output-design.md`(方案 A,"matmul 输出 int 化",已于
`f49da09` 整体退场)的收尾一段已经把这条路指清楚了:未来重拾这条线
应该接的是 `epi_fixed` 的 `(s, e)` 结果对,不是方案 A 自己的原始累加器
+ float scale 重新推导(那是一套更弱、和 `epi_fixed` 重复的设计)。
**真正阻塞的是一个未解问题**:一组输出行(同一个 matmul 调用的多个
`i`)会跨越多个不同的 `w->sexp[row]` 值,下游把它们当一个 int 数组
折算之前需要先做**指数对齐**——这是留下的问题,不是被否决的方向。
方案 A 本身被否决的三条理由(见该文档"退场"一节)都不是"int 化不
可能",而是"方案 A 选错了基础、选错了触发条件、也没有真实的
producer→consumer 直连点可以省"——`epi_fixed` 的 `(s,e)` 不受第二条
理由影响(它已经比方案 A 更省),第三条理由(没有真实直连点)在
"下游默认也是 fixed"的新前提下同样需要重新核对,因为旧结论是在
"下游可能是 float 所以必须转"这一前提下量的。

## 6. SwiGLU 融合:两个具体的实现坑,不是方向错了

`b2cfb0e`(wire `lz_quantize_q8_silu` into MoE SwiGLU)被 `58fc52e`
回退,理由记在 `b2cfb0e` 自己的提交信息里,是**性能回归,不是正确性
问题**——`--dump-logits` 在两个编译器、`--fixed all` 与默认配置下都
逐位相同,`make check` 全绿。回归的两个具体原因:

1. `lz_quantize_q8_silu` 无条件跑一遍"把结果重建成 float 存进 o[]"
   的 pass,而调用点从不读这个 float 值(它唯一可能的读者是 F32 权重
   matmul 的回退,而 `gs>0` 的闸门已经把这种情况排除在外)——这是一次
   调用点用不上的真实乘法,可以在 `gs>0` 时跳过,不需要等 int 贯通
   设计敲定。
2. 它的取整步骤是一个裸标量循环,不是 `lz_quantize_q8` 在
   `LZ_MMX_TU`/`LZ_SSE2_TU` 构建下会走的 `lz_q8round32_sse`/`_sse2`
   SIMD 派发路径——这是本仓库既有的教训(铁律八:每个算子的路径不能
   只有标量一条)在新写的融合代码里没有被应用。

**结论**:任何未来的 int 贯通重做,凡是涉及量化取整的新代码路径,都
必须复用 `lz_quantize_q8` 已有的 SIMD 派发(或给新路径同样配齐
MMX/SSE/SSE2 三档),不能新写一个只有标量循环的版本——否则即使数据流
方向完全正确,也会重复 `b2cfb0e` 已经踩过的这个坑。

## 7. 未实现的部分,及为什么

按团队负责人的范围界定,本任务**不实现** int 贯通管线本身——上面
§4/§5/§6 指出的每一处"可选强制,已知怎么拆"都还需要:一次未解决的
技术问题(§5 的指数对齐)、一个新的算法实现(§4 的 int 输入 FWHT)、
或者一次绕开已知实现坑的重写(§6)。这些是本文档的**设计输入**,不是
本次交付的一部分。Part A(见其自身报告)只处理"CLI 怎么选 tier、
默认解析成什么",不改变任何一条边界当前实际携带的是 float 还是 int。

## 8. 权重格式对边界的完整清单(供后续设计参考)

任何权重张量的 dtype 都独立于七个 write-tier,由模型导出格式决定。
`lz_act_gs`(`ops.h`)对 `LZ_FMT_F32` 返回 0,这是 `gso<=0`/`gsa<=0`
检查在多处(`forward_ssm`/`forward_kda`/`dense_ffn_step`/
`forward_moe`)能正确短路 int 融合分支的唯一原因。F32 权重不是死
代码——`model.c:1159-1160` 的 loader 会按 checkpoint 里的 dtype 字段
实际赋值 `LZ_FMT_F32`,且 §3 已确认它在出货配方上真实出现的位置只有
conv1d、`A_log`/`dt_bias`、以及每一个 norm 自己的权重——**没有一处
是 write-tier 管线里"norm 输出 → matmul 输入"这条边界的下游消费者**,
这正是 §3 结论成立的原因。
