# 定点档管线边界图(Part B,2026-08-21)

SHA 见提交记录(本文与 Part A 的实现在同一批提交里)。分支
`feat/attn-conv-fixed-tiers`。方法:通读 `forward.c`/`ops.c`/`ops_matmul.c`/
`ops_gdn.c` 的实际数据流,逐条验证交付方给出的候选(不采信候选本身的措辞),
按代码里能看到的函数签名和调用点下结论。

## 0. 结论先说

七个 write-tier 里,六个(`--recur-write` / `--conv-write` / `--attn-write` /
`--sig-write` / `--norm-write` / `--epi-write`)的 `auto` 分支全部只问同一个
谓词 `lz_soft_float_target()`(`ops_sched.c:127`,CPUID SSE 位)。这一点交付方
的前提是对的。但**由此推出的"stage A fixed 而 stage B float 时 A 必须为 B
物化 float"并不成立**——本仓库把每一个内部定点算子都设计成 float 进/float
出的固定调用约定,不看自己的 tier、也不看邻居的 tier,所以物化 float 这件事
在**每一种 tier 组合下都会发生**,不是"六选一乘六选一"矩阵里才多出来的
成本。真正因 tier 组合而生的收益来自另一件事:六个函数各自内部选择"要不要
把这次调用的算术留在整数域",而不是"要不要把结果吐成整数交给下一站"。

七轴之外还有第八个真正的分支点:matmul 权重是不是 `LZ_FMT_F32`
(`model.h:51`)。它不受任何 `--*-write` 旗标控制,是模型本身的存储格式,
但它是本文能找到的唯一一处**结构性**强制物化 float 的地方,并且**与
交付方候选 3 的原始表述("非 Q8_0 的 matmul")不同**——Q4_1/Q6_1/Q16_0/T2
四种非 Q8_0 格式全部走 int8 激活路径,只有 F32 权重才落到 float 回退。

## 1. 边界表

管线顺序按交付方给出的链路:`残差 x -> norm -> quantize -> matmul ->
epilogue -> {conv | recurrence | attention} -> sigmoid/silu -> quantize ->
matmul -> ... -> 残差加`。

| # | 边界 | 分类 | 依据 |
|---|---|---|---|
| B1 | 残差流 `x`(进入 norm) | **structurally float** | `s->x[i] += s->xb2[i]`(`forward.c:3200`/`3220`/`3038`/`3044`,以及 MoE 共享专家合并 `forward.c:2819`)全部是无条件 float 加法,没有任何 tier 判据、没有 int 版本。四个 norm 变体(`lz_rmsnorm`/`lz_rmsnorm_gated`/`lz_rmsnorm_int`/`lz_rmsnorm_gated_int`,`ops.h:129/137/146/158`)入参签名统一是 `const float *x` |
| B2 | norm 输出 → quantize 输入 | **两档都有,由 `--norm-write` 控制** | float/fixed 档:norm 写回 `float *o`,随后 `lz_quantize_q8` 读 float。int 档:`lz_rmsnorm_int`/`lz_rmsnorm_gated_int` 把 Q15 结果整体写成 `int32_t *t`,`lz_quantize_q8_int`/`lz_quantize_q8_tok_int` 直接读 int32 配一个二次幂 `deq`,不落 float(`forward.c:1981-2002`,`ops.h:146/158`)。**int 档本身是这条边界"could carry int"这件事已经被证明可行的证据**,fixed 档不用它只是因为 fixed 只改 norm 内部平方和的算法,不改输出类型 |
| B2a | norm 的一个真实结构性反例:SubLN Hadamard | **structurally float** | `forward.c:2944-2947` 原话:"`--norm-write int` only applies when dblk<=1: the dblk>1 branch feeds `subn_fwht_quant`, which needs the FLOAT row to run the Hadamard transform before it quantizes - there is no int-input twin of it, so that branch stays exactly as before regardless of the tier." `subn_fwht_quant`(`forward.c:2053`)本身接 `const float *x`,自己内部再转定点做 FWHT。这是交付方候选 2 的真实机制:不是"consumer 未知",是"consumer(FWHT)只有 float 输入的实现" |
| B2b | norm-int 的另一处保护:F32 权重前置条件 | **structurally float(间接)** | `forward_ssm`(`forward.c:1979-1982`)等处的 norm-int 分支都带 `gso > 0 && (vd % gso) == 0` 前置条件;`lz_act_gs` 对 F32 权重返回 0(`ops.h` 该函数注释)。若下游 matmul 权重是 F32,`gso<=0` 直接短路 norm-int 分支,回退到会把归一化结果原地写回 `s->ssm_out`(float)的分支——这保证了 `lz_matmul_xq_nt` 的 float 回退读到的是**正确值**而不是 norm-int 分支留下的陈旧 pre-norm 值。链路是:F32 权重(B4 的分支点)→ 反向约束 B2 必须走 float |
| B3 | quantize 输出(int8+scale) → matmul 输入 | **carries int,无条件,与七轴无关** | 只要权重不是 F32,matmul 一律吃 `xq`/`xqs`(int8+float scale),这个事实不随任何 `--*-write` 取值改变。这条边界本来就不属于本次讨论的"六轴对齐"范畴 |
| B4 | matmul 权重格式分派 | **唯一的结构性硬分支,由权重 dtype 决定,与七轴正交** | `lz_matmul_xq_nt`(`ops.c:1498-1601`)只有 `w->dtype == LZ_FMT_F32` 落到 `x`(float 激活)回退(`ops.c:1508`,经 `lz_matmul`);Q8_0/Q4_1/Q6_1/Q16_0/T2 全部走 `xq`/`xqs` int8 路径(`ops_matmul.c:313-420` 逐格式确认:`matmul_q8_impl`/`matmul_q41_impl`/`matmul_q61_impl`/`matmul_q16_impl`/`matmul_t2_impl`/`matmul_scalar_ref` 内部一致用 `xq[...]`)。**纠正交付方候选 3**:分界线是"是不是 F32",不是"是不是 Q8_0" |
| B5 | matmul 内部累加器(int32/int64)→ epilogue 输出 | **structurally float** | `epi_fixed`(int64 累加器 + 幂次指数,`ops.c:1117`)与 `epi_q8`(int32 累加器 + float scale,`ops.c:1146`)都以 `float` 返回,由 `q8_fixed_epi` 谓词(`ops_matmul.c:243`,读 `lz_epi_mode()`)决定跑哪一个,**但两个分支返回类型相同**。原因不是"consumer 未知"这么抽象——是matmul 的输出缓冲 `o` 被 30+ 个语义各异的调用点复用(q/k/v proj 后接 RoPE、`o_proj`/`out_proj` 后接残差加、gate/up 后接 SiLU、router_logits 后接 softmax、`embed_tokens` 后接采样……),epi 函数不知道也不该知道下一步是谁,float 是唯一公共接口。**确认交付方候选 4**,补上具体机制 |
| B6 | epilogue → {conv \| recurrence \| attention} 输入 | **structurally float,且与 `--epi-write` 的取值无关** | `lz_causal_conv1d_step`/`_fixed`(`ops.c:1780`/`1840`)签名相同:`float *o, const float *x`——fixed 版内部状态是 int16(`sh_in`/`sh_out`),接口不变。`lz_gdn_step`(`ops_gdn.c:1931`)的 pass-1"标量尾声"(产出每 token 的 `out[]`)完全没有 tier 判据、恒为 float 算术(`ops_gdn.c:1853-1856`);`--recur-write` 只切换 pass-2(状态写回)是否经过 float 往返,不切换 pass-1 输出类型,也不切换 pass-2 的存储格式——两档下 `sq_out` 都是 int8,差别只在得到它的算法路径。attention 的 Q 同理:`lz_attn_score_q8`(fixed scoring kernel)仍收 `const float *qhh`(`ops.h:1082`),只是函数内部把 Q 就地量化,函数边界仍是 float |
| B7 | conv/recur/attn 输出 → sigmoid/silu 输入 | **绝大多数 structurally float,attn 输出门是唯一的例外** | `lz_sigmoid`/`lz_silu`(`ops.c:168`/`195`)签名恒为 `float -> float`,`--sig-write` 只切内部近似算法。`lz_causal_conv1d_step_fixed` 把 sigmoid 内联成 `sigmoid_q15_t` 直接作用在 fixed-point 累加器上(`ops.c:1868-1871`),不经过 `lz_sigmoid` 这个接口,但只在 conv 函数体内部生效,对外仍是 `o[c]` 这个 float。attention 输出门(`forward.c:1751-1782`)是本表里第二处真正的 int/float 双路径:`attn_int[tk]` 为真时 `(ac[d]*(int64_t)s15)>>15` 全程 int64,不经过任何 float;否则 `out[d] *= lz_sigmoid(gate[d])` |
| B8 | sigmoid/silu(或 gated norm)输出 → quantize → matmul | **与 B2 同构,已有的两个已实现分支** | attention 侧是 `lz_quantize_q8_int64`(`forward.c:1802-1809`,读 `s->attn_acc` int64,配 `s->attn_ss`),SSM/KDA 侧是 B2 讨论过的 `lz_quantize_q8_int`。两者都要求 `gso==32`(o_proj/out_proj 的 Q8 组大小与 wsum/norm 组大小对齐),且都用 `!c->use_subn`(attn)或前述 `gso>0`(norm)保护 |
| B9 | matmul 输出 → 残差加 | **structurally float** | 同 B5 结论:`epi_fixed`/`epi_q8` 都返回 float,`s->x[i] += s->xb2[i]` 无条件 float |
| B10 | `--scalar-fixed`(`lz_exp`/`lz_rsqrt`) | **不在六轴管线内,横切关注点** | `lz_exp`/`lz_rsqrt`(`ops.c:315`/`301`)签名恒为 `float -> float`,`--scalar-fixed` 只切内部实现(Q20/Q23 查表+牛顿 vs 多项式),从不改变调用点看到的类型。**额外发现**:`lz_softplus`(`ops.c:262`)完全不受 `--scalar-fixed` 影响,永远用 double libm——七轴/`--scalar-fixed` 之外还有一个"从不参与任何 tier"的浮点专用函数,GDN/KDA 的 decay 门控(`a + dt_bias` → softplus)会经过它 |

## 2. "只因为 tier 可独立选择才是 float"的边界:结论为空

逐条核对 B1–B10 后,**没有一条边界属于"仅因为相邻轴的 tier 取值不一致才
物化成 float"**。原因是结构性的:conv/recur/sig/scalar/epi/attn-scoring/
attn-wsum 这七个内部定点算子,每一个都无条件维持 float 进 / float 出的调用
约定——不查自己的 tier(边界类型不随 tier 变,只有内部算法随 tier 变),
更不查邻居的 tier。唯二真正让边界类型分裂成 int/float 两种可能的机制
(norm-int、attn-int)都是**自包含的单算子融合**(把 norm+quantize 或
wsum+quantize 焊在一起跳过 float 往返),不读取、也不依赖其余五个轴当前
处于什么状态。

所以交付方给出的"stage A fixed and stage B float means A must materialise
float for B"这个前提在当前代码里**不成立**:A 不管取什么值都已经物化成
float,B 不管取什么值都从 float 起步。六轴合并成一个开关的价值不在于消除
一笔"因组合而多付"的物化成本(没有这笔成本),而在于:

1. **消除"同一个决策被拼写六次"的漂移风险**——六份 `(g_X < 0) ?
   lz_soft_float_target() : ...` 独立存在,新增第七个这样的轴、或给某一份
   打补丁时漏改另外五份,不会有任何测试报错,因为它们语法上互不相关;
2. **让"这是一个决策"这件事本身在代码里可验证**,而不是只存在于注释里
   给读者一个"据说如此"的印象。

## 3. "could carry int, currently does not" 的真实机会(超出本次 Part A 范围)

两处记录下来,供以后需要时参考,均**不**在这次 Part A 的改动范围内:

- **epilogue → conv/recur/attn 之间的物化(B6)**:若两端都被设为 fixed,
  理论上可以让 `epi_fixed` 的 int64 累加器直接喂给 `lz_causal_conv1d_step_fixed`
  的 int16 历史更新,跳过"转 float 再 `q8_round` 回定点"这一步。目前不存在
  这样的融合——不是因为不可能,是因为每个算子被设计成独立的、可各自 A/B
  测试的模块,融合会牺牲这个独立性,且矩阵形状会从"7 个二元轴"变成
  "若干个必须联合决策的算子对",与铁律九"有争议就做成旋钮"的可组合性
  相悖。
- **norm-int/attn-int 覆盖不全**:两个融合目前只接到 `forward_ssm`/
  `forward_kda`/`forward_attn` 里的部分调用点;`forward_moe` 的
  `router_logits`/`moe_lat_x` 两处 norm(`forward.c:2618-2657`)即使
  `gso>0` 也仍是纯 float 路径,没有接入 `lz_norm_int`。这是候选清单里
  "could carry int, currently does not"的具体例子,不是被禁止,只是没写。

## 4. 权重格式对 quantize/matmul 边界的完整清单(B3/B4 的展开)

供 Part A 设计参考:任何权重张量的 dtype 都独立于七个 write-tier,由
模型导出格式决定。`lz_act_gs`(`ops.h`)对 `LZ_FMT_F32` 返回 0,这是
`gso<=0`/`gsa<=0` 检查在多处(`forward_ssm`/`forward_kda`/`dense_ffn_step`/
`forward_moe`)能正确短路 int 融合分支的唯一原因。F32 权重不是死代码——
`model.c:1159-1160` 里 loader 会按 checkpoint 里的 dtype 字段实际赋值
`LZ_FMT_F32`。
