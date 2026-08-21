# int 贯通管线立项计划（2026-08-21）

写于 HEAD `5bebc1eadd0b3f650fa108aa32ed2213836bb25f`（本文档本身作为新提交加在它
之上，不改 `src/`），分支 `feat/attn-conv-fixed-tiers`。方法：把
`docs/fixed-tier-pipeline-breakpoints.md`（Part B，`e0215a1`）与
`docs/matmul-int-output-design.md` 的结论当作既定输入，不重新推导；对每条
候选链路重新读 `forward.c`/`ops.c` 的实际调用点核实消费者数量（不采信交付
材料里的转述）；三个"不要重复"的历史提交逐个 `git show` 核实提交信息与前后
diff。

## 0. 结论先说

前提已经由用户定案：出货是**无条件**定点，float 路径留着但降级为**正确性
基准**——不再是「六选一乘六选一」的可选项。但真正挡路的不是"tier 要不要
可选"，而是七个内部定点算子（conv/recur/sig/scalar/epi/attn-scoring/
attn-wsum）**全部维持 float 进/float 出的调用约定，不管自己或邻居的 tier
取什么值**（`fixed-tier-pipeline-breakpoints.md` §2 的结论）——去掉 tier
可选性只是去掉了"不改"的借口，一行代码都没有因此变成必须改。

读完 `forward.c` 的全部 48 处 matmul 调用点（合并 subn/非-subn 的 if/else
分支后是 **33 条不同的 producer→consumer 链路**）后：

- **9 条**是团队规则下的候选（consumer 集合小且可枚举，见下表）——全部属于
  `epi_fixed`/`epi_q8` 之后紧跟 `conv1d` 或 `SiLU→quantize` 的窄链路。
- **9 条中有 7 条被同一个未解问题挡住**：`epi_fixed` 的 `(s[row], e[row])`
  按**输出行**各自独立（不是按 token、也不是按张量统一），任何想吃整段输出
  向量的下游消费者都要先把这些互不相同的指数对齐成一个。这不是"以后可能
  需要"的假设性障碍——它是这 7 条链路共同的、唯一的阻塞点。
- **2 条**（SwiGLU 家族里的两条，及其变体共 3 处调用点）不受这个问题阻塞，
  已经有一次真实尝试（`b2cfb0e`），因两个可修的实现缺陷被打回（`58fc52e`），
  是本计划里"重做，这次做对"置信度最高的一项。
- **约 7 条**终点是残差加（`s->x[i] += s->xb2[i]` 一类），结构性 float，
  团队已经把这类划为不做——核实后确认（见 §7）。
- **1 条家族**（attn 的 q_proj/k_proj/v_proj）真的有很多语义不同的消费者
  （RoPE 的三角函数、Hadamard 变换、KV cache 的 Q8/float/Q4 三份写入）——
  这条家族证实了交付材料"consumer 真的不同"的说法，但那个说法**不能推广
  到全部 33 条**；其余大多数链路的消费者集合是单一且可数的。

## 1. 链路清单

产出方式：`grep -nE "^\s*lz_matmul_(xq_nt|xq|w)\(" src/forward.c`（48 处调用
点）逐条读，合并 `c->use_subn` 的 if/else 两个分支（它们只改变产出这个 buffer
的 norm 路径，不改变下游消费者集合），得到 33 条链路。每条链路的"消费者数"
是在 `forward.c` 里读代码数出来的调用点数，不是猜的；LZ_TAP（`LZ_TAP("kq",
layer, s->kda_q, kdim)` 这类）是 debug 探针，`#ifndef` 出的正常构建里是
no-op，不计入消费者。

| # | producer（权重张量） | 输出 buffer | 消费者（数量，逐个列出） | 分类 |
|---|---|---|---|---|
| 1 | q_proj/k_proj/v_proj | `s->qg`/`ktmp`/`vtmp` | 3+：`lz_rmsnorm`(QK-Norm)、`lz_rope`、`rot_head`(Hadamard，条件性)、KV cache 的 Q8 写入、KV cache 的 float 参考面写入(`kf`/`vf`，条件性)、`lz_kv4_quantize`(条件性) | **多消费者，且部分本质需要 float**（RoPE 三角函数、Hadamard）——不是候选 |
| 2 | o_proj | `s->xb2` | 1：残差加 `s->x[i] += s->xb2[i]`(`forward.c:3200`/`3220`) | 结构性 float（B9）——**不在本项目范围**，见 §7 |
| 3 | in_proj_qkv | `s->qkv` | 1：`lz_causal_conv1d_step`/`_fixed`(`forward.c:1895`/`1885`) | **候选**（B6 家族之一） |
| 4 | in_proj_z | `s->zbuf` | 1：`lz_rmsnorm_gated`/`_int` 的门控输入(`forward.c:2008`) | 消费者单一但消费者本身是"z 门控输入"，与已完工的 B2 输出侧融合是不同的量——低优先级，见 §2.7 |
| 5 | in_proj_a | `s->avec` | 1：`lz_softplus(s->avec[...] + dt_bias)`(`forward.c:1949`) | 消费者是 `lz_softplus`——七轴/`--scalar-fixed` 之外唯一"从不参与任何 tier"的函数（`fixed-tier-pipeline-breakpoints.md` B10）——**二阶依赖，不在本项目范围** |
| 6 | in_proj_b | `s->bvec` | 1：`lz_sigmoid(s->bvec[...])`(`forward.c:1944`) | 消费者是 `lz_sigmoid`，B7 结构性 float 算子——**二阶依赖**，见 §2.7 |
| 7 | out_proj(SSM) | `s->xb2` | 1：残差加 | 结构性 float——不在范围 |
| 8 | kda_q_proj | `s->kda_q` | 1：`lz_causal_conv1d_step`(`forward.c:2338`附近) | **候选**（B6 家族） |
| 9 | kda_k_proj | `s->kda_k` | 1：`lz_causal_conv1d_step`(`forward.c:2341`) | **候选** |
| 10 | kda_v_proj | `s->kda_v` | 1：`lz_causal_conv1d_step`(`forward.c:2346`) | **候选** |
| 11 | kda_g_proj | `s->zbuf` | 1：门控输入，同 #4 | 低优先级，见 §2.7 |
| 12 | kda_f_a_proj | `s->kda_gate_lat` | 1：`lz_matmul_w` 内部的量化(`forward.c:2362`，喂 kda_f_b_proj) | **候选**（matmul→matmul，无非线性，`matmul-int-output-design.md` 已定位为"q16_0 consumer"） |
| 13 | kda_b_proj | `s->bvec` | 1：`lz_sigmoid`，同 #6 | 二阶依赖 |
| 14 | kda_o_proj | `s->xb2` | 1：残差加 | 结构性 float——不在范围 |
| 15 | moe_gate_w | `s->moe_router_logits` | 1：`lz_moe_route`(`forward.c:2674`，softmax+top-k) | 消费者单一但做离散排序决策，规模小（`ne` 个专家 vs 隐藏维）——低优先级，见 §2.8 |
| 16 | moe_down_proj | `s->moe_lat_x` | 1（每 token 一次，w1/w3 共享同一次量化结果）：`lz_quantize_q8`(`forward.c:2708`) | **候选**（matmul→matmul，非对称量化格式，`matmul-int-output-design.md` 的"q4_1/q6_1 consumer"） |
| 17 | moe_expert_w1[ei] | `s->moe_h1` | 1：SiLU 组合 `s->moe_h1[vv]=lz_silu(...)*s->moe_h3[vv]`(`forward.c:2714`) | **候选**（SwiGLU 家族，routed） |
| 18 | moe_expert_w3[ei] | `s->moe_h3` | 1：同上 SiLU 组合 | 同上 |
| 19 | moe_expert_w2[ei] | `s->moe_h2` | 1：累加进 `lat_y`(`forward.c:2719`) | 累加目标非残差流而是 latent 累加器，未深入评估，未纳入候选也未排除 |
| 20 | moe_up_proj | `s->xb2` | 1：残差加(经 `s->xb2[i] += s->moe_shared_out[i]` 合并共享专家后) | 结构性 float——不在范围 |
| 21 | moe_shared_gate | `s->moe_h1` | 1：SiLU 组合(`forward.c:2805`) | **候选**（SwiGLU 家族，shared） |
| 22 | moe_shared_up | `s->moe_h3` | 1：同上 | 同上 |
| 23 | moe_shared_down | `s->moe_shared_out` | 1：`s->xb2[i]+=s->moe_shared_out[i]`(`forward.c:2819`) | 结构性 float（并入残差流）——不在范围 |
| 24 | gate_proj | `s->hb` | 1：SiLU 组合(`forward.c:2934`) | **候选**（SwiGLU 家族，dense FFN） |
| 25 | up_proj | `s->hb2` | 1：同上；`use_subn` 时还有 `subn_fwht_quant` 读它做 down_norm 的 Hadamard 输入(`forward.c:2966`) | 同上；后者是 B2a 的结构性 float 反例，见 §2.6 |
| 26 | down_proj | `s->xb2` | 1：残差加 | 结构性 float——不在范围 |
| 27 | mtp.fc(第一次调用) | `s->mtp_x` | 1：`lz_rmsnorm` 进入 MTP 自己的 `forward_attn`(`forward.c:3028`)；随后又被当作 MTP 自己的残差流两次 `+=`(`3038`/`3044`) | `s->mtp_x` 起 MTP 残差流的作用，与 B1 同构——不在范围 |
| 28 | embed_tokens(MTP draft logits) | `s->mtp_draft_logits` | MTP 草稿 logits，供外部推测采样比较 | 面向调用方的最终数值——不在范围 |
| 29 | embed_tokens(主 lm_head, 批量) | `logits_out` | 面向调用方 | 不在范围 |
| 30 | embed_tokens(主 lm_head, 单 token) | `s->logits` | 同上 | 不在范围 |
| 31 | mtp.fc(第二次调用点, `forward.c:3512`) | `x_t` | 与 #27 同构（MTP 另一个入口的残差流） | 不在范围 |
| 32/33 | q/k/v_proj 的 SubLN 分支(`p[pi].out`) | 同 #1 三个 buffer | 同 #1 | 同 #1，不是独立链路，只是同一 buffer 的另一条产出路径 |

**头条数字**：33 条链路，9 条满足团队"consumer 集合小且可枚举"的候选门槛
（#3/#8/#9/#10/#12/#16/#17+18/#21+22/#24+25，最后三对各算一条 SwiGLU 链
路），其余约 7 条终点在残差加、1 个家族（#1）消费者真正异质、3 条被
`lz_sigmoid`/`lz_softplus` 这两个结构性 float 算子挡住（二阶依赖）、2 条低
优先级但未被硬性排除（#4/#11 的 z 门控、#15 的 router）。

## 2. 逐候选可行性

### 2.1 共同的阻塞点：`epi_fixed` 的按行指数，不是按张量、按 token 统一

`epi_fixed`(`ops.c:1117`) 已经在整数域算完累加，只在收尾做
`(float)s * pow2f(e-8)`，其中 `e = -g_epi_xe[tk] - w->sexp[row]`。`sexp` 是
`lz_epi_prep`(`ops.c:879`) 按**每一输出行**分配的（`malloc(rows)`,不是
`malloc(1)`），理由写在同一函数的注释里：`.prof/wepirange.c` 量过权重
scale 的行内分布 99.6% 落在 8x 以内，**但跨行的分布远宽**——这正是为什么
它们没有选"整张量一个指数"。

结果是：`epi_fixed` 产出的向量（例如 `s->qkv[tk][0..cdim-1]`，喂给候选 #3
的整段 conv 输入）里，**每个元素可能带不同的 `e`**。任何想让下游整段吃
`(s,e)` 而不物化 float 的方案，第一步都是把这一组互不相同的指数对齐成
consumer 能用的形式。`matmul-int-output-design.md` 退场记录只在
matmul→matmul 场景下提到这个问题；核实后发现它同样挡住候选 #3/#8/#9/#10
这四条 matmul→conv 链路——**这是 9 条候选里 7 条共享的唯一阻塞点，不是
某条链路独有的复杂度**。

具体到候选 #3（最小的一条，见 §3 的排序理由）：`lz_causal_conv1d_step_fixed`
(`ops.c:1840`) 已经在做类似的事——`float t = x[c] * in_scale; q =
q8_round(t);`——但 `in_scale` 是**整次调用共享的一个标量**（`pow2f(es)`,
`lz_conv_build_scales`(`ops.c:1888`) 只在 setup 时算一次），逐通道变化的只有
`mw[c]`/`sig_k1[c]`/`sig_oscale2[c]` 这些**权重**侧的量，不是输入侧。要接
`epi_fixed` 的按行指数，两条路都要新增代码，不是接线：

- **(a) 对齐成单一共享指数**：取这组 `e[row]` 的最小值（或按 `.prof/
  wepirange.c` 的方法量一次实际分布,取能覆盖 99% 情况的界),其余行的尾数
  右移对齐——这正是 `epi_fixed` 自己因为"跨行分布太宽"而没敢在 producer
  端做的事,现在要在 consumer 端做,右移会吃掉这些行的部分精度。
- **(b) 让 conv 核接受按通道变化的输入指数**：把 `in_scale` 从标量改成
  `in_scale[c]` 数组。工程上是局部的（`mw`/`sig_k1`/`sig_oscale2` 已经是
  per-channel,只是 `in_scale` 不是),但仍是新写的核,要重新过铁律二的双
  编译器逐位一致。

两条路都没有实现,方案 (a) 还是 (b)、或是否值得做,是这条链路本身要先答
的问题,不是可以绕过的接线细节。

### 2.2 候选 #3：`s->qkv` → conv（GDN/SSM）

- **接口改动**：需要 §2.1 的对齐方案之一，外加 `lz_causal_conv1d_step_fixed`
  签名从 `float *o, const float *x` 换成接受 `(int32_t/int16 *xi, int32_t
  e_or_exp, ...)` 或类似的整数入口。
- **省的东西**：census 目前把 conv1d 记在自己的 `LZ_FC_CONV` 位点
  （`missing-facilities.md`：ctx 1024、`--fixed all` 下 conv1d 占残留的
  4.5%、55296 次/token）,这条链路省的不是 conv1d 自己的位点,而是喂给它的
  那次 `epi_fixed` 收尾 `(float)s*pow2f(e-8)` 外加 conv 入口自己的
  `q8_round(x[c]*in_scale)`——**census 记不记得到这笔账要先核实**,见 §5
  第二条:census 按操作数计,不按代价计,这类"消除一次转换"的收益在把手法
  写清楚之前无法从 census 数字里直接读出来。
- **数值风险**：§2.1 的指数对齐是有损操作(方案 a)或需要新核(方案 b);
  `lz_causal_conv1d_step_fixed` 现在的输入量化路径已经有一次
  `q8_round`+clamp(`bound`)防溢出,新路径要重新证明不会因为对齐后的指数
  选择不当而系统性触顶。
- **验证方式**：铁律二双编译器逐位一致门（复用
  `build/fixed_tier_xcompiler_gate.sh` 的方法论,加一个新 tier 值)，配对
  NLL/argmax/top-5（不是 `|logits diff|`），且要在长序列上测——见 §6 的
  norm-int 教训,8/16 token 的探针曾经放过同一类缺陷。

### 2.3 候选 #8/#9/#10：`s->kda_q`/`kda_k`/`kda_v` → 各自的 conv

与 #3 同构（同一个 `epi_fixed`→conv 的阻塞点、同一个接口改动形状），三处
call site 用同一个 `lz_causal_conv1d_step`/`_fixed` 函数（`forward.c:2338`
附近三次调用），所以 §2.1 的核一旦写出来对齐方案，这三条链路复用它，边际
成本远低于从零开始的 #3。**排序上应该跟在 #3 后面一起做，不是分开评估**——
这是"先把最小的一条做对，再看它是否推广"的具体落点。

### 2.4 候选 #12：`s->kda_gate_lat`(kda_f_a_proj) → `lz_matmul_w`(kda_f_b_proj)

`matmul-int-output-design.md` 已经把这条链路核实到 dtype 级别：producer
`kda_f_a_proj` 是 q16_0（gs=512），consumer `kda_f_b_proj` 也是 q16_0
（gs=64），两端同格式、对称（只有 scale，无 zero-point），是文档里"最干净"
的一条 matmul→matmul 无非线性对。但同一份文档也把它的收益量过了：`f_a→f_b`
是 512→64×1024（gate_rank 很窄），6 个 KDA 层合计约占每 token f32 的 0.4%
（解析估算，未实测）。**仍然被 §2.1 的指数对齐问题挡住**，且收益小——列为
候选是因为它满足"单一消费者"门槛且格式最简单，可以作为 §2.1 核完成后的
第二个试点（比 #16 简单，比 conv 家族的收益验证方法更直接：两端都是纯
matmul，不涉及非线性核的接口改动）,但不建议在 §2.1 之前单独立项。

### 2.5 候选 #16：`s->moe_lat_x`(moe_down_proj) → 每 token 一次的量化(喂 w1/w3)

`matmul-int-output-design.md` 同样核实过：producer q8_0（gs=512），consumer
q4_1（L 层 gs=128）/q6_1（A 层 gs=256）——**非对称格式（scale + min/zero
point）**，整数域累加要多处理零点项，文档原话"复杂得多"。这是 9 条候选里
唯一同时叠加"指数对齐未解"和"非对称量化格式"两层复杂度的，**优先级最低**，
不建议在 §2.1/§2.2 之前排期,且即使 §2.1 做出来了,这条仍要单独设计零点
项的处理——不能指望 §2.1 的核直接复用。

### 2.6 候选 #17/18、#21/22、#24/25：SwiGLU 三处（routed / shared / dense FFN）

与其它候选**性质不同**：不需要 §2.1 的指数对齐,因为它不吃 `epi_fixed` 的
`(s,e)`——它面对的是已经物化成 float 的 `moe_h1`/`moe_h3`/`hb`/`hb2`（B5 决定
的,不可避免),要消除的是**这之后**的一次多余的 float 具现化：现有路径是
`moe_h1[i] = lz_silu(moe_h1[i]) * moe_h3[i]`（原地写回 float）,然后
`lz_quantize_q8` 单独扫一遍。`lz_quantize_q8_silu`(`ops.c:737`,当前零调用者)
把这两步融合：量化的 amax pass 直接算 `p[i]*(g[i]*sig[i])`,不先写回
`moe_h1`。

**这条已经真实尝试过一次**（`b2cfb0e`，MoE 两处），被 `58fc52e` 打回，原因
记在两个提交里、**都是实现缺陷，不是链路本身不可行**：

1. `lz_quantize_q8_silu` 末尾无条件把重建的 float 值写回 `o[]`
   （`ops.c:810-814`，注释自称"for the f32 matmul fallback and the sgdn
   tap"），但 MoE 的两个调用点权重都不是 F32、也不喂 sgdn tap——这次写回
   在那两个调用点上是没有读者的纯浪费。
2. 量化的 round 步骤是裸标量循环（`ops.c:795-801`），没有走
   `lz_quantize_q8` 已经有的 SIMD 分发（`LZ_HAVE_Q8R_SIMD`/`LZ_HAVE_Q8R_SSE`
   分支，`ops.c:416-449`，`lz_q8round32_sse`/`lz_q8round_group_sse2`）。

修复后的重做**必须**：把 `o` 参数在这三处调用点传 `NULL`（或加一个"要不要
物化"参数）使第 810-814 行那趟重建可跳过；并把 795-801 行的 round 循环
换成与 `lz_quantize_q8` 同款的 SIMD 分发，而不是让它继续裸跑标量。原
revert 的提交信息已经把回归的数字拆到逐 call 精确值（`4.0625 * n`/call,
n=SwiGLU 宽度），重做后应该拿这同一个公式反过来验证：省下的量应该是这
个数的量级，不是随便一个"变快了"的印象。

**优先级次于 §2.1 家族**，但**是全项目里唯一不需要先解决指数对齐就能动手
的候选**——如果要一个和 §2.1 并行推进的第二条线，选这个。

### 2.7 低优先级但未排除：`s->zbuf`/`s->avec`/`s->bvec` 的门控输入(#4/#6/#11/#13)

消费者单一（`lz_rmsnorm_gated`/`lz_sigmoid`/`lz_softplus`），但这三个消费者
自己就是 B2/B7/B10 里"结构性 float"的算子——`lz_softplus` 更是"七轴之外
从不参与任何 tier"的函数,永远走 double libm(`fixed-tier-pipeline-
breakpoints.md` B10)。让这三条链路吃 int 输入,前提是先给 `lz_rmsnorm_
gated`(z 门控这一侧)、`lz_sigmoid`、`lz_softplus` 本身长出 int-in 变体——
那是比这个项目大一级的题目(动算子接口,不是接现成的两端)。记录在案,
不建议本项目动手;如果以后 `lz_softplus` 真的要长出定点档,那是它自己的
立项,不是这里的一个 stage。

### 2.8 低优先级：`s->moe_router_logits` → `lz_moe_route`(#15)

消费者单一，但消费者做的是 softmax+top-k 的离散排序决策，`ne`（专家数）
通常远小于隐藏维——即便解决了 §2.1，这条链路的绝对收益也很小（参照 §2.4
里 gate_rank 窄导致收益 0.4% 的同类推理）。不建议排期，除非有模型把专家数
开得很大。

## 3. 排序（收益 × 可行性，含推理）

1. **§2.1 指数对齐原语 + 候选 #3 作为试点**——排第一不是因为它单独收益最大，
   是因为它是 7 条候选（#3/#8/#9/#10/#12，以及未来任何想吃 `epi_fixed`
   输出的链路）共同的前置条件。选 #3 而不是 #12/#16 做试点，是因为它的
   consumer（`lz_causal_conv1d_step_fixed`）内部已经有一段结构相似的
   float→int 量化代码可以对照（§2.2 详述），且它是**单一 tensor、单一
   consumer**，没有 #16 的非对称格式问题、也没有 #12 那种"收益小到可能
   不值得"的先天限制。如果 #3 在数值上或工程上失败，#8/#9/#10/#12 大概率
   也失败，止损成本最低。
2. **#8/#9/#10（KDA 的 conv 家族）**——#3 做出来之后的直接推广，边际成本
   低，合并汇报。
3. **SwiGLU 重做（#17/18、#21/22、#24/25）**——与 1/2 并行的独立线，不依赖
   §2.1，是"重做，这次做对"里置信度最高的一项（缺陷已知、量化公式已知、
   修复动作已知）。可以先于 #3 做，也可以并行；排在 §2.1 之后仅仅是因为
   §2.1 的产出对全项目更关键，不是因为 SwiGLU 更难。
4. **#12（kda_f_a→kda_f_b）**——§2.1 完成后的第二个试点，验证对齐原语是否
   能推广到真正的 matmul→matmul（不只是 matmul→conv）。收益小（0.4%），
   价值主要是验证性的。
5. **#16（moe_down→expert w1/w3）**——只有在 §2.1 和 §2.4 都验证通过、且
   有人愿意再单独设计非对称量化格式的零点项处理时才启动。当前证据不支持
   现在排期。
6. **#4/#6/#11/#13/#15 及其它未列入候选的链路**——不排期，原因见 §2.7/§2.8。

## 4. 逐阶段 go/no-go 判据

**§2.1 指数对齐原语（阶段 0）**
- go：能找到一种对齐方式（共享指数或 per-channel 核改造），在
  `build/fixed_tier_xcompiler_gate.sh` 同款方法论下 gcc/Watcom 逐位一致；
  且用带比较运算的仪器（不是现成 census，见 §5）量出**真实**减少的操作，
  不是操作数搬家。
- no-go：对齐只能靠"取最小指数、其余行右移"这种有损方式，且长序列（见
  §6）配对 NLL/top-5 一致率相对 float 基准的下降超过已有定点档（`--fixed
  all`）自身相对 float 的下降幅度——即对齐引入的误差比"定点本身"这件事
  引入的误差还大，那就是在拿数值质量换一个未经证明的收益，止损。

**候选 #3（试点）**
- go：阶段 0 通过，且census（带 §5 修正）显示 `qkv`→conv 这一段的浮点操作
  数下降，census 之外用手工过一遍 `.prof` 探针交叉核实（同
  `matmul-int-output-design.md` 第一次"−9.9%"是仪器假象那次的教训，不能
  只信一个数字来源）。
- no-go：下降量在噪声范围内（参照 `matmul-int-output-design.md` 记录的
  +0.14%/-9.5%/-86.7% 三种真实量级，判断这次改动落在哪一档），或
  cross-compiler 逐位一致做不到。

**#8/#9/#10（推广）**
- go：#3 通过且改动能以"同一个对齐函数 + 同一种核改造"的形式复用，不用
  为每个 tensor 重新设计。
- no-go：三者的 `w->sexp[row]` 分布差异大到需要三套不同的对齐参数——那说明
  #3 的解法没有真正抽象出可复用的原语，回到阶段 0 重新设计,不要在
  三个调用点各自打补丁。

**SwiGLU 重做**
- go：新版 `lz_quantize_q8_silu` 同时做到（a）`o` 在 MoE/dense FFN 三个
  调用点不做无谓的物化,（b）round 步骤走 SIMD 分发；`--dump-logits` 在
  gcc/Watcom、`--fixed all`/default 四种组合下与不融合的路径逐位一致；
  census 净变化与 `b2cfb0e` revert 记录的 `4.0625*n` 公式方向相反(即
  这次是净省,量级可比)。
- no-go：修完两个缺陷后 census 仍然是净增——那说明"融合能省"这个前提本身
  错了(比如量化 SIMD 路径本来就已经把 amax pass 和 round pass 高度优化,
  融合并不能再省),必须承认并停止,不要继续在这条线上加第三个"缺陷修复"
  尝试。

**#12**
- go：阶段 0 的对齐原语能直接套用到两个纯 matmul 端点(不需要 conv 核那种
  per-channel 改造),且验证方法论(逐位一致 + 配对 NLL)复用 #3 的现成脚本。
- no-go：0.4% 的解析估算在实测中进一步缩水,或对齐引入的误差在 gate_rank
  很窄(容易被单个 token 的极端值支配)时被放大——先在真实数据上量,不要
  只信解析估算。

**#16**
- 在 #12 拿到 go 之前不评估这条的 go/no-go；#12 之后如果做，第一步是先
  单独设计并验证零点项的整数域处理，验证方式独立于对齐原语，验证不过就
  不要往下走。

## 5. 不要重复的三个错误（均在 git 历史里，已逐条 `git show` 核实）

1. **`9569da9`：trunk-norm int 加在 float 旁边,不是替换它。** 提交信息
   给出了精确数字：打开这个开关净亏 +49562 f32 ops/token(+5.4%)，norms
   168020→196788，quantize 101112→121848，rsqrt 1117→1174。根因不是编码
   错误，是 trunk 的两处 norm（`input_layernorm`/`post_attention_
   layernorm`）有多个不同权重格式、不同 group size 的消费者，其中一个
   F32 消费者直接读 `s->xb`，导致 float 版 `lz_rmsnorm` 永远不能省——
   `lz_rmsnorm_int` 于是是**额外**跑一遍，不是**替代**。已完工的 gated/
   SubLN int 路径能省，恰恰因为各自只有一个消费者、float 输出真的能丢。
   **本计划§1 的候选表就是照这条教训筛出来的**——"consumer 数 = 1"是
   硬门槛,不是建议。
2. **`f49da09`：scheme-A matmul int 机制作为 `epi_fixed` 的替代品,而不是
   建在它上面。** 两个入口函数一遇到 `q8_fixed_epi(...)` 为真就直接
   `return 0` bail——而 `lz_epi_mode()` 的 auto 恰好在需要它的无 SSE
   机器上打开 `epi_fixed`,方向整个是反的：`--fixed all` 下开关 on/off
   连 top logit 都逐位相同。**本计划§2.1/§2.2/§2.4 因此明确要求任何
   新方案接在 `epi_fixed` 的 `(s,e)` 结果对之上,不是再造一套独立的
   matmul int 累加器**——退场提交自己的收尾段落已经指名了这个方向，本
   文档是对它的展开，不是另起一套。
3. **SwiGLU 融合（`b2cfb0e`，被 `58fc52e` 打回）：标量融合量化输了给已有
   的 SIMD 分发 round 路径。** 见 §2.6 的完整分析——**任何 int 贯通的
   重新设计都必须继续走到 `lz_q8round32_sse`/`lz_q8round_group_sse2`，
   不能用一个新写的标量循环替换它**，这是交付材料里点名要求的约束，
   核实后确认成立且理由具体（不是泛泛的"SIMD 更快"，是这次融合的循环
   体量刚好落在标量循环显著慢于 SIMD 分发的区间）。

## 6. 每个阶段都适用的正确性约束

- **`int32→float` 转换只能走 `lz_i32f`，`int64→float` 必须经过一个
  `volatile float` 中间量**——裸转换在 gcc/SSE 和 Watcom/x87 上舍入次数
  不同（SSE 转换一次、乘法再舍入一次；x87 `fild`+`fmul` 融合只舍入一次）。
  这不是假设性风险：`build/fixed_tier_xcompiler_gate.sh` 的文件头注释记着
  一次真实事故——`lz_quantize_q8_int` 的快速分支曾经裸转 `grp[k]`，在
  kmr20 模型、153 token 里的第 146 个位置（layer 0, group 1）才第一次分歧,
  更短的探针和纯英文输入都测不出来。`src/ops.c` 里 `lz_quantize_q8_int`
  的函数注释记着同一机制。任何本项目新写的 int→float 转换点都要过这同一
  个门,不能假设"这次不一样"。
- **跨编译器逐位一致是硬约束，不是容差**——CLAUDE.md 原话:"gcc-fixed and
  Watcom-fixed must not [differ]"。任何新 tier/新路径都要在两个工具链上
  逐位核对，不是"看起来数值接近就算过"。
- **`|logits diff|` 不是判据，它饱和**——`fixed-tier-precision-audit.md`
  §1 实测过：单次扰动 1.89e-2~1.30e-1，而整条序列的 worst-case 能到
  1.55~1.66，两者差一个数量级以上，都不能用来排序 tier。判据是配对 NLL、
  argmax 一致率、top-5 一致率（`--dump-nll`/`--dump-top5`）。
- **逐位一致的门必须用长序列、且不止一种语料**——`norm-write int` 这个
  缺陷本身就是反例：8/16 token 探针和 211-token 纯英文样本都是干净的，
  在 153-token 中文样本的第 146 个位置才第一次出现分歧。任何本项目新增
  的门都要复用或扩展 `build/norm_int_gate_tokens_{zh,en}.txt`（603/661
  token）这一类长且真实的语料，不能只在短探针上宣称"通过"。

## 7. 仪器诚实声明

- **census 在 `LZ_FC_QUANT` 位点(`f18f02d`)加入之前，quantize 这块工作
  完全不计费**。那次提交自己量过：同一个 `/tmp/kmr20 --fixed all` 8-token
  跑，总数从 820945 涨到 919945（+99000，相对新总数约 10.8%，即团队材料
  说的"~11%"）。这不是"quantize 现在被多算了"，是**在这个提交之前的任何
  总量数字，quantize 这部分本来就是 0，是真实存在但不可见的工作**——本
  项目如果要引用某个"之前测过的总量"做对照基线，必须确认那次测量是在
  `f18f02d` 之后做的，否则基线本身就低了一成。
- **census 不计比较运算、不计取绝对值**，因此**系统性高估**任何"消除一次
  amax 扫描或 clamp"的收益。`missing-facilities.md` 在软浮点 ARM 上实测
  过反例：一次浮点 amax 扫描花 0.409（QEMU 墙钟单位），而**被计费**的浮点
  平方和臂只花 0.247——比被计费的工作量还大 1.6 倍，census 却记 0。本
  项目的候选里，§2.2/§2.3 的 conv 入口量化、§2.4/§2.5 的 matmul 量化，
  全部含有 amax/clamp 步骤，**census 报的"节省"数字方向是对的，幅度多半
  是低的**——报数字时要连这句话一起报，不能只贴 census 输出。
- **census 与 ISA 无关**：`--kernel mmx` 与 `--kernel sse2` 报同样的数字
  （`missing-facilities.md` 426 行附近，recurrence write 单独验证过）。
  它回答"操作数少了多少"，回答不了"在目标机上是不是真的更快"——本项目
  没有铁律里点名的 WinChip 模拟器台架，所以任何"更快"的表述都要明确降级
  成"操作数更少"，且照 §5 的 `.409 vs .247` 案例说明操作数不是许可证。
- **必须量在出货配置（`--fixed all`）上，不是默认配置**，且要证明测量的
  路径真的在跑。`matmul-int-output-design.md` 自己踩过这个坑：scheme A
  的收益全部量在浮点档，出货配置（无 SSE 机器的 `auto` → 定点 epilogue）
  一开，两条入口函数直接 `return 0`——"这条线只在不需要它的机器上生效"。
  本项目任何候选的收益数字，第一步先确认 `q8_fixed_epi(...)`（或对应的
  tier 谓词）在测量时为真，而不是默认配置下悄悄没跑。

## 8. 本项目明确不做的部分

- **残差加**（`s->x[i] += s->xb2[i]` 及其在 MTP/MoE 共享专家合并里的
  变体，§1 表里 7 条终点在这里的链路）——结构性 float，没有 int 版本，
  交付材料点名排除，核实后确认没有反例。
- **`q_proj`/`k_proj`/`v_proj` 的输出**（§1 #1）——消费者集合是 RoPE 三角
  函数、Hadamard 变换、三种 KV cache 格式的并行写入，其中至少两个（RoPE、
  Hadamard）在当前实现范围内没有整数版本、也不是这个项目要新开的题目。
- **LM head / MTP 草稿 logits**（§1 #28/29/30）——面向调用方的最终数值，
  下游是采样、`--dump-nll`、推测解码比较，这些语义明确需要浮点。
- **`lz_softplus`**——不受任何 `--*-write` 轴或 `--scalar-fixed` 影响，
  永远走 double libm；给它建一个定点版本是比本项目大一级的题目（改的是
  算子本身而不是链路两端的接线），不在本计划范围。若以后要做，应该是它
  自己的立项。
- **`lz_sigmoid`/`lz_rmsnorm_gated` 的输入侧**（§2.7 的 #4/#6/#11/#13）——
  同理，需要先给这两个结构性 float 算子长出 int-in 接口，本项目不做。
- **`s->moe_router_logits`**（§2.8）——收益太小，不值得为一个离散排序
  决策引入量化噪声的额外复杂度。
- **`s->moe_h2`→`lat_y` 累加**（§1 #19）——目标是 latent 空间的累加器而
  非残差流，未深入评估其数值语义（是否可能像残差加一样结构性需要 float
  以避免多专家累加误差放大），本文档不下结论，留给专门评估。
