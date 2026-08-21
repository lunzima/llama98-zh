# int 贯通管线立项计划（2026-08-21，第二次重排）

写于 HEAD `4dbb5c85aef7615fb750c491ad7dbd65d0502445`（本文档本身作为新提交加
在它之上，不改 `src/`），分支 `feat/attn-conv-fixed-tiers`。方法：把
`docs/fixed-tier-pipeline-breakpoints.md`、`docs/matmul-int-output-design.md`
的既有结论当输入；本次重排额外核实的每一条 dtype/n 声明，都用 `.prof/
dtype_probe.c`（权重面按 dtype 汇总）和扩展过的 `.prof/dtype_probe_kda.c`
（按张量名逐个打印 dtype/gs/n）在隔离 checkout 里独立编译、独立跑出来，
不采信 `model.h` 里的类型注释——这正是上一轮排序出错的原因（见下）。

## 0. 先说清楚：这次重排为什么发生，以及仪器口径

**上一版的排序建在一个用错的判据上。** 判据是"这条链路占它起点站
（`epi_fixed`）的份额有多大"，量出来 `epi_fixed` 只占真实 f32 总量的
0.55%（见下），据此把 §3 的排序定成"先做 #3/#8/#9/#10，因为最小最干
净"。但这个判据量的是起点站的大小，不是这个机制（"让 int 穿过 epilogue
边界而不是每站物化 float"）能覆盖多大的面。真正的面是**整个 dequant
epilogue 位点**，而这个位点在 `5d852a2` 之前，`epi_q41`（服务 Q4_1/Q6_1，
以及 `matmul_t2_impl` 共享的同一份实现——T2 在 kmr20 权重面上是 0%,但
代码路径上确实是第三个调用方）**完全没有记账**：

```
                    census 位点         每 token 次数
epi_pack_act              544 ops           30 次
epi_fixed              32,768 ops       16,384 次   (Q8_0 · fixed 档专用)
epi_q8                 86,304 ops        6,752 次   (q16_0 全部 + Q8_0 在
                                                      q8_fixed_epi 判否时)
epi_q41              5,089,280 ops      101,376 次   (Q4_1 + Q6_1 + T2，
                                                      此前零记账)
```

四者相加 5,208,896，与补账后 census 报的新 dequant-epilogue 数字逐位对
上；`pack+fixed+q8=119,616` 与补账前的旧数字逐位对上——两次独立加总互相
印证，不是巧合。补账前 dequant epilogue 只占旧总量 903,561 的 13.2%，
补账后是 5,992,841 里的 86.9%（**43.5x**），norms/sigmoid/conv1d 这些原来
看着占大头的位点相应从 13–19% 掉到 2–3%。**这些数字由另一个会话落地
（`5d852a2` 的记账改动 + 配套的未记账路径闸门），本文档按团队负责人的
交底原样采用，不重新推导**；本文档独立验证的是 dtype/n 这一类结构性
事实，见下面每一节的 `.prof/dtype_probe*.c` 输出。

**这次重排改的不是"发现了新证据所以推翻结论"，是"判据本身选错了分母"**
——旧判据拿 `epi_fixed` 一个站点的份额当"这条机制值不值得做"的代理，
新判据是"这个机制能盖住的真实 f32 面有多大"。两者在 `epi_fixed` 补账
之前的数字下几乎看不出区别（`epi_fixed` 补账前后的绝对值都没变，只是它
所在的家族——dequant epilogue 整体——补上了 43.5x 之前看不见的部分）。

**仪器口径，本次同样成立，且更该强调**（§7 有完整版本）：
- census 仍然**不计比较运算、不计取绝对值**——`epi_q41`/`epi_q8`/`epi_fixed`
  内部的 amax 扫描、clamp 一概不计，所以下面任何"省下多少"的估算都是
  **方向对、幅度偏低**。
- census **与 ISA 无关**（`--kernel mmx`/`--kernel sse2` 报同一个数字），
  回答不了"在目标机上更快"，只回答"操作数变少"。
- 必须在**出货配置**（`--fixed all`）、**已证明真的在跑**的路径上测——
  `epi_q41` 三个调用点（`matmul_q41_impl`/`matmul_t2_impl`/
  `matmul_q61_impl`）都**不看任何 tier**，无条件跑 float，这点和
  `epi_fixed`/`epi_q8` 靠 `q8_fixed_epi()` 分岔不同，下面会说这意味着什么。

## 1. 头条数字（替换旧版）

- 出货配方（`E:/LLM/models/kmr20`，`.prof/dtype_probe.c` 独立编译独立跑，
  与团队材料给出的数字逐位相同）：

  ```
  dtype          elements  tensors    share
  q8_0            9437184       30     7.1%
  q4_1           44040192      192    33.0%
  q6_1           79167488      241    59.4%
  q16_0            704512       26     0.5%
  total         133349376
  ```

  q4_1+q6_1 = **92.4%** 的权重面，全部只有 `epi_q41` 这一条永远 float 的
  出口。`epi_fixed`（唯一的定点 epilogue）只服务 Q8_0 且只在 `q8_fixed_epi`
  判真时才跑，覆盖的权重面上限是 7.1%，而它的实际 f32 占比（0.55%，见
  §0）比这个上限还低一个数量级——因为 kmr20 上大多数 Q8_0 张量走的是
  `epi_q8`（对称，但从不进 `epi_fixed` 的整数分支），真正落进 `epi_fixed`
  的调用点更少。
- 用扩展过的 `.prof/dtype_probe_kda.c`（新增 `kda_f_a/f_b/b_proj`、
  `gate_proj`/`up_proj`/`down_proj`、全部 MoE 张量的打印行，改动只在
  `.prof/` 下，gitignore 覆盖，不进仓库）逐层打印，确认了旧版 §1 表里
  三类此前只读过 `model.h` 类型注释、没有实测的声明：

  ```
  in_proj_qkv        dtype=f32    n=0        (SSM/GDN 分支，模型 8 层
                                              全是 KDA/full_attention，
                                              forward_ssm 从不执行)
  kda_q/k/v_proj     dtype=q6_1   gs=512     (不是 model.h 注释暗示的
                                              Q8——见下)
  kda_g_proj         dtype=q8_0   gs=512     (会进 epi_fixed/epi_q8)
  kda_o_proj         dtype=q8_0   gs=1024    (同上，但终点是残差加)
  kda_f_a_proj       dtype=q16_0  gs=512
  kda_f_b_proj       dtype=q16_0  gs=64
  kda_b_proj         dtype=q16_0  gs=512     (model.h 注释写"Q8"，实测
                                              是 q16_0——不是唯一一处
                                              注释与实测不符的张量)
  gate_proj/up_proj/down_proj    dtype=f32  n=0   (稠密 FFN 整体不存在;
                                                   模型是纯 MoE)
  moe_gate_w(router) dtype=q16_0 gs=512
  moe_down_proj      dtype=q8_0  gs=512
  moe_up_proj        dtype=q8_0  gs=256
  moe_shared_gate/up/down        dtype=q6_1 gs=512
  moe_expert_w1/w2/w3(层 0)      dtype=q4_1 gs=128
  ```

  **新发现，超出团队材料原来点名的范围**：旧版 §1 表把候选 #24/#25
  （dense FFN 的 `gate_proj`/`up_proj` SwiGLU）列为和 MoE 版 SwiGLU 同一
  家族的候选之一——**这条链路在这个模型上根本不存在**：`gate_proj`/
  `up_proj`/`down_proj` 三个张量 `n=0`，和 `in_proj_qkv` 是同一类错误
  （读 `model.h` 注释以为总有一个 dense FFN 分支，没有实测这个具体
  checkpoint）。这不影响 SwiGLU 重做本身的结论（MoE 的两处，#17/18 与
  #21/22，都是真实存在且已验证会跑的链路——`moe_expert_w1/w3` 是
  q4_1，`moe_shared_gate/up` 是 q6_1，两者都进 `epi_q41`），只是把
  "SwiGLU 三处"改成"SwiGLU 两处（在这个模型上）"。

### 1.1 链路清单（原表，编号不变；分类列按本次实测更新）

产出方式不变（见首版方法说明）：`grep -nE "^\s*lz_matmul_(xq_nt|xq|w)\("
src/forward.c`，48 处调用点合并 `use_subn` if/else 后是 33 条链路。下面
只更新分类列里被本次 dtype 实测推翻或细化的行；其余行（结构性 float、
消费者异质、已完成等）原样保留，编号是后文 §2/§8/§9 引用的唯一依据。

| # | producer（权重张量） | 输出 buffer | 消费者（数量，逐个列出） | 分类 |
|---|---|---|---|---|
| 1 | q_proj/k_proj/v_proj | `s->qg`/`ktmp`/`vtmp` | 3+：`lz_rmsnorm`(QK-Norm)、`lz_rope`、`rot_head`(Hadamard，条件性)、KV cache 的 Q8 写入、KV cache 的 float 参考面写入(`kf`/`vf`，条件性)、`lz_kv4_quantize`(条件性) | **多消费者，且部分本质需要 float**（RoPE 三角函数、Hadamard）——不是候选 |
| 2 | o_proj | `s->xb2` | 1：残差加 `s->x[i] += s->xb2[i]`(`forward.c:3200`/`3220`) | 结构性 float（B9）——**不在本项目范围**，见 §8 |
| 3 | in_proj_qkv | `s->qkv` | 1：`lz_causal_conv1d_step`/`_fixed`(`forward.c:1895`/`1885`) | **已退出优先级**——`.prof/dtype_probe_kda.c` 实测 `n=0`，链路在 kmr20 上不存在，见 §2.3 |
| 4 | in_proj_z | `s->zbuf` | 1：`lz_rmsnorm_gated`/`_int` 的门控输入(`forward.c:2008`) | producer `in_proj_z` 实测 `n=0`（同属 SSM 分支）——推迟，且在这个 checkpoint 上不会跑，见 §9.1 |
| 5 | in_proj_a | `s->avec` | 1：`lz_softplus(s->avec[...] + dt_bias)`(`forward.c:1949`) | producer `n=0`；消费者 `lz_softplus` 是二阶依赖——推迟，见 §9.1 |
| 6 | in_proj_b | `s->bvec` | 1：`lz_sigmoid(s->bvec[...])`(`forward.c:1944`) | producer `n=0`；消费者 `lz_sigmoid` 是二阶依赖——推迟，见 §9.1 |
| 7 | out_proj(SSM) | `s->xb2` | 1：残差加 | 结构性 float，且 producer `n=0`——不在范围 |
| 8 | kda_q_proj | `s->kda_q` | 1：`lz_causal_conv1d_step`(`forward.c:2338`附近) | **已退出优先级**——实测 `q6_1`，不经过 `epi_fixed`，见 §2.3；`epi_q41` milestone 2 是新的复活路径 |
| 9 | kda_k_proj | `s->kda_k` | 1：`lz_causal_conv1d_step`(`forward.c:2341`) | 同上 |
| 10 | kda_v_proj | `s->kda_v` | 1：`lz_causal_conv1d_step`(`forward.c:2346`) | 同上 |
| 11 | kda_g_proj | `s->zbuf` | 1：门控输入，同 #4 | producer 实测 **q8_0**（会进 `epi_fixed`）——本节推迟项里触发条件最松的一条，见 §9.1/§2.5 |
| 12 | kda_f_a_proj | `s->kda_gate_lat` | 1：`lz_matmul_w` 内部的量化(`forward.c:2362`，喂 kda_f_b_proj) | **已退出优先级**——两端实测 `q16_0`，走 `epi_q8` 不走 `epi_fixed`，见 §2.3 |
| 13 | kda_b_proj | `s->bvec` | 1：`lz_sigmoid`，同 #6 | producer 实测 **q16_0**（不是 model.h 暗示的 Q8）；消费者仍是二阶依赖——推迟，见 §9.1 |
| 14 | kda_o_proj | `s->xb2` | 1：残差加 | 结构性 float；producer 实测 q8_0，但终点在残差加——不在范围 |
| 15 | moe_gate_w | `s->moe_router_logits` | 1：`lz_moe_route`(`forward.c:2674`，softmax+top-k) | producer 实测 **q16_0**，走 `epi_q8`——推迟，见 §9.2 |
| 16 | moe_down_proj | `s->moe_lat_x` | 1（每 token 一次，w1/w3 共享同一次量化结果）：`lz_quantize_q8`(`forward.c:2708`) | producer 实测 q8_0，consumer q4_1/q6_1，非对称——推迟，见 §9.3 |
| 17 | moe_expert_w1[ei] | `s->moe_h1` | 1：SiLU 组合 `s->moe_h1[vv]=lz_silu(...)*s->moe_h3[vv]`(`forward.c:2714`) | **候选**（SwiGLU，routed）——producer 实测 q4_1，见 §2.4 |
| 18 | moe_expert_w3[ei] | `s->moe_h3` | 1：同上 SiLU 组合 | 同上 |
| 19 | moe_expert_w2[ei] | `s->moe_h2` | 1：累加进 `lat_y`(`forward.c:2719`) | producer 实测 q4_1；累加目标非残差流而是 latent 累加器，未深入评估，未纳入候选也未排除，见 §8 |
| 20 | moe_up_proj | `s->xb2` | 1：残差加(经 `s->xb2[i] += s->moe_shared_out[i]` 合并共享专家后) | 结构性 float；producer 实测 q8_0——不在范围 |
| 21 | moe_shared_gate | `s->moe_h1` | 1：SiLU 组合(`forward.c:2805`) | **候选**（SwiGLU，shared）——producer 实测 q6_1，见 §2.4 |
| 22 | moe_shared_up | `s->moe_h3` | 1：同上 | 同上 |
| 23 | moe_shared_down | `s->moe_shared_out` | 1：`s->xb2[i]+=s->moe_shared_out[i]`(`forward.c:2819`) | 结构性 float（并入残差流）；producer 实测 q6_1——不在范围 |
| 24 | gate_proj | `s->hb` | 1：SiLU 组合(`forward.c:2934`) | **链路不存在**——producer 实测 `n=0`（模型无 dense FFN 层），见 §2.4 |
| 25 | up_proj | `s->hb2` | 1：同上；`use_subn` 时还有 `subn_fwht_quant` 读它做 down_norm 的 Hadamard 输入(`forward.c:2966`) | 同上，**链路不存在** |
| 26 | down_proj | `s->xb2` | 1：残差加 | 结构性 float；producer 实测 `n=0`——不在范围（且链路本身不存在） |
| 27 | mtp.fc(第一次调用) | `s->mtp_x` | 1：`lz_rmsnorm` 进入 MTP 自己的 `forward_attn`(`forward.c:3028`)；随后又被当作 MTP 自己的残差流两次 `+=`(`3038`/`3044`) | `s->mtp_x` 起 MTP 残差流的作用，与 B1 同构——不在范围 |
| 28 | embed_tokens(MTP draft logits) | `s->mtp_draft_logits` | MTP 草稿 logits，供外部推测采样比较 | 面向调用方的最终数值——不在范围 |
| 29 | embed_tokens(主 lm_head, 批量) | `logits_out` | 面向调用方 | 不在范围 |
| 30 | embed_tokens(主 lm_head, 单 token) | `s->logits` | 同上 | 不在范围 |
| 31 | mtp.fc(第二次调用点, `forward.c:3512`) | `x_t` | 与 #27 同构（MTP 另一个入口的残差流） | 不在范围 |
| 32/33 | q/k/v_proj 的 SubLN 分支(`p[pi].out`) | 同 #1 三个 buffer | 同 #1 | 同 #1，不是独立链路，只是同一 buffer 的另一条产出路径 |

## 2. 重排后的候选

### 2.1 头号目标：`epi_q41` 的定点/int 化（Q4_1/Q6_1/T2 共享的 dequant epilogue）

`epi_q41`（`ops.c:1563`，被 `matmul_q41_impl` `ops_matmul.c:595`、
`matmul_t2_impl` `ops_matmul.c:867`、`matmul_q61_impl` `ops_matmul.c:925`
三处调用）算的是:

```
o[i] = Σ_g [ Σ_{s∈g} lz_i32f(accb[g*r+s]) * xsb[g*r+s] ] * wd[g]   (dotsum)
     + Σ_g xgb[g] * wm[g]                                          (zsum)
```

`accb` 是 int32 的整数点积累加器（和 `epi_fixed`/`epi_q8` 共享同一种
row-kernel 产出），`wd[g]`/`wm[g]` 是这一组的浮点 scale 和 min（**非对称
格式，这是和 Q8_0 的本质区别**），`xgb[g]` 是纯激活侧的零点项
（`Σ_{s∈g} xqs[s] * Σ_k xq[k]`，每 token 算一次，被这一行所有输出复用）。

**dotsum 这一半，可以直接复用已交付的对齐原语，不需要新设计**：
`lz_epi_prep`（`ops.c:879`）本身不看 dtype——它读 `w->scale`/`w->gs`/
`w->n`，把每行的 scale 组量化成 int16（`w->sq`）+ 一个共享的整数指数
（`w->sexp[row]`），这套逻辑对 Q4_1/Q6_1 的 `w->scale` 同样成立。当前
它只被 `q8_fixed_epi`（`ops_matmul.c:243`，只在 `matmul_q8_impl` 里调用）
触发，纯粹是调用点没接，不是算法只认 Q8_0。`epi_fixed_raw`（`ops.c:1124`，
`70f4130` 从 `epi_fixed` 里拆出来的共享 MAC）按 `(ng, r, w->sq, w->sexp)`
工作，`epi_q41` 自己的分组形状（`ng` 组、每组 `r` 个 32-宽子块）与
`epi_fixed_raw` 的参数化完全对齐——**把 `epi_q41` 的 dot/scale 内层循环
换成一次 `epi_fixed_raw` 调用是直接可行的复用，不是"形状相似,仍要重写"**。
唯一需要新增的接线：`epi_pack_act`（`ops_matmul.c:260`，目前只在
`matmul_q8_impl` 里跑，每 token 544 ops、可忽略）要在 q4_1/q6_1/t2 三个
调用点也跑一次，`lz_epi_prep` 要在这三处权重上也被触发一次（一次性，
不是每 token）。

**zsum 这一半，是团队材料点名的"非对称格式"真问题，对齐原语完全不覆盖**：
`wm[g]` 是加法偏移量,不是乘法 scale——`epi_fixed`/`epi_align_i16` 的整
套"取行内 scale 的公共指数,右移到整数域"技巧对**乘法**量成立,对**加法**
偏移量不成立(偏移量要先落到和 dot 项相同的量纲才能相加,而不是先取
自己的指数)。`xgb[g]` 本身还是一个要另外维护的运行时量(每 token 由
纯整数的激活值算出,当前直接用浮点乘加)。给 zsum 找一个整数域表示,是
一个全新的设计,不是"跟 dotsum 一样套一遍现成函数"能解决的。

**第一个可证明的里程碑**（milestone 1）：只把 dotsum 换成
`epi_fixed_raw` + 一次浮点转换，zsum 保持现在的浮点算法不变，两者在
函数末尾相加——`return (float)s * pow2f(e - 8) + zsum;`。这**不需要**
解决零点项的整数化，也**不需要**让结果本身携带 int 穿过 epilogue 边界
（返回值仍然是 float，和现在一样）；它省的是 dotsum 内层循环从"每个
32-宽子块一次 `lz_i32f` 转换 + 一次浮点乘 + 一次浮点加,再乘一次组
scale"换成"一次整数 MAC + 一次浮点转换"。用 `epi_q41` 自己的记账公式
（`ng*(r+2)+1` mul/add，`ng*r` cvt，`ops.c:1577`）拆开：dot/scale 部分是
`ng*(r+1)` 的 mul/add 和**全部** `ng*r` 的 cvt，zero 部分只有 `ng` 的
mul/add。取这个模型实测到的两种 `r`（`r=gs/32`）：`gs=128`（层 0 的
`moe_expert_w1/w2/w3`)得 `r=4`,dot/scale 部分占
`(4+1)/(4+2)=83.3%`；`gs=512`（`moe_shared_gate/up/down`,`kda_q/k/v`)
得 `r=16`,占 `(16+1)/(16+2)=94.4%`——不管这两类张量各自在 101,376 次
调用里占多少比例（未拆解，不假设），milestone 1 在**每一次**调用上都能
拿掉八到九成的记账量，且**不触碰**零点项这个真正难的部分。

**数值风险，milestone 1 就要面对，不能假设"和 Q8_0 一样"**：`LZ_EPI_MAX_NG`
（`ops.c:864`）的推导是"per group the product is |acc| * |xw*ww| <=
2^27 * 2^30, so ng of them stay inside int64 while ng <= 64"——这个界是
按 Q8_0 的**有符号** int8×int8（±127）算的。Q4_1 的权重码是**无符号**
nibble（0..15），Q6_1 是无符号 6-bit（0..63），量级比 Q8_0 更窄不是更宽，
但**这只是直觉，不是证明**——在把 `epi_fixed_raw`/`epi_align_i16` 接到
`matmul_q41_impl`/`matmul_q61_impl` 之前，要重新推导（或至少扫一遍实际
checkpoint 的权重+激活极值）确认 `ng<=64` 这个界仍然覆盖 Q4_1/Q6_1 的
真实累加范围，不能直接搬 Q8_0 的推导。

**第二个里程碑**（milestone 2，真正的"int 贯通"）：给 dotsum 换成
`epi_fixed_align_i16` 风格的 int16 出口（对齐到调用者给定的共享指数），
**并且**把 zsum 也折进同一个整数域（需要新设计：给 `wm[g]` 一个类似
`lz_epi_prep` 的定点分解，给 `xgb[g]` 一个类似 `epi_pack_act` 的整数
表示）。做到这一步之后,`epi_q41` 才真正具备了对外吐 int 而不物化 float
的能力——**到那时**，候选 #8/#9/#10（`kda_q/k/v_proj → conv`，已实测是
q6_1，见 §1）会通过 `epi_q41` 的这个新出口重新变得可达，而不是通过原来
为它们设计的 `epi_fixed` 路径（那条路径对它们从一开始就走不通，见
§2.3）。milestone 2 是否值得做,取决于 milestone 1 量出来的真实收益,以及
是否能找到一个真实的、milestone 2 能接上的整数消费者（同样的"census 记
不到、需要证明路径真的在跑"的约束，§0/§7 已经说明）。

**milestone 2 状态：已交付（浮点出口这一半）。** 落地的形状和上一段的
设想有一处偏差,要记下来：零点项确实拿到了 `lz_epi_prep` 式的定点分解
（`w->zq`/`w->zexp`，与 `sq`/`sexp` 共用一个 `epi_prep_plane`）和
`epi_pack_act` 式的整数激活表示（`epi_zero_act_int` → `g_epi_zg`），
两个 int64 部分和在整数域按公共指数对齐后**只物化一个 float**——但
**出口仍然是 float，不是 int16**。`epi_fixed_align_i16` 式的 int 出口
没有接,因为 §2.1 自己给的前置条件（"找到一个真实的、能接上的整数
消费者"）没有满足;把它接出来是 #8/#9/#10 复活时的下一步,不是这一步。

关键的量化结果（kmr20，`--fixed all`，两条长语料）：`ng == 1` 占
81.8% 的调用,所以"两项各自一个 float 出口"（2 cvt + 2 mul + 1 add = 5）
在多数调用上等于 `epi_q41` 自己的 `2ng+3 == 5`,**零收益**——整数域合并
才是收益与 `ng` 无关的原因。dequant epilogue 记账 710,276 → 323,856
（-54.4%），总量 1,649,610 → 1,263,190（-23.4%）。省下的是 zsum 循环
（每次调用 `ng` mul + `ng+1` add）与浮点版 `epi_zero_act`（定点路径上
不再运行）。对齐方向由实测决定：零点面的指数在 100% 的调用上更粗,所以
永远是把 dot 和右移,而右移的前 8 位是内核 x256 的精确抵消（实测
953,024,512 个 acc32 项无一不是 256 的倍数），真正有舍入的只有 0..6 位。
数值上不是退让而是改善：对同一浮点参照的 rms 相对误差 6.94e-5（M1）→
4.29e-5（M2）。

**闸门**：milestone 1（float 出口不变，只换内部算法）是**value-changing**
的——`epi_fixed_raw` 的整数 MAC 累加顺序和现在 `epi_q41` 逐子块浮点累加
的顺序不同,不是 bit-identical 的重构，要走和其它定点档一样的门：新 tier
内部 gcc/Watcom 逐位一致（铁律二），配对 NLL/argmax/top-5（不是
`|logits diff|`，§6 已有的教训直接适用），长序列语料（同样是
`norm_int_gate_tokens_{zh,en}.txt` 这一类，不能只在短探针上过）。census
上的验证要用 `5d852a2` 已经示范过的方法——用一条独立的、从 LZ_FCX 参数
之外重新推导的计数器复核，不能只信一次记账。

### 2.2 已交付但未接线的对齐原语（`epi_align_i16`/`epi_fixed_align_i16`）

**状态：已交付（`70f4130`），已独立验证，故意未接线。** `epi_align_i16`
（`ops.c:1229`）把 `epi_fixed` 的 `(s, e)` 折到调用者给定的共享指数/边界
下的一个 int16，纯整数运算，不经过任何浮点转换，因而不需要 `lz_i32f`/
`volatile` 那类 gcc-vs-Watcom 舍入两次的护栏——这条性质是"用整数移位而
不是浮点乘除做对齐"这个设计选择的直接结果，不是额外证明出来的。独立于
长双精度参照验证过 1,647,376 组输入（含 `LLONG_MIN`/`shift==-64` 角落，
`.prof/verify_epi_align.c`），0 不符。`epi_fixed_align_i16` 是它的
epilogue 形状包装（`ops.c:1268`），和 `epi_fixed` 同一套调用签名，
没有接到 `forward.c` 的任何调用点。

**它最初是为 #3/#8/#9/#10/#12 准备的，那四条链路现在都已知走不到
`epi_fixed`**（见 §2.3，用 §1 的 dtype 实测核实），所以这个原语**目前
没有已知能付得起它的消费者**——但它并不是白做的：`epi_fixed_raw`（它
和 `epi_fixed`/`epi_align_i16` 共享的 MAC）的 `(ng, r, w->sq, w->sexp)`
参数化，正是 §2.1 milestone 1/2 里说的"可以直接复用"的那部分。**它现在
更准确的定位是 epi_q41 项目的一个组件，不是被退回原地的死代码**——
milestone 1 需要它（`epi_fixed_raw`），milestone 2 需要它的完整包装
（`epi_fixed_align_i16`/`epi_align_i16`）。原语本身不需要改一行就能被
复用；需要新写的是 §2.1 说的"让 `lz_epi_prep`/`epi_pack_act` 也在
Q4_1/Q6_1/T2 的调用点上跑"这一层接线，以及 milestone 2 的零点项设计。

### 2.3 已退出优先级：#3/#8/#9/#10/#12（原排序的头两名和第四名）

用 §1 的实测数据逐条核对：

- **#3**（`in_proj_qkv → conv`）：`in_proj_qkv` 在 kmr20 上 `n=0`——模型
  8 层全是 `linear_attention (kda)`/`full_attention`，没有一层是普通
  GDN/SSM（`forward_ssm` 从不被调用），这条链路在这个 checkpoint 上
  **不存在**，不是"存在但小"。
- **#8/#9/#10**（`kda_q/k/v_proj → conv`）：三者实测都是 `q6_1`
  （`gs=512`），不是 Q8_0。`epi_fixed` 只被 `matmul_q8_impl` 调用
  （`ops_matmul.c:243` 的 `q8_fixed_epi` 只在那一条路径上被问），
  q6_1 走 `matmul_q61_impl` → `epi_q41`，从一开始就不经过 `epi_fixed`。
  `70f4130` 的提交信息已经记录了这一点，本次用独立编译的探针复核，数字
  一致。
- **#12**（`kda_f_a_proj → kda_f_b_proj`）：两端都是 `q16_0`
  （`gs=512`/`gs=64`），走的是 `epi_q8` 分支（`q8_fixed_epi` 对 q16_0
  的判据从不为真，见 §0 的位点表），不是 `epi_fixed`。和 #8/#9/#10 是
  同一类原因，本次用扩展过的探针独立复核。

**这四条候选本身没有被否证——原来对它们的接口改动、风险、验证方式的
分析（旧版 §2.2–§2.4 的内容）在"如果它们真的在跑 `epi_fixed`"这个前提
下仍然成立**，只是这个前提在这个 checkpoint 上不成立。它们退出优先级
的理由是"当前证据下没有可付费的路径"，不是"设计错了"：

- #8/#9/#10 有一条**新的**复活路径——§2.1 milestone 2 如果做出一个
  `epi_q41` 的 int 出口，它们可以通过那个出口而不是 `epi_fixed` 变得
  可达（producer 是 q6_1，`epi_q41` 恰好服务 q6_1）。
- #3 除非未来有一个 checkpoint 真的用了普通 GDN/SSM 层（`in_proj_qkv`
  非空），否则在"这个项目度量所依据的模型"上没有意义；这不是这个
  checkpoint 独有的巧合——`kmr20` 是 KDA 架构，普通 GDN 层本来就是这个
  模型家族里的次要分支，换一个 checkpoint 未必更好。
- #12 的收益本来就小（§2.4 旧分析：0.4% 解析估算），加上现在确认它走
  `epi_q8` 不走 `epi_fixed`，如果要救它，前提是先给 `epi_q8`（q16_0 的
  唯一出口）也设计一个定点/int 分支——那是和 §2.1 milestone 2 平行的
  另一个小项目，不是免费搭车。

### 2.4 SwiGLU 重做（两处，不是三处）

内容与旧版 §2.6 相同（`b2cfb0e`/`58fc52e` 的两个具体实现缺陷、修复方向、
必须复用 `lz_q8round32_sse`/`lz_q8round_group_sse2` 的约束），**唯一的
更正是范围**：旧版把它列为"三处"（MoE routed #17/18、MoE shared #21/22、
dense FFN #24/25），本次用 §1 的探针核实 `gate_proj`/`up_proj`/
`down_proj` 三个张量在 kmr20 上 `n=0`——**这个模型没有 dense FFN 层，
每一层的 FFN 都是 MoE**。SwiGLU 重做因此是**两处**：MoE 的 routed 专家
（`moe_expert_w1`/`w3` 是 q4_1，`gs=128`）和 shared 专家
（`moe_shared_gate`/`up` 是 q6_1，`gs=512`）。两处的 producer 矩阵都
经过 `epi_q41`——**这条重做线和 §2.1 是互补的，不是竞争的**：§2.1 省的
是 `epi_q41` 自己内部算 dotsum 的浮点开销，SwiGLU 重做省的是 `epi_q41`
算完之后、`lz_silu` 组合与 `lz_quantize_q8` 之间那次多余的 float 具现化
——两者可以完全独立地并行推进，谁先做完都不影响另一条线的正确性。

优先级仍然次于 §2.1（§2.1 覆盖的面大一个数量级以上），但**仍然是全项目
里唯一不需要等待任何新设计就能动手的一项**——缺陷已知、修复动作已知、
验证公式已知，是"重做，这次做对"里置信度最高的一项，这一点没有变。

### 2.5 候选 #16、#4/#5/#6/#11/#13、#15：结论未变，补一条修正

`moe_down_proj(q8_0) → moe_expert w1/w3(q4_1/q6_1)` 这条非对称 matmul→
matmul 链路（#16）、z 门控/avec/bvec 家族（#4/#5/#6/#11/#13）、router
（#15）——旧版 §2.5/§2.7/§2.8/§8/§9 对这些链路的分析和触发条件不变。

**一处需要补充**：§9.1 把 #4/#11（`zbuf` 门控输入）当一个整体处理，触发
条件写的是"等 §2.1 的对齐原语完成"。用本次的探针核实后，#11 的 producer
`kda_g_proj` 实测是 **q8_0**（不是 q6_1），本来就会经过 `epi_fixed`——
如果 `--fixed all` 且 `q8_fixed_epi` 判真，`kda_g_proj` 的输出已经是
`epi_fixed` 的整数域计算结果，只是最后一步转成了 float。这意味着 #11
的触发条件比 #4（producer `in_proj_z`——同样 `n=0`，因为这也是 SSM 分支
的一部分，不是 KDA 的）更容易满足：#11 不需要等 §2.1/§2.2 的任何新工作,
它现在就能拿到 `epi_fixed` 的 `(s,e)`,缺的只是 `lz_rmsnorm_gated` 长一个
int-in 入口（§9.1 已经分析过这部分独立于哪个 producer）。这是"再评估
触发条件"的一处具体化，不改变 §9 的整体结论（仍然是推迟，不是否决）。

## 3. 排序（重排后）

1. **`epi_q41` milestone 1**（§2.1）——新的头号目标。理由：它覆盖的
   f32 面（84.9% 的补账后总量）比原头号目标 `epi_fixed`（0.55%）大两个
   数量级，dotsum 部分可以直接复用已交付的 `epi_fixed_raw`（§2.2），
   不需要等待任何新算法设计，只需要接线（`lz_epi_prep`/`epi_pack_act`
   扩展到三个新调用点）+ 重新核验 `LZ_EPI_MAX_NG` 的界对 Q4_1/Q6_1 是否
   仍然成立。覆盖两条已确认在跑的真实生产路径（`moe_expert_w1/w2/w3`、
   `moe_shared_gate/up/down`，以及 `kda_q/k/v_proj` 的 producer 端）。
2. **SwiGLU 重做（两处）**（§2.4）——和 1 并行，不依赖它，缺陷/修复/
   验证公式都已知，是置信度最高的一项，理由和旧版相同。
3. **`epi_q41` milestone 2**（§2.1）——**浮点出口这一半已交付**（见
   §2.1 的状态段）；剩下的 int16 出口仍然按原条件挂着，即先找到一个
   真实的整数消费者，那也正是给 #8/#9/#10 的第二条复活路径。
4. **#16（`moe_down_proj → expert`）**——不变：需要 milestone 2 级别的
   零点项设计（这里是 producer 侧对称的 q8_0，consumer 侧非对称，和
   `epi_q41` 自己的非对称问题是同构但不同的两个零点设计,不能指望共用）。
5. **#4/#5/#6/#11/#13、#15、#12**——推迟，触发条件见 §9（§2.5 补充了
   #11 的一处细化）。#3/#8/#9/#10 单独列出：#3 在可预见的将来都没有
   触发条件（依赖一个这个项目没有在用的 checkpoint 特征）；#8/#9/#10
   的触发条件现在是"milestone 2 完成"，见第 3 项。

## 4. 逐阶段 go/no-go 判据（新增 `epi_q41` 两档，其余不变）

**`epi_q41` milestone 1**
- go：`lz_epi_prep`/`epi_pack_act` 接到三个新调用点后，`LZ_EPI_MAX_NG`
  的界对 Q4_1/Q6_1 的真实累加范围重新推导或实测确认仍然成立；新 tier
  在 gcc/Watcom 逐位一致；用 `5d852a2` 同款的"独立重新推导 + 比对 LZ_FCX
  账单"方法验证 census 降幅与 §2.1 给出的 `(r+1)/(r+2)` 公式吻合；配对
  NLL/argmax/top-5 在长序列语料上通过。
- no-go：`LZ_EPI_MAX_NG` 的界对某个真实权重张量不成立（需要额外的溢出
  处理，侵入 `epi_fixed_raw` 本身）；或者 census 降幅和公式对不上（说明
  接线有遗漏或误算，同 `matmul-int-output-design.md` 第一次"−9.9%"是
  仪器假象的教训）。
- 一旦通过：**如果时间/精力有限，milestone 1 单独就是一个完整可交付的
  结果**——不必等到 milestone 2 才算数，86.9% 的位点里能拿掉八到九成
  已经是这个项目迄今为止最大的一笔。

**`epi_q41` milestone 2**
- go：milestone 1 已经 go，且有人给零点项设计出了一个整数域表示（`wm[g]`
  的定点分解 + `xgb[g]` 的整数表示），标准的铁律二双编译器逐位一致门 +
  长序列配对判据都通过，且找到一个真实会跑的 int 消费者（不能重复
  scheme A"造了但没人接"的教训——`f49da09` 的教训在§5 有完整记录）。
- no-go：零点项的整数表示做不到和现有浮点算法数值等价（在长序列上配对
  判据的下降幅度超过定点档自身相对 float 的下降），或者找不到真实消费
  者——这种情况下 milestone 1 的成果原样保留，milestone 2 无限期搁置，
  不算失败。
- 实际交付把这条判据劈成了两半：零点项的整数表示这一半 go 了（双编译器
  逐位一致、四条语料的配对判据、census 独立重推导都过，且相对浮点参照
  的 rms 误差比 milestone 1 更小），**int 消费者这一半没有 go**，因为
  §2.1 要求的"真实会跑的 int 消费者"仍然不存在。所以出口留在 float，
  没有按"造了但没人接"的方向多走一步。

其余（#3/#8/#9/#10 旧分析、SwiGLU、#16、#12）的 go/no-go 判据与旧版相同，
§2.3/§2.4/§2.5 已经标出了各自现在依赖的前置条件，此处不重复。

## 5. 不要重复的三个错误（不变，均在 git 历史里，已逐条 `git show` 核实）

1. **`9569da9`：trunk-norm int 加在 float 旁边,不是替换它。** 提交信息
   给出了精确数字：打开这个开关净亏 +49562 f32 ops/token(+5.4%)，norms
   168020→196788，quantize 101112→121848，rsqrt 1117→1174。根因不是编码
   错误，是 trunk 的两处 norm（`input_layernorm`/`post_attention_
   layernorm`）有多个不同权重格式、不同 group size 的消费者，其中一个
   F32 消费者直接读 `s->xb`，导致 float 版 `lz_rmsnorm` 永远不能省——
   `lz_rmsnorm_int` 于是是**额外**跑一遍，不是**替代**。已完工的 gated/
   SubLN int 路径能省，恰恰因为各自只有一个消费者、float 输出真的能丢。
   **本计划§1 的候选表就是照这条教训筛出来的**——"consumer 数 = 1"是
   硬门槛,不是建议。（注：`docs/fixed-tier-pipeline-breakpoints.md` §3
   在"出货无条件定点"这个新前提下重新核过这条——出货配方上确实没有任何
   投影角色是 F32,所以 `9569da9` 描述的那个可能性在新前提下不成立;但
   +5.4% 的回归数字本身是真的,根因是"总是无条件跑 float 再叠加 int"这个
   实现选择,不是数据强制的硬约束。这条与本次 `epi_q41` 重排无关,列在
   这里是因为它是"读了但没有重新核实就当结论用"的同一类风险的另一个
   例子——本文档同样应该被下一次重排前先核实,而不是当作永久真理。）
2. **`f49da09`：scheme-A matmul int 机制作为 `epi_fixed` 的替代品,而不是
   建在它上面。** 两个入口函数一遇到 `q8_fixed_epi(...)` 为真就直接
   `return 0` bail——而 `lz_epi_mode()` 的 auto 恰好在需要它的无 SSE
   机器上打开 `epi_fixed`,方向整个是反的：`--fixed all` 下开关 on/off
   连 top logit 都逐位相同。**§2.1/§2.2 因此明确要求 `epi_q41` 的新方案
   接在 `epi_fixed_raw`/`epi_align_i16` 之上,不是再造一套独立的累加器**
   ——这条约束现在直接适用于新的头号目标,不只是旧候选。
3. **SwiGLU 融合（`b2cfb0e`，被 `58fc52e` 打回）：标量融合量化输了给已有
   的 SIMD 分发 round 路径。** 见 §2.4 的完整分析——**任何 int 贯通的
   重新设计都必须继续走到 `lz_q8round32_sse`/`lz_q8round_group_sse2`，
   不能用一个新写的标量循环替换它**，这条约束同样适用于 `epi_q41` 的
   新方案：`epi_fixed_raw` 本身已经是复用现成 MAC（`lz_epi_mac`）而不是
   新写标量循环，milestone 2 的零点项设计如果要新写取整/量化代码，
   同样不能绕开这条铁律。

## 6. 每个阶段都适用的正确性约束（不变）

- **`int32→float` 转换只能走 `lz_i32f`，`int64→float` 必须经过一个
  `volatile float` 中间量**——裸转换在 gcc/SSE 和 Watcom/x87 上舍入次数
  不同（SSE 转换一次、乘法再舍入一次；x87 `fild`+`fmul` 融合只舍入一次）。
  这不是假设性风险：`build/fixed_tier_xcompiler_gate.sh` 的文件头注释记着
  一次真实事故——`lz_quantize_q8_int` 的快速分支曾经裸转 `grp[k]`，在
  kmr20 模型、153 token 里的第 146 个位置（layer 0, group 1）才第一次分歧,
  更短的探针和纯英文输入都测不出来。任何本项目新写的 int→float 转换点
  都要过这同一个门——`epi_align_i16` 本身刻意绕开了这个问题（全程整数
  移位,不经过浮点转换,见 §2.2），milestone 1 的"一次浮点转换"（`(float)s
  * pow2f(e-8)`）沿用的是 `epi_fixed` 已经验证过的同一条转换,不是新的。
- **跨编译器逐位一致是硬约束，不是容差**——CLAUDE.md 原话:"gcc-fixed and
  Watcom-fixed must not [differ]"。任何新 tier/新路径都要在两个工具链上
  逐位核对，不是"看起来数值接近就算过"。
- **`|logits diff|` 不是判据，它饱和**——判据是配对 NLL、argmax 一致率、
  top-5 一致率（`--dump-nll`/`--dump-top5`）。
- **逐位一致的门必须用长序列、且不止一种语料**——`norm-write int` 这个
  缺陷本身就是反例：8/16 token 探针和 211-token 纯英文样本都是干净的，
  在 153-token 中文样本的第 146 个位置才第一次出现分歧。任何本项目新增
  的门都要复用或扩展 `build/norm_int_gate_tokens_{zh,en}.txt`（603/661
  token）这一类长且真实的语料，不能只在短探针上宣称"通过"。

## 7. 仪器诚实声明（更新：加入 `epi_q41` 记账缺口）

- **`epi_q41` 在 `5d852a2` 之前完全零记账，任何在此之前测过的"dequant
  epilogue"或"总 f32 量"数字都低估了 43.5x / 84.9 个百分点**——这不是
  "低估了一点"，是"这个位点在补账前的任何数字里实质上是缺席的"。本文档
  上一版（重排前）的排序判据正是建在这个缺口之上：把 `epi_fixed` 的份额
  当成"这类机制值不值得做"的代理，而 `epi_fixed` 本身在补账后的真实份额
  只有 0.55%。**任何引用本文档旧版数字的人，第一步要确认引用的是补账
  前还是补账后的版本**。
- **census 在 `LZ_FC_QUANT` 位点(`f18f02d`)加入之前，quantize 这块工作
  完全不计费**（这条与 `epi_q41` 的缺口是同一类问题的更早一次实例，
  规模不同：quantize 缺口约 11%，`epi_q41` 缺口是 43.5 倍）。那次提交
  自己量过：同一个 `/tmp/kmr20 --fixed all` 8-token 跑，总数从 820945
  涨到 919945（+99000，相对新总数约 10.8%）。本项目如果要引用某个
  "之前测过的总量"做对照基线，必须确认那次测量是在哪次记账修复**之后**
  做的，否则基线本身就低了——这条规则现在管两个已知缺口，不止一个。
- **census 不计比较运算、不计取绝对值**，因此**系统性高估**任何"消除一次
  amax 扫描或 clamp"的收益。`missing-facilities.md` 在软浮点 ARM 上实测
  过反例：一次浮点 amax 扫描花 0.409（QEMU 墙钟单位），而**被计费**的浮点
  平方和臂只花 0.247——比被计费的工作量还大 1.6 倍，census 却记 0。
  §2.1 milestone 1 给出的 `(r+1)/(r+2)` 估算只算了 `epi_q41` 自己
  `LZ_FCX` 记的 mul/add/cvt，不含它内部（和 `q8_amax`/`q8_group_scale`
  同款的）任何比较运算——报数字时要连这句话一起报。
- **census 与 ISA 无关**：`--kernel mmx` 与 `--kernel sse2` 报同样的数字。
  它回答"操作数少了多少"，回答不了"在目标机上是不是真的更快"——本项目
  没有铁律里点名的 WinChip 模拟器台架，所以任何"更快"的表述都要明确降级
  成"操作数更少"。
- **必须量在出货配置（`--fixed all`）上，不是默认配置**，且要证明测量的
  路径真的在跑。`epi_q41` 的三个调用点**不看任何 tier 谓词**，无条件跑
  float——这和 `epi_fixed`/`epi_q8` 靠 `q8_fixed_epi()` 分岔不同，意味着
  `epi_q41` 的问题不会像 scheme A 那样"在需要它的机器上被自己的判据关掉"
  （`matmul-int-output-design.md` 记录的教训），但也意味着 milestone 1/2
  一旦接上，需要**自己新增**一个 tier 谓词（不能假设复用 `q8_fixed_epi`
  ——那个判据是为 Q8_0 设计的，字面上会在 Q4_1/Q6_1 的调用点里读到错误
  的权重字段）。

## 8. 本项目问题定义之外的部分（不是"还没排上"，是压根不问这个问题）

下面几项和 §9 的推迟项不同：它们不是被 §2.1 的原语或某个算子接口挡住、
等条件满足就能重新评估的候选，是这个项目要回答的问题（某条链路能不能
在批次一的机制下省掉一次 float 物化）对它们根本不适用，或者适用的前提
是打开一个远大于本项目的题目。

- **残差加**（`s->x[i] += s->xb2[i]` 及其在 MTP/MoE 共享专家合并里的
  变体，§1 表里 7 条终点在这里的链路）——结构性 float，没有 int 版本，
  交付材料点名排除，核实后确认没有反例。
- **`q_proj`/`k_proj`/`v_proj` 的输出**（§1 #1）——消费者集合是 RoPE 三角
  函数、Hadamard 变换、三种 KV cache 格式的并行写入，其中至少两个（RoPE、
  Hadamard）在当前实现范围内没有整数版本、也不是这个项目要新开的题目。
- **LM head / MTP 草稿 logits**（§1 #28/29/30）——面向调用方的最终数值，
  下游是采样、`--dump-nll`、推测解码比较，这些语义明确需要浮点。
- **`lz_softplus` 自己的定点档**——不受任何 `--*-write` 轴或
  `--scalar-fixed` 影响，永远走 double libm；给它建一个定点实现是比本
  项目大一级的题目（改的是算子本身，不是链路两端的接线），不在本计划
  范围，若以后要做应该是它自己的立项。**注意这和 §9.1 里"avec 能不能把
  int 交给 lz_softplus"是两个不同的问题**——后者只需要 softplus 长一个
  int-in 入口且仍然内部转 float 算，前者是让 softplus 本身变成定点算法；
  §9.1 的 #5 之所以比 #4/#6/#11/#13 门槛更高，正是因为它的触发条件里
  牵扯到这一条，而 #4/#6/#11/#13 不需要。
- **`s->moe_h2`→`lat_y` 累加**（§1 #19）——目标是 latent 空间的累加器而
  非残差流，未深入评估其数值语义（是否可能像残差加一样结构性需要 float
  以避免多专家累加误差放大），本文档不下结论，留给专门评估。

## 9. 推迟到第二批的项目（不是否决，是带触发条件的再评估清单）

批次一改变的是这些项目的**成本基础**，不是它们的价值：它们现在被挡住，
要么是因为 §2.1 的指数对齐原语还不存在，要么是因为它们的消费者本身是
结构性 float 算子——而这两类障碍批次一都会往前推一步（前者直接产出，
后者把"接口要接什么、下游怎么拿到一个可用的整数表示"从假设变成已验证
的东西）。铁律九：有争议的选择应该变成一个带默认值的旋钮，不是一次性
否决——下面每一项都写明让它翻盘的具体条件，不是一句"以后再说"。

### 9.1 `s->zbuf`/`s->avec`/`s->bvec` 的门控输入（#4/#5/#6/#11/#13）

消费者是 `lz_rmsnorm_gated`（z 门控侧）、`lz_sigmoid`、`lz_softplus` 三个
结构性 float 算子，§2.7（旧版）分析过为什么现在挡住。触发条件按算子拆开，
**不是均匀的**：

- **`lz_sigmoid`（#6/#13，bvec→beta 门控）**：触发条件是 §2.1 的对齐
  原语（`epi_fixed_raw`/`epi_align_i16`）在某个真实消费者上完成接线并
  验证通过——现在最可能先满足这个条件的是 `epi_q41` 的 milestone 2
  （§2.1），不是原来设想的 #3。理由不变：`sigmoid_q15`/`sigmoid_q15_t`
  已经是树里现成的、被两处真实调用的 Q15 定点 sigmoid 实现
  （`forward.c:1771`、`ops.c:1870`），缺的只是一个 int-in 入口。
- **`lz_rmsnorm_gated` 的 z 门控侧（#4/#11）**：**本次重排细化**——#11
  （`kda_g_proj → zbuf`）的 producer 实测是 **q8_0**，本来就会经过
  `epi_fixed`（不需要等 `epi_q41`）；#4（`in_proj_z → zbuf`）的 producer
  是 `in_proj_z`，和 `in_proj_qkv` 同属 SSM 分支，在 kmr20 上同样
  `n=0`，不会跑。`lz_rmsnorm_gated_int`（B2）已经存在，只是 `zbuf`
  参数仍然是 float（`forward.c:1993-1997`）——对 #11 而言，缺的只是
  给这一个参数补上 int 入口，不需要等任何新的对齐工作；对 #4 而言，
  和 #3 一样，在这个 checkpoint 上没有意义。
- **`lz_softplus`（#5，avec→衰减门控）**：producer `in_proj_a` 同样
  `n=0`（SSM 分支）。即使换一个有 SSM 层的 checkpoint，触发条件也仍然
  最高——树里没有任何地方给 `lz_softplus` 做过定点实现，不像 sigmoid
  有 `sigmoid_q15` 可以现成扩展；即使对齐原语完全就位，还需要有人先
  单独决定要不要给 `lz_softplus` 本身开一个定点档（见 §8）。

### 9.2 `s->moe_router_logits` → `lz_moe_route`（#15）

producer `moe_gate_w` 实测是 **q16_0**（§1），走 `epi_q8` 不走
`epi_fixed`/`epi_q41`。消费者单一但做离散 top-k 排序决策，`ne`（专家数）
规模小。触发条件有两条独立的路，**任意一条满足就够格重新排期**：

- **成本路**：给 `epi_q8`（q16_0 的唯一出口）设计一个定点/int 分支——
  这本身没有在 §2.1/§2.2 的范围内（`epi_q8` 和 `epi_fixed`/`epi_q41`
  是三个不同的函数），是一个平行的小项目，不是免费搭车。
- **风险路**：有人专门测过量化噪声是否会翻转 top-k 选择，证明"不翻转"
  或"翻转的频率不影响下游质量"——这条独立于成本路。

在这两条都还没有之前不建议排期，原因是"还没有证据"，不是"结构性
不可行"。

### 9.3 `s->moe_lat_x`(moe_down_proj) → expert w1/w3 的量化（#16）

producer `moe_down_proj` 实测是 q8_0（`gs=512`），consumer
`moe_expert_w1/w3` 是 q4_1/q6_1（§1）——两端 dtype 不对称这一点没有变，
触发条件也没有变：**两个都要满足**——(a) `epi_fixed`（producer 侧）和
`epi_q41`（consumer 侧的零点项设计，milestone 2）都要先分别成立，
(b) 有人单独设计并验证了这条链路自己的零点项处理方式——不能假设
`epi_q41` milestone 2 的零点项设计能直接复用在这里：`epi_q41` 的零点项
是"权重的 min 项"，这里如果要做，是"consumer 的量化格式本身非对称"，
是形状相似但输入不同的两个设计问题。
