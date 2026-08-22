# 定点档精度复审（2026-08-21，逐档独立度量）

SHA `4c4b4b8c8d9af5282813ee7c36a4bf1c5ca87856`，分支 `feat/attn-conv-fixed-tiers`。
模型 `/tmp/kmr20`（`L98Z` 二进制头部 magic + version=4，即 v4 checkpoint，
`use_subn=0` —— 见 §4.3）。

本文件是纯度量：不改引擎行为，只重跑 `CLAUDE.md` `## Verify` 一节记录的对照，
换成当前 HEAD、当前模型、当前 tier 列表。工作区当时另有一个会话在 `src/ops.c`
留了约 286 行未提交改动（ARM 三值核），本文的两个二进制都不是从那棵脏树编的 ——
用 `git archive HEAD | tar -x` 分别解到 `/tmp/audit-build`（度量用，plain build）
和 `/tmp/audit-f32count`（`-DLZ_F32COUNT=1`，仅用于举证 engagement，见 §2）两个
隔离目录，两边都用 `make native` + `make watcom-cli` 编译，日志干净（只有既存的
`-Wunused-function` 警告）。

## 0. tier 清单核对

`grep` `src/cli_main.c` 的 `--help` 文本和 `src/ops_sched.c` 的 `lz_*_select`，
得到七个独立轴：

| flag | 取值 | 默认 |
|---|---|---|
| `--recur-write` | `float\|fixed` | fixed |
| `--conv-write` | `float\|fixed` | fixed |
| `--attn-write` | `float\|scoring\|wsum\|fixed\|int` | fixed |
| `--epi-write` | `float\|fixed` | fixed |
| `--sig-write` | `float\|fixed` | fixed |
| `--norm-write` | `float\|fixed` | fixed |
| `--scalar-fixed` | `float\|fixed` | fixed |
| `--fixed` | `off\|all`，群控开关 | all |

**`auto` 曾是这七个轴各自的第三个取值，现已删除。** 它从来只是 `fixed` 的
别名——`lz_soft_float_target()` 退役后没有任何轴的 auto 再问硬件，而它作为
一个「有名字的取值」持续产出假注释（`lz_gdn_p2_select` 与 `q8r_have_simd`
两处都写着「AUTO 问硬件」，两处都不成立）。「没人设过这个轴」这个**状态**仍
存在，`--fixed` 的优先级机制需要它，但它不再有 CLI 名字，banner 里显示为
`(default)`。`--pair` 与 `--prefetch` 的 `auto` 保留——那两个真的问硬件。

交付方给的 12 项与这份枚举**完全对上，没有遗漏项**。两处需要单独说明：

- `--recur-write` 的**覆盖面变了**：它现在同时选 GDN/KDA 的 pass 1 与 pass 2。
  此前它只到得了 pass 2，pass 1 的定点没有任何运行期关闭方式——于是当时以
  `--fixed off` 为参照的量都是对着一条仍在跑定点 pass 1 的臂量的。
- **「什么都不传」不是 float 基线，七个轴都不是**（此前只有 `--scalar-fixed`
  一个轴是这样，现在全部统一）。要把它们全钉在 float 上用 `--fixed off`；
  本文全部 baseline 命令都带这个覆盖，若省略，baseline 本身就已经混了各档的
  偏差，任何后续对照都会把这份偏差错记成别的 tier 的贡献。

`--fixed all` 是群控：把 `recur/conv/attn/epi/sig/norm/scalar` 七轴（除已被
命令行上其它 per-tier flag 显式设过的）都设成 "fixed" 这个**词**，但 `attn` 和
`norm` 各自的 "fixed" 和 "int" 是同一个枚举上的不同值——`--fixed all` 只到
"fixed" 一级，不会把 attn/norm 推到 "int"（§3.7/§3.8 的 census 证实了这一点：
`fixed_all` 的 quantize 位点账单与单独 `--attn-write fixed`、不含 `--norm-write
int` 时完全一致）。per-tier flag 无论出现在 `--fixed` 前后都赢（已用
`--fixed all --attn-write float` 和 `--attn-write float --fixed all` 两种顺序
验证，结果字节相同）。

## 1. 度量口径（按 CLAUDE.md 的 Verify 节，不是自定）

- 容差是 Q8 激活噪声下限 **3.9e-3 相对**。
- **`|logits diff|` 不是任何档的判据**——它饱和；本次在 152×32768 个 (position,
  vocab) 格点上取最大值，`norm-write fixed` 测得 1.55、`fixed_all` 测得 1.66,
  比 CLAUDE.md 记录的「最小扰动 1.89e-2、更大扰动 1.30e-1」大一个数量级以上，
  但这正是「饱和到没有信息量」的意思——测量口径不同（本文取 152 个位置的
  worst-case，CLAUDE.md 原数字看起来是单一扰动的量级），两者都只是佐证
  「这个数不能用」，不是相互矛盾的结论。判据是**配对 NLL、argmax 一致率、
  top-5 一致率**（`--dump-nll`/`--dump-top5`）。
- **跨编译器逐位一致仍然是硬约束，不是容差**：同一 tier 设置下 gcc-fixed 与
  Watcom-fixed 必须字节相同。

## 2. Engagement 举证方法

没有改引擎源码来插桩。改用引擎自带、纯编译期开关的 `-DLZ_F32COUNT=1`
（`src/ops.h` 的 `LZ_FC_*` 站点普查，`ops.c` 已经把存储和累加都写在
`#if LZ_F32COUNT` 里，一个 `-D` 就能打开，不需要碰任何 `.c` 文件）。用它编出
`/tmp/audit-f32count/llama98.exe`，对每个 tier 设置单独跑 8 个 token，读
`f32 operations per token` 表里对应站点的 mul/add/div/cvt/cmp 分解——float 和
fixed/int 走的是不同代码，账单构成必然不同；如果某 tier 的账单和 baseline
一字不差，那就是没生效，而不是「精度好到看不出来」。

（中途有一次操作事故：第一次插桩误写到了真实的 `E:\llama98-zh\src\forward.c`
而不是 `/tmp/audit-build` 的隔离副本，`git diff` 发现后立即 `git checkout --`
撤销，确认 `git status` 干净后改用上面这条零源码改动的路线重做。）

### 2.1 十二项单独 engagement（8 token, baseline 全 float 显式）

| tier | 被改变的站点账单（baseline → tier，mul/add/div/cvt） | 判定 |
|---|---|---|
| `--recur-write fixed` | recurrence write: 2,777,088 → **110,592**/tok（-96%，int64 写回跳过 float 复量化） | 生效 |
| `--conv-write fixed` | conv1d 构成从 mul73728/add73728/cvt0 变成 mul92160/add18432/**cvt36864**；连带 exp 750568→511578、sigmoid 197280→123552（conv 门控的 exp/sigmoid 账单一并移出 float） | 生效 |
| `--attn-write scoring` | attention scoring: 28296 → **1512**（-95%） | 生效 |
| `--attn-write wsum` | attention wsum: 36864 → **7136**（-81%） | 生效 |
| `--attn-write fixed` | scoring 1512 + wsum 7136（等于上两行的和） | 生效 |
| `--attn-write int` | scoring/wsum 与 fixed 完全相同，但 **quantize 站点** 从 fixed 的 101112 变成 **103352**，新增一个 fixed 没有的 `add=2048` 类——证明 int 走的是自己独立的 `lz_quantize_q8_int64` 分支，不是复用 fixed 分支 | 生效，且与 fixed 可区分 |
| `--epi-write fixed` | dequant epilogue: 1,020,192 → **119,616**（-88%） | 生效 |
| `--sig-write fixed` | exp: 750568 → **82682**（-89%）；sigmoid 构成从 mul43008/add102848/div51424 塌缩成 mul197280（div/add 归零，全部折进 Q15 表查） | 生效 |
| `--norm-write fixed` | norms 构成从 mul101376/add41285/div133 变成 mul101701/add53573/div133/**cvt12613**；连带 exp 750568→671322、sigmoid 197280→172704（gated-norm 的定点输出环内建 sigmoid_q15 查表，绕开了独立的 sigmoid 站点） | 生效 |
| `--norm-write int` | 相对 `--norm-write fixed`：norms 168020→**155732**（cvt 12613→6469），quantize 96888→**103800**（新增 `add=6144`）——两个站点都和 fixed 不同，证明 int 是 fixed 之外单独的分支 | 生效，且与 fixed 可区分 |
| `--scalar-fixed fixed` | exp: 750568 → **404096**（-46%）；rsqrt 构成从 mul2275/add650 变成 mul777/**cvt325**（牛顿迭代换成查表+精修） | 生效 |

### 2.2 `--fixed all` 与「全部拉满」

```
--fixed all
```
把七轴全部推到 "fixed" 一级（不含任何 "int"）。总账单 5,200,583 → **854,260**/tok
（-83.6%），quantize=101112，与单独 `--attn-write fixed` 的 101112 完全相同、
`--norm-write fixed` 不额外增加 quantize —— 印证了 §0 的判定：群控只到 fixed，
不到 int。

```
--recur-write fixed --conv-write fixed --attn-write int \
--epi-write fixed --sig-write fixed --norm-write int --scalar-fixed fixed
```
（下文表格记作 `all_max`，把 attn/norm 也推到各自的顶格 "int"）总账单再降到
**840,884**/tok。quantize=110264，恰好等于 `101112`（`fixed_all` 的 quantize）
`+ 2240`（attn int 相对 attn fixed 的增量，§2.1）`+ 6912`（norm int 相对 norm
fixed 的增量，§2.1）= 110264，分毫不差——两个 "int" 变体在同时打开时各自独立
叠加生效，不是互相覆盖或被静默丢弃。

### 2.3 `--norm-write int` 与 SubLN、与 v4 checkpoint 的关系（交付方假设的修正）

交付方原始假设是「SubLN 范式需要 v7+ checkpoint 而 kmr20 是 v4」，猜测
`--norm-write int` 可能因此不生效。`model.bin` 头部实测 magic=`L98Z`
version=**4**（`xxd -l 16`），按 `model.c:1423-1435` 的解析规则 `ver < 7` 时
`use_subn` 保持初值 0，这一步是确定性的二进制解析，不需要再跑一次来确认。

但 `--norm-write int` 在 `forward.c` 里一共有 8 个调用点，其中 4 个的确挂在
`if (c->use_subn)` 之下（KDA 的 q/k/v/g 输入范数、dense FFN 的 gate/up/down
范数）——这些在 kmr20 上确实不跑。**另外 2 个不挂 `use_subn`**：GDN/KDA 线性
注意力输出端「plain-weight gated RMSNorm」（`forward.c:1947` 附近注释明说
「not the (1+w) kind used by the outer layers」，每个线性注意力层都跑，与
SubLN 无关）。§2.1 的 census 证明就是从这条路径生效的——kmr20 上
`--norm-write int` **照样生效**，只是走了一条交付方没预料到的调用点，不是
因为 SubLN。原假设的方向性判断（"可能不生效"）不成立，具体机制判断（"经由
SubLN"）也不成立，但两者合在一起的净结论碰巧对了一半——生效了。这一条记录
下来是为了不让下一个人重复"因为不是 SubLN 所以想当然不生效"这一步。

## 3. 精度结果

两个 token 集合已存在于 `.prof/`：`tz.bin`（153 token，中文，152 个可评位置）、
`te.bin`（211 token，英文，210 个可评位置），都是 `--dump-prompt-tokens` 格式
（CRLF 分隔的十进制 id），可以直接喂给 `--tokens-file`。

**方法论修正（度量过程中发现，写在这里避免下一个人重复踩）**：`--dump-nll`
自己在 `cli_main.c` 里用 C 标准库的 `exp()`/`log()`（double）在测量代码里现算
一遍 softmax，MinGW 的 libm 和 Open Watcom 的 libm 在最后几位不保证相同——
这会让**测量工具自己**报出 gcc/Watcom 字节不同，即使被测的引擎输出（原始
logits）逐位相同。baseline（全 float）的 `--dump-nll` 输出跨编译器确实不是
字节相同，一度看起来像是「float 基线都跨编译器不一致」这种大问题；用
`--dump-all-logits` 直接比较原始 logits 字节后确认那 152/210 个位置全部
相同，问题出在测量脚本的 `exp/log`，不出在引擎。**因此下表的跨编译器一致性
列一律以 `--dump-all-logits` 的原始字节比较为准，不用 `--dump-nll` 的字节
比较**；NLL/argmax/top-5 数值本身则用 Python 对同一份 `--dump-all-logits`
数据自己算 softmax（跳过两边 libm 的差异），保证「NLL 差多少」这个数字本身
不掺入这个已知的测量噪声。

### 3.1 token set = tz（中文，152 个可评位置）

| config | mean NLL | ΔNLL vs baseline | 配对 t | argmax 一致率 | top-5 一致率 | gcc≡Watcom（全部位置原始 logits 逐位比较）|
|---|--:|--:|--:|--:|--:|---|
| baseline（七轴全 float） | 3.9577 | — | — | 100.00% | 100.00% | 相同 |
| `--recur-write fixed` | 3.9552 | -0.00255 | -0.51 | 95.39% | 84.21% | 相同 |
| `--conv-write fixed` | 3.9507 | -0.00703 | -1.43 | 98.03% | 86.84% | 相同 |
| `--attn-write scoring` | 3.9510 | -0.00671 | -2.00 | 96.71% | 88.16% | 相同 |
| `--attn-write wsum` | 3.9567 | -0.00104 | -0.37 | 96.71% | 90.79% | 相同 |
| `--attn-write fixed` | 3.9514 | -0.00631 | -2.27 | 96.05% | 90.13% | 相同 |
| `--attn-write int` | 3.9513 | -0.00636 | -1.64 | 96.05% | 88.16% | 相同 |
| `--epi-write fixed` | 3.9531 | -0.00459 | -1.02 | 96.05% | 86.18% | 相同 |
| `--sig-write fixed` | 3.9542 | -0.00355 | -0.78 | 98.03% | 92.76% | 相同 |
| `--norm-write fixed` | 3.9455 | -0.01217 | -2.78 | 97.37% | 89.47% | 相同 |
| `--norm-write int` | 3.9455 | -0.01217 | -2.78 | 97.37% | 89.47% | **不同**——见 §4.1 |
| `--scalar-fixed fixed` | 3.9565 | -0.00118 | -0.34 | 96.05% | 92.76% | 相同 |
| `--fixed all` | 3.9489 | -0.00883 | -1.50 | 98.03% | 90.13% | 相同 |
| all_max（七轴顶格，含两个 int） | 3.9506 | -0.00713 | -1.28 | 98.03% | 85.53% | 相同 |

### 3.2 token set = te（英文，210 个可评位置）

| config | mean NLL | ΔNLL vs baseline | 配对 t | argmax 一致率 | top-5 一致率 | gcc≡Watcom（原始 logits）|
|---|--:|--:|--:|--:|--:|---|
| baseline | 5.3704 | — | — | 100.00% | 100.00% | 相同 |
| `--recur-write fixed` | 5.3686 | -0.00174 | -0.53 | 99.05% | 87.14% | 相同 |
| `--conv-write fixed` | 5.3788 | +0.00841 | 1.39 | 98.57% | 81.90% | 相同 |
| `--attn-write scoring` | 5.3710 | +0.00060 | 0.15 | 99.52% | 88.57% | 相同 |
| `--attn-write wsum` | 5.3747 | +0.00429 | 1.10 | 99.05% | 87.62% | 相同 |
| `--attn-write fixed` | 5.3639 | -0.00652 | -1.42 | 99.05% | 84.29% | 相同 |
| `--attn-write int` | 5.3700 | -0.00038 | -0.12 | **100.00%** | 88.10% | 相同 |
| `--epi-write fixed` | 5.3731 | +0.00268 | 0.44 | 98.57% | 84.29% | 相同 |
| `--sig-write fixed` | 5.3729 | +0.00248 | 0.57 | 99.52% | 82.86% | 相同 |
| `--norm-write fixed` | 5.3701 | -0.00028 | -0.05 | 98.10% | 83.33% | 相同 |
| `--norm-write int` | 5.3701 | -0.00028 | -0.05 | 98.10% | 83.33% | 相同（在 te 上未复现 §4.1 的分歧）|
| `--scalar-fixed fixed` | 5.3712 | +0.00079 | 0.21 | 99.52% | 87.62% | 相同 |
| `--fixed all` | 5.3659 | -0.00445 | -0.77 | 99.05% | 82.38% | 相同 |
| all_max | 5.3720 | +0.00163 | 0.33 | 99.05% | 83.81% | 相同 |

**读法**：argmax 一致率全部落在 95%–100%，top-5 一致率落在 82%–93%，两个语言
方向一致（没有哪个 tier 在中文上崩、在英文上正常，或反过来）；配对 t 的量级
（|t| < 3）说明 NLL 的漂移相对其自身抖动而言不算离谱地系统性偏移。这些数字
不构成「argmax/top-5 崩溃」，即容差判据（argmax/top-5 一致率）没有发现任何
一个 tier 在中/英两个方向上都表现异常——**除了 §4.1 那一条跨编译器分歧**。

## 4. 违反硬约束的一条：`--norm-write int` × tz

`--norm-write int` 单独打开（其余六轴显式 float）时，gcc build 与 Watcom build
在 tz 这 152 个位置上，**从第 146 个位置起 6/152 个位置的原始 logits 整条
32768 维向量全部不同**（不是几个 LSB，`max|diff|` 到 0.13–0.28），第 147
位甚至翻了一个 top-5 名次（gcc `[...,11821]` vs Watcom `[...,11526]`，rank
5 不同）。

复核结果：

- 在 **te**（210 个位置）上，同一 `--norm-write int` 配置**逐位完全相同**，
  没有任何分歧——不是这个 tier 普遍性地跨编译器不一致，是 tz 这份具体输入
  触发的。
- `--norm-write fixed`（不含 int）在 tz 上 152 个位置全部逐位相同——问题
  专属于 int 这一级，不是 fixed 的sum-of-squares 计算本身。
- `all_max`（`--norm-write int` 与其它六轴全部打开同跑）在 tz 上 152 个位置
  也全部逐位相同——单独打开会分歧、和别的 fixed 轴一起打开反而不分歧，说明
  触发条件与其余轴处于 float 状态时残留的跨编译器噪声（x87 vs SSE2 的 float
  基线本身在 kmr20 + tz 这条组合上并非处处逐位一致，只是恰好在其它单 tier
  测试里没有踩中分支翻转的边界——参见下面的机制猜测）有关，不是 int 路径
  本身在整数域不可移植。
- 用 `--norm-write int`（省略其余六个 float 覆盖，纯默认 auto）复现，同样在
  第 146 个位置起分歧——不是我显式 float 组合本身引入的假象。

**机制（推测，未验证，标注为推测）**：`lz_rmsnorm_int`/`lz_rmsnorm_gated_int`
的量化尺度 `ex = 14 - expof(q8_amax(...))` 是对一个 float 输入取 amax 后的
一个**阶梯函数**——上游（仍是 float 的 attention/conv/sigmoid 等路径）里
1-ULP 级别的 x87-vs-SSE2 跨编译器噪声，一旦让某个 amax 落在两个 build 里
刚好卡在阶梯的两侧，`ex` 就会相差 1（尺度整体翻倍/减半），继而让整条 Q15
int64 链条的结果出现远大于 1 ULP 的偏差，并通过 KDA/GDN 的循环状态向后
传播、放大成后续所有位置全向量分歧——这与本仓库既有的「量化混沌放大」记录
（`int64-to-float-fusion-diverges`：跨编译器 1 ULP 被量化混沌放大成整层
分歧）是同一类机制，只是触发的具体站点和阈值位置不同。这个解释没有另开
探针验证，写在这里是为了给下一个要修它的人一个起点，不是结论。

这一条**直接违反 CLAUDE.md 记录的硬约束**（"gcc-fixed vs Watcom-fixed must
be byte-identical" / "Cross-compiler bit-identity still holds inside every
tier"）——需要用户判断是否更新那份记录、要不要立即修，还是先记录着。

## 5. 与 CLAUDE.md 记录数字的对照

| CLAUDE.md 记录 | 当时条件 | 本次复核 |
|---|---|---|
| "every tier alone and all four together...identical bytes in all cases" | kunmoe d_ref, ntok 8 / ctx 1440 | 在 kmr20 + tz/te（152/210 个位置）上，13/14 个 config 组合确认相同；`--norm-write int` 在 tz 上从位置 146 起不同（§4）。**把 `--norm-write int` 本身单独拿到 ntok=8（tz 的前 8 个 token）下重跑，逐位相同**——分歧只在更长的序列上出现，ntok=8 这种短探针天生看不见它。 |
| "`\|logits diff\|`...1.89e-2...1.30e-1" | 同一模型上的最小/较大扰动 | 本次在 152 个位置上取 worst-case 得到 1.55（norm-fixed）/1.66（fixed_all），量级更大，口径不同（全序列 worst-case vs 单一扰动），但结论一致：这个数没有信息量，不能当判据。 |
| "fixed 2.7x–15x less f32 work" | 未指名具体 tier | 本文 census：`fixed_all` 总账单降 83.6%，`all_max` 降 83.8%，与该区间量级吻合；单 tier 里 epi(-88%)、recur(-96%)、attn-scoring(-95%) 都在区间上端，符合"哪个 tier 更省 f32 work 因 tier 而异"的既有描述。 |

## 6. 本次未覆盖的范围

- **KV ring**：token 集合最长 211，kmr20 context 32768，远没有触发 KV 环形
  缓冲区回绕；CLAUDE.md 自己也写了"KV-precision claim 需要 ~20k 个预测位置"，
  这条没有重新验证，§4 的分歧和 KV ring 无关（远没到那个规模）。
- 没有测 `--kernel ref|mmx|sse2` 这条独立轴（matmul 核选择），题目范围是七个
  write-tier 轴，不含它。
- `--attn-write int` 的结构性前置条件（`gso==32`、`kv_rot_v==0`、`!v4`、
  `!vf`）在 kmr20 上成立，是从 census 的账单变化**反推**出"一定成立"（因为
  不成立的话 census 不会变），没有另外去读 `o_proj` 的量化 group size 数值
  做静态确认——这是举证链条里唯一一处"证明生效"但"没有验证具体是哪个前置
  条件在起作用"的地方。

## 7. 结论摘要

- 交付方列出的 12 个 tier 值 + `--fixed all`，加上 `ops_sched.c` 交叉核对，
  **没有遗漏项**。
- **12 个 tier 值全部在 kmr20 上生效**，逐一有 census 账单变化作证据，且
  `attn-int`/`norm-int` 都证明走的是与其"fixed"版本不同的独立分支，不是
  静默复用。
- **argmax/top-5 一致率**：95%–100% / 82%–93%，中英文方向一致，没有发现
  容差判据下的异常 tier。
- **跨编译器逐位一致**：13/14 组合确认成立；`--norm-write int` 在 tz 上从
  位置 146 起违反，te 上不违反，机制推测见 §4，需要用户决定下一步。
- CLAUDE.md 记录的旧数字（ntok=8/kunmoe d_ref）在其原始条件下复测仍然成立
  ——分歧是本次用更长序列、不同模型才发现的，不是旧记录本身测错了。
