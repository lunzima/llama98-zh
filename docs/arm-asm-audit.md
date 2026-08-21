# ARM 手写汇编格核对（matmul 之外的算子）

铁律八的 ARM 表是两格：**标量 C**（与 x86 共用同一份）与**手写汇编**（GNU as /
`__asm__`）。缺格要带原因，而唯一可以不写的理由是「指令集上根本没有独立内容」，
举证责任在不写的一方。这份文档是那份举证：把 matmul 以外的每个算子逐个打开，
列出它在昆昆CE 上实际执行的操作序列、逐条核对 ARMv5TE 能作用在这个序列上的指令、
给出判定。

**收益大小不进入判定。** 「收益太小」「本机测不出差别」「值不值得」是铁律八
作废掉的框架，本文一次都不用它。判定只有两种：

- `有独立内容` —— 存在 ARMv5TE 指令，能替代或减少该序列中的步骤，而
  `arm-linux-gnueabi-gcc -O2 -march=armv5te` 从现有 C 源码出发发不出它。
- `IMPOSSIBLE` —— 逐条核对完可用指令，没有一条能替代或减少任何一步。

排序是另一件事，见 §5：它按「这个算子在 CE 前向里每 token 跑多少次」排，
不按收益猜测排。

matmul 本体（`lz_matmul*`、`lz_dot32_*`、`dot32_x16`、`epi_q8` / `epi_q41` /
`epi_fixed` / `epi_pack_act` / `lz_epi_prep`）不在这里，见 §6。

---

## 0. 取证方法

三样东西，都可重跑：

1. **实际发出的指令**。用 CE 目标的真实标志把每个 TU 编成汇编读：

   ```sh
   arm-linux-gnueabi-gcc-14 -O2 -std=c99 -march=armv5te -mtune=arm926ej-s \
       -mfloat-abi=soft -DLZ_NO_FPU=1 -ffp-contract=off -Isrc -S src/<tu>.c
   ```

   （与 `build/arm/build_engine.sh` 同一组标志。工具链在 WSL 侧，名字是
   `arm-linux-gnueabi-gcc-14`，Git Bash 的 PATH 上没有。）本文引用的每个
   「gcc 发出 N 条指令」都是从这份 `.s` 数出来的。

2. **软浮点例程的体量**。CE 无 FPU，每个浮点运算都是一次 `bl __aeabi_*`。
   这些例程的静态指令数（`arm-linux-gnueabi-objdump -d $(arm-linux-gnueabi-gcc-14
   -print-libgcc-file-name)`）：

   | 例程 | 整体 | 正常操作数快路径 |
   |---|---|---|
   | `__aeabi_fmul` | 102 | 28 |
   | `__addsf3`（`fadd`/`fsub`） | 100 | ~40 |
   | `__aeabi_fdiv` | 88 | — |
   | `__aeabi_f2iz` | 23 | — |
   | `__aeabi_i2f` | 8，尾接 `__aeabi_l2f`(31) | ~20 |
   | `__aeabi_fcmpXX` | 5，各自 `bl __cmpsf2`(25) | ~25 |
   | `__aeabi_dmul` / `__adddf3` | 155 / 171 | — |

   静态指令数正是铁律三说的「能跨机器迁移的量」；QEMU 的墙钟不能，所以本文
   一个周期数都不给。

3. **ARMv5TE 指令全集**，由汇编器本身裁定而不是靠记忆。下面这些
   `arm-linux-gnueabi-as -march=armv5te` 全部接受：

   `SMLA<x><y>` `SMUL<x><y>`（16×16(+32)，ARM9E 上近单周期）、
   `SMLAW<y>` `SMULW<y>`（32×16→取高 32，即乘完 `>>16`）、
   `SMLAL<x><y>`（16×16+64）、`SMLAL` `SMULL` `UMULL` `UMLAL`（32×32→64）、
   `MUL` `MLA`、`QADD` `QSUB` `QDADD` `QDSUB`（int32 边界饱和）、`CLZ`、
   `LDRD` `STRD`、`LDM` `STM`（含 `!` 回写）、`PLD`、桶形移位器
   （`add r0,r1,r2,lsl #3` 一条）、全条件执行（`movgt` / `addne` / `rsbmi` …）。

   下面这些同一命令**拒绝**，所以本文任何一处都不许出现：
   `SSAT` `USAT`（16 位饱和是 ARMv6 才有）、`SADD16`/`UQADD8` 等 v6 SIMD、
   `UXTB` `SXTB` `SXTH` `REV` `PKHBT` `SMUAD`（v6）、`MOVW`（v6T2）、
   `SDIV`（v7）。符号扩展只能靠两条移位（`lsl #24` + `asr #24`），
   或者让 `LDRSB`/`LDRSH` 在载入时顺带做掉——后者与「打包载入」互斥，
   这是 §2 M3 在 int8 数据上要付的代价，写内核时要算进去。

   **没有除法指令**，浮点、整数都没有。

4. **执行指令计数**（不是静态计数），用在静态数不够用的地方。
   本机的 `qemu-arm-static`（8.2.2，Debian 打包）**没有编 TCG 插件**
   （`-plugin` 报 `unknown option`），所以走的是退路：

   ```sh
   qemu-arm-static -one-insn-per-tb -d exec,nochain ./drv N 2>&1 >/dev/null | wc -l
   ```

   一条指令一个翻译块、一个翻译块一行日志，于是行数就是执行指令数；
   `nochain` 防止链接过的块不打日志，程序自己的 stdout 丢弃。
   三条判据一个都不能省：**零对照**（`N=0` 的两条臂必须逐条相同，
   把 QEMU 启动与 libc 初始化减掉）、**线性检查**（`X(512)-X(256)`
   与 `X(256)-X(0)` 必须一致，否则「每次多少条」这个说法本身不成立）、
   以及**真实输入分布**（合成值会系统性偏向某条分支）。
   探针在 `.prof/arm_sp_*` 与 `.prof/arm_bias_*`（`.prof/` 按决定 gitignore）。
   已经用它量过六个叶子，结果见 §5 的「`.s` 估计能不能信」一节 ——
   **结论是不能给一个修正因子**，所以 §5 的表用一列标出每一行的数
   是实测、是从 `.s` 逐条数的循环体、还是估计。

一条贯穿全文的事实：**ARMv5TE 上 `QADD`/`QSUB` 只在 int32 边界饱和**，
本引擎的钳位全是 ±127 / ±32767，所以饱和指令族对这些钳位一律用不上；
钳位只能是 `cmp` + `movgt`/`movlt`，而这个形状 gcc 已经在发。凡是下文说
「钳位没有独立内容」，指的都是这一条。

---

## 1. CE 上实际会走到的是哪一档

判定必须按 ARM 上真正执行的那一档，不是按 x86 上执行的那一档。
`build/arm/build_engine.sh` 传 `-DLZ_NO_FPU=1`，`ops_sched.c:127` 的
`lz_soft_float_target()` 因此恒为 1，于是所有 `auto` 档解析为定点：

| 旋钮 | ARM(auto) 的解析结果 | 走到的代码 |
|---|---|---|
| `--recur-write`（`lz_gdn_p2_mode`） | fixed | `gdn_p2_fixed_rows` / `gdn_p2_group_fixed` |
| `--conv-write`（`lz_conv_mode`） | fixed | `lz_causal_conv1d_step_fixed` |
| `--attn-write`（`lz_attn_mode`） | fixed（3 = scoring+wsum） | `lz_attn_score_q8` / `lz_attn_wsum_q8` |
| `--sig-write`（`lz_sig_mode`） | fixed | `sigmoid_q15` |
| `--norm-write`（`lz_norm_mode`） | fixed | `norm_ss_fixed` + **float 输出环** |
| `--epi-write`（`lz_epi_mode`） | fixed | `epi_fixed`（matmul 侧） |
| `--scalar-fixed`（`lz_scalar_mode`） | **fixed（在每台机器上都是）** | `lz_exp_fixed` / `lz_rsqrt_fixed` |
| `--kernel`（`lz_kernel_select`） | `arm-asm` | 目前只接 T2 matmul 行内核 |
| GDN pass 1（`lz_gdn_mode`） | fixed（编译期 `LZ_GDN_FIXED=1`） | `gdn_sum_gg` 标量分支 |

三个必须写下来的细节，搞错任何一个整节结论都会反：

- **`--norm-write auto` 到不了 `int`。** `lz_norm_mode()` 只回答「是不是定点」，
  而 `lz_norm_int()` 要求 `g_norm == LZ_NORM_INT`，`auto`（-1）永远不是。
  于是 `lz_rmsnorm_int` / `lz_rmsnorm_gated_int` / `lz_rmsnorm_out_fixed_int` /
  `lz_quantize_q8_tok_int` / `lz_quantize_q8_int` 在 CE 默认配方里**一次都不跑**；
  定点档换掉的只有平方和（`norm_ss_fixed`），**输出环仍是浮点**。
- **`--attn-write auto` 到不了 `int`** 同理（`lz_attn_int()` 要 `g_attn == 4`），
  所以 `lz_attn_wsum_q8_int` 与 `lz_quantize_q8_int64` 也不跑。
- **`--scalar-fixed` 的 auto 是 fixed，不问 CPU**（`lz_scalar_mode()` 是
  `(g_scalar < 0) ? 1 : g_scalar`，`cli_main.c:253` 的帮助文本写的也是
  `default fixed`）。`ops.h` 该函数上方注释说它「问和别的档一样的 CPUID 问题」，
  与代码不符；以代码与 CLI 帮助为准。

编译期开关：`LZ_GDN_STATE_2PLANE` 默认 **1**（递推状态两平面，pass 1 与 pass 2
的每元素工作量翻倍），`LZ_KV_2PLANE` 默认 **0**，`--kv-rot` 默认 **off**。

CE 出货几何（`E:/LLM/llama98-zh/docs/kunkun-ce-params.md` §1 定案 +
`docs/kunkun-ce-bitnet-engine.md` Task 4 的 SubLN）：

```
hidden 512  layers 6 = LLALLA（4 KDA + 2 全注意力，interval 3）  inter 2048  vocab 6144
KDA        16 头 × (kd=vd=32) → kdim=vdim=512，gate_rank=32，conv k=4，conv_dim=1536
全注意力    8 q 头 / 2 kv 头 / head_dim 64 → qd=512, kvd=128, rotary_dim=64
SubLN 开，hadamard_o=512，hadamard_down=512（2048 % 512 == 0 → 每层 4 块）
first_k_dense_replace 6 = n_layers → 六层全是 dense SwiGLU，无 MoE
```

> **深度换宽度在 FFN 上是守恒的，在别处不是**，这是 §5 里几处名次变动的
> 唯一来源。`n_layers × intermediate_size = 6 × 2048 = 12,288` —— 凡是只按
> 「层数 × inter」走的量（SwiGLU 的 `lz_silu` 次数、`hadamard_down` 的
> FWHT 蝶形数、down_proj 侧的 norm 与量化元素数）**一个数都不变**；
> 而按「层数」单独走的量（每层固定开销的 norm、量化、以及全部 KDA 递推）
> 按 4/6 或 6/8 缩。两个方向不同，所以不能等比缩放。
>
> 盘上的 `E:/LLM/models/kunkun-ce` 与定案**逐项一致**，本文档的计数
> 因此全部建立在已定的值上，没有一处建立在假设上：
> `num_hidden_layers:6`、`intermediate_size:2048`、
> `layer_types:[L,L,A,L,L,A]`、`first_k_dense_replace:6`、
> `use_subn:true`、`kda_gate_lower_bound:-5.0`。
> 后两个字段曾经是本表最大的两处不确定性，现在都不是了；
> `kda_gate_lower_bound` 定案的经过写在 §3 C6。
>
> `.prof/budget/ce_sizing.py` 的数**不能直接引用**：它少算 28,672
> （正好是 SubLN 每投影 norm，mlp `6×3072` + KDA `4×2560`），
> 且它第三个位置参数是**注意力 `head_dim`(64)** 而不是 KDA 头宽(32)，
> 传错不报错。本文档的所有计数都不经它，直接从几何与 `forward.c`
> 的循环结构数出来。

---

## 2. 五个反复出现的机制

下面 30 多条判定里，绝大多数落在这五个机制之一。写在这里一次，条目里只引用。

**M1 —— 乘 2 的幂是指数域加法。** 本引擎里大量乘法的乘数是精确的 2 的幂
（`pow2f(e)`、`LZ_SIG_STEP=16`、`1/32768`、`lz_i32f` 的 4096、`p2_group_norm`
的 `p`/`sacc`、`norm_ss_fixed` 的 `sc`）。C 层面写出来的是 `float * float`，
gcc 只能发 `bl __aeabi_fmul`（快路径 28 条，含解包两个操作数、`umull`、
规格化、舍入）。汇编格知道乘数是 2^e，于是只剩「指数字段加 e + 溢出/下溢
两条条件分支」约 5 条指令，且**逐位一致是可证的**（乘 2 的幂在 IEEE 下精确）。
这是 CE 上最大的单一杠杆，而它够不着的原因恰恰是 C 那一格与 x86 共用
（x86 上 `mulss` 一条，没有动机把它写成位操作）。

**M2 —— 浮点比较是整数比较。** 每个 `x < 0.0f` / `a > amax` 在 ARM 上是
`bl __aeabi_fcmpXX` → `bl __cmpsf2`，约 30 条。非 NaN 浮点的序与其位模式的
符号-幅值序一致，2–4 条整数指令即可。引擎自己在 `q8_amax` 里已经这么做了
（`bic` 清符号位后按 `uint32` 比），所以这不是新发明，是把已有做法推广到
其余比较点。

**M3 —— 打包载入。** `int8_t*` / `int16_t*` / `float*` 数组上，gcc 只能按
元素类型发 `ldrsb` / `ldrh` / `ldr`，因为 C 类型只承诺 1/2/4 字节对齐。
本引擎的行指针实际都来自 `calloc` 或 `static`，4 字节对齐；把这个契约写进
汇编格的前提条件（`lz_dot32_t2_arm_asm` 已经是这么做的，见 `ops_t2_arm.h`
里那条 `EXTRA PRECONDITION`），一条 `LDM`/`LDR` 就能顶 4 条 `ldrsb`。

**M4 —— `SMLA<x><y>` 的半字选择让一次载入喂两次乘加。** C 只能说
「int16 × int16」，gcc 因此只发 `smlabb`，且每个元素单独载入。用齐
BB/BT/TB/TT，一个载入的 32 位字喂两次 MAC。这正是 `ops_t2_arm.c` 里已经
写下的三条理由中的第一条。

**M5 —— 库例程是通用的，内核不是。** `__aeabi_fmul` 每次调用都要重新解包
两个操作数、处理 NaN/Inf/subnormal/零。内核知道：一个操作数循环不变
（解包一次即可）、量程有界（`LZ_Q8_MIN_SCALE` 已经在全引擎范围内把
subnormal 消掉了，`ops.h` 该宏上方注释是权威文本）。于是内联的软浮点乘法
可以做到 10–12 条而不是 28 条 + 调用开销。**代价是必须自证舍入与
`__aeabi_fmul` 逐位相同**——这是铁律二对这一格的要求，不是可选项。

---

## 3. 逐算子核对

### A. 量化 / 舍入 / 位操作叶子

#### A1 `q8_round`（`ops_quant.c`）

1. **操作序列**：`qv + 12582912.0f`（1.5·2^23 魔数加）→ 取结果低 23 位 →
   减 0x400000。ARM 实际发出：`push {r4,lr}` / `ldr r1,=魔数` /
   `bl __aeabi_fadd` / `bic` / `bic` / `sub` / `pop`。**每次调用是两层调用**
   （`bl q8_round` 再 `bl __addsf3`），整条路径约 48 条指令。它是**非 static
   的跨 TU 函数**，没有 LTO，所以在每个调用点都是真调用，不内联。
2. **可用指令核对**：魔数加这个技巧存在的理由是 x87 的控制字重写
   （`ops_quant.c` 该函数注释里的 89.8 ns vs 4.0 ns）。**ARMv5TE 没有 FPU，
   没有控制字，也没有 float→int 指令**——这个技巧在这台机器上不但没收益，
   还把一个纯位操作换成了一次完整的软浮点加法。可用指令：`CLZ`（把
   float 的指数取出来算出需要的移位量）、桶形移位器（对阶）、条件执行
   （round-half-to-even 的进位判定）、`QADD`（不适用，饱和点不对）。
   直接在位域上做「按 24 位尾数舍入到最近偶数再取整数部分」是 10–14 条
   整数指令，无调用、无分支。属 M1/M5。
3. **判定**：**有独立内容（该写）**。写法：把 float 位模式拆成
   `e = (u>>23)&0xFF`、`m = (u&0x7FFFFF)|0x800000`，按 `150-e` 做带
   round-half-to-even 的右移（用移出位与粘位判定），符号位取反补；
   与现行魔数加逐位对拍的定义域是 `|qv| < 2^22`（该函数注释已声明的契约）。

#### A2 `q8_amax`（`ops_quant.c`）

1. **操作序列**：ARM 上 `lz_amax_pick` 返回 NULL（无 SIMD 表项），走标量尾环：
   每元素 `ldr` / `bic r3,r3,#0x80000000` / `cmp` / `movcc`，4 条。
2. **可用指令核对**：`LDM`（M3）一条顶 4 条 `ldr`——数组是 `const float*`，
   4 字节对齐由调用方保证，gcc 发不出。桶形移位器：`bic` 已是单条，
   没有更短的清符号位方式。`CLZ`：最大值扫描不需要。`QADD`：不适用。
   `SMLA*`：无乘法。所以独立内容恰好只有 `LDM`：4 元素从 16 条降到 13 条，
   且 `movcc` 链可以做成两路累加最后合并（比较是精确的，顺序无关，
   与 `ops_quant.c` 该函数注释里「max 是精确且可结合，所以这是少数
   允许向量化的归约」同一条理由）。
3. **判定**：**有独立内容（该写）**。写法：`LDM {r4-r7}` + 四路
   `bic`/`cmp`/`movcc`，尾数标量。

#### A3 `q8_group_scale`（`ops_quant.c`）

1. **操作序列**：`q8_amax` → `amax > 0.0f`（fcmp）→ `amax * LZ_Q8_INV127`
   （fmul）→ `127.0f / amax`（fdiv，88 条）→ `inv <= FLT_MAX`（fcmp）→
   `sg < LZ_Q8_MIN_SCALE_F`（fcmp）。每组一次，不是每元素。
2. **可用指令核对**：三次 fcmp 走 M2（各 ~30 条 → 各 3 条）。fdiv 是真除法，
   ARMv5TE 无除法指令，只能用 `CLZ` + Newton-Raphson 的整数迭代——**但那会
   改值**，与 `__aeabi_fdiv` 逐位一致要额外一步修正（可做，代价接近
   直接软除）。这一格的独立内容因此局限在三次比较与那一次
   `amax * (1/127)`（不是 2 的幂，M1 用不上，只剩 M5 的解包复用）。
3. **判定**：**有独立内容（该写）**，但内容只有比较那一部分；除法这一步
   **没有**独立内容（无除法指令，且逐位一致把 Newton 路堵死了）。

#### A4 `pow2f`（`ops_quant.c`）

1. **操作序列**：`if (e < -126) return 0` / `if (e > 127) e = 127` /
   `((e+127) << 23)`。gcc 发出的全部是 7 条：`cmn r0,#126` / `blt` /
   `cmp r0,#127` / `movge r0,#127` / `add r0,r0,#127` / `lsl r0,r0,#23` / `bx lr`。
2. **可用指令核对**，逐条：`SMLA<x><y>`/`SMULW<y>`/`SMLAL*`/`MUL`/`MLA` ——
   这里没有乘法。`QADD`/`QSUB`/`QDADD` —— 饱和点是 int32 边界，而这里要在
   127 处饱和，用不上；何况 `movge` 已经是一条。`CLZ` —— 没有要数的前导零。
   `LDRD`/`LDM`/`STM`/`PLD` —— 不访存。桶形移位器 —— `lsl #23` 已经用了，
   且不能与 `add` 合并成一条（`add r0,r0,#127` 的移位槽是给第二个源操作数的，
   而这里要移的是结果）。条件执行 —— `movge` 已经用了；把 `blt` 也变成
   条件执行（`movlt r0,#0` + 条件跳过后续）在最坏路径上不省，因为后面三条
   都得带条件后缀，条数相同。**没有一条指令能替代或减少任何一步。**
3. **判定**：**IMPOSSIBLE**。附注一件与本判定无关但同族的事实：`pow2f` 是
   跨 TU 的真调用（`bl pow2f`），调用开销存在，但那是内联/LTO 的问题，
   不是 ARMv5TE 指令集能解决的，所以不构成这一格的内容。

#### A5 `expof`（`ops_quant.c`，`LZ_GDN_FIXED` 下）

1. **操作序列**：`(u >> 23) & 0xFF` 减 127。gcc 发出 3 条 + `bx lr`：
   `lsr r0,r0,#23` / `and r0,r0,#255` / `sub r0,r0,#127`。
2. **可用指令核对**：`UXTB`（一条完成 `and #255`）**是 ARMv6，v5TE 拒绝**，
   已用汇编器核实。`CLZ` —— 不是数前导零。`SMLA*`/`MUL` —— 无乘法。
   `QADD` —— 不饱和。桶形移位器 —— `lsr #23` 已用；`and` 的立即数 255 是
   合法的 8 位轮转立即数，不能再合并。访存指令 —— 不访存。
   **三条都不可替代。**
3. **判定**：**IMPOSSIBLE**。

#### A6 `lz_i32f`（`ops_kernel_shared.h`，静态内联，全引擎每一次 int32→float 都走它）

1. **操作序列**：`(float)(v >> 12) * 4096.0f + (float)(v & 0xFFF)`。
   ARM 上展开成 `bl __aeabi_i2f` + `bl __aeabi_fmul` + `bl __aeabi_i2f`
   + `bl __aeabi_fadd`，**四次库调用**，约 105 条指令（在
   `lz_causal_conv1d_step_fixed` 的 `.s` 里可以逐条看到这个序列）。
2. **可用指令核对**：这个二段拆分存在的唯一理由写在它自己的注释里——x87 的
   `fild` 与 SSE 的 `cvtsi2ss` 舍入时机不同，拆开后只有最后那次加法会舍入，
   而加法归 PC=24 管。**ARM 没有 x87，只有一种转换语义**，而且拆分与直接
   转换**可证等价**：`|v>>12| < 2^19` 与 `|v & 0xFFF| < 2^12` 都精确，
   乘 4096 精确（M1），于是唯一的舍入是把精确值 v 舍到最近 float——正是
   一次正确舍入的 int32→float32。可用指令：`CLZ` 一条求出规格化移位量，
   桶形移位器对阶，条件执行处理 round-half-to-even 与零/负数，
   全程约 10–12 条无调用。
3. **判定**：**有独立内容（该写）**。写法：`CLZ` + 移位 + 指数拼装的
   直接 i32→f32；闸门是在 `|v| ≥ 2^24` 的边界区间（以及 `v` 恰好落在
   两个 float 中点的那批值）上与现行拆分式逐位对拍，因为等价性证明
   只在那一段是紧的。

#### A7 `lz_quantize_q8`（`ops.c`）

1. **操作序列**（ARM 无 SIMD 舍入档，`tier == 0`，走 fast 分支）：
   每组一次 `q8_group_scale`；每元素 `ldr` / `bl __aeabi_fmul`
   （`grp[k] * inv`，`inv` 循环不变）/ `bl q8_round` / 两次
   `cmp`+`mov` 钳位 / `strb`。
2. **可用指令核对**：乘数 `inv` 循环不变 → M5（解包一次，循环体只剩
   尾数 `umull` + 规格化，约 12 条而不是 28+调用）。`q8_round` 见 A1。
   钳位到 ±127：`QADD` 的饱和点不对，`SSAT` 是 v6，`cmp`+`movgt` gcc 已发，
   **这一步没有独立内容**。载入：`float*` 4 字节对齐，`LDM` 一次取 4 个
   （M3）。存储：`strb` 四个可以攒成一个 `str`（同 M3 的反方向），
   要求输出 4 字节对齐——`int8_t*` 的组基址是 `gs` 的倍数，`gs ≥ 32`，成立。
3. **判定**：**有独立内容（该写）**。

#### A8 `lz_quantize_q8_tok`（`ops.c`）

1. **操作序列**：与 A7 同形，只是整行一个组（`q8_group_scale(x, n, …)`），
   然后把同一个 `sg` 复制进 `n/32` 个 `s[]` 槽。
2. **可用指令核对**：与 A7 逐条相同（M5 的循环不变乘数、M3 的打包
   载入/存储、A1 的 `q8_round`、钳位无内容）。额外一点：`s[]` 的复制是
   `n/32` 次同值 `str`，`STM` 一条写 4 个。
3. **判定**：**有独立内容（该写）**。与 A7 是同一份内核加不同的分组包装，
   实现时应共享。

#### A9 `lz_quantize_q8_silu`（`ops.c`）

1. **操作序列**：每元素 `og[k] = pg[k] * (gg[k] * (float)sg[k])`
   （i2f + 两次 fmul）→ `q8_amax` → 每元素 `q8_round(og[k]*inv)` + 钳位 →
   每元素再乘 `1/32768`。ARM 上每元素约 4 次库调用 + 一次 `q8_round`。
   **注意**：CE 默认路径上它**不被调用**——把它接进 MoE SwiGLU 的提交
   已被回退（`58fc52e`），forward 走的是 `lz_quantize_q8` / `lz_quantize_q8_tok`。
2. **可用指令核对**：末尾那次 `* (1.0f/32768.0f)` 是 2 的幂 → M1。
   `(float)sg[k]` 是 int32→float → A6。两次一般乘法 → M5。
   `q8_amax` → A2。钳位无内容。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（§3.H）。

#### A10 `lz_quantize_q8_int` / `lz_quantize_q8_tok_int` / `lz_quantize_q8_int64`

1. **操作序列**：整数域 absmax（`t[k] < 0 ? -t[k] : t[k]`，纯整数）→
   `(float)tam * deq` → 与 A7 同形的每元素 `(float)grp[k] * inv_elem`
   + `q8_round` + 钳位。`int64` 版多一次 `__aeabi_l2f` 与那个
   `volatile float tamf`（存在的理由是 x87 会把 `fild+fmul` 融合而 gcc 不会，
   见 `ops.c` 该处注释）。
2. **可用指令核对**：整数 absmax 环——`ldr`/`cmp`/`rsblt`/`cmp`/`movgt`，
   gcc 已用条件执行；`LDM` 打包载入是唯一内容（M3）。元素环的
   `(float)grp[k]` → A6；`inv_elem` 循环不变 → M5；`q8_round` → A1。
   `int64` 版的 `__aeabi_l2f` 同样可用 `CLZ` 内联。那个 `volatile` 中间量
   在 ARM 上是无害的（没有可融合的 `fild+fmul`），汇编格自己保证不融合即可。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`--norm-write auto` /
   `--attn-write auto` 都到不了 `int`，见 §1）。

#### A11 `lz_gdn_quantize_2p`（`ops_gdn.c`）

1. **操作序列**：`q8_group_scale` → 每元素 `f = grp[k]*inv` / `q8_round(f)` /
   钳位 / `l = q8_round((f - (float)h) * 256.0f)` / 钳位。ARM 上每元素
   2 次 `q8_round` + 3 次库调用（fmul、i2f、fsub）+ 一次乘 256。
2. **可用指令核对**：`* LZ_GDN_LO_SCALE`（=256）是 2 的幂 → M1。
   `(float)h`（h ∈ [-127,127]）→ A6，而且量程这么小时可以查表或直接
   拼指数。`f = grp[k]*inv` 循环不变乘数 → M5。两次 `q8_round` → A1。
   钳位无内容。`LDM` 载入 + 两路 `str` 打包存储 → M3。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**：它的两个调用点是
   pass 2 的**浮点**回退（ARM 上 `lz_gdn_p2_mode()` 恒为 fixed，走不到）
   与 `LZ_KV_2PLANE`（默认 0）。

#### A12 `lz_kv4_quantize`（`ops.c`）

1. **操作序列**：`sum x*x`（fmul+fadd 每元素）→ 两次 `sqrt`（libm **double**）→
   一次 fdiv → 每两个元素两次 fmul 与最多 15 次浮点比较走边界表。
2. **可用指令核对**：那 15 次比较是 M2 的教科书案例（每次 ~30 条 → 3 条），
   而且边界表 `kv4_mid` 单调，比较可整体搬进整数域（符号-幅值序）后
   用二分（4 次而不是 15 次）。`sqrt` 走 double libm —— ARMv5TE 无
   平方根指令，这一步没有独立内容。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（只有 `--kv q4` 触发）。

### B. 归一化

#### B1 `norm_ss_fixed`（`ops_quant.c`）—— 定点档的平方和

1. **操作序列**（gcc 发出的 `.L136` 环，逐条数过）：每元素
   `ldr` / `bl __aeabi_fmul`（`x[i] * sc`，`sc = pow2f(14-expof(amax))`）/
   `bl __aeabi_fadd`（`q8_round` 在这里被内联了，剩下的是魔数加）/
   三条 `bic`/`sub` / 四条钳位 / `mul r3,r0,r3` / `adds`+`adc`
   （int64 累加）。约 20 条本地 + 两次库调用 ≈ 88 条/元素。
2. **可用指令核对**：
   - `sc` 是 `pow2f` 的返回值，**精确的 2 的幂** → M1：28 条的
     `__aeabi_fmul` 变 5 条指数域加法。
   - 魔数加 → A1：40 条的 `__addsf3` 变 10–14 条位操作。
   - `v*v` 后 `acc += `：gcc 发 `mul` + `adds` + `adc` 三条。`|v| ≤ 32767`，
     所以 **`SMLALBB` 一条**（16×16 + 64 位累加）就够，ARMv5TE 有，
     汇编器已核实接受。这是 M4 的变体，**减两条并且把 32×32 乘法换成
     16×16**（ARM9E 上后者是单周期发射）。
   - 钳位到 ±32767：无内容（`QADD` 饱和点不对，`SSAT` 是 v6）。
   - `float*` 载入 → `LDM` 四路（M3）。
3. **判定**：**有独立内容（该写）**。这是全表里指令数落差最大的一条：
   88 条/元素 → 约 18 条/元素。

#### B2 `lz_rmsnorm` 的输出环（`ops_norm.c`）

1. **操作序列**：`lz_normout_pick` 在 ARM 上返回 NULL，全部走尾环
   （gcc 的 `.L61`）：每元素 `ldr` / `bl __aeabi_fmul`（`x[i]*inv`）/
   `ldr` / `bl __aeabi_fadd`（`1.0f + w[i]`）/ `bl __aeabi_fmul` /
   `str`，**三次库调用**，约 100 条/元素。
   （注意：`--norm-write` 的定点档只换掉平方和，**这个环在 CE 上仍是浮点**。）
2. **可用指令核对**：`inv` 是 `lz_rsqrt` 的输出，**不是** 2 的幂，M1 用不上；
   `1.0f + w[i]` 也是真加法。剩下的是 M5：`inv` 循环不变，解包一次；
   两次乘法与一次加法内联后共用规格化代码；`LDM` 一次取 4 组 x/w（M3）。
   `UMULL` 是尾数乘法的原生指令（`__aeabi_fmul` 自己就在用它）。
   `CLZ` 用于规格化。100 条 → 约 55–60 条是可达的，但**不存在把它
   变成整数的路**——这是本表里独立内容最薄的浮点环之一，写下来的价值
   一半在指令数、一半在它是 §5 排名第三的量。
3. **判定**：**有独立内容（该写）**。附一条与判定无关但必须记的事实：
   这个环真正的出路是给 norm 加一个**输出侧定点档**（`lz_rmsnorm_out_fixed_int`
   已经存在，只是 `auto` 到不了它），那是档位问题不是汇编格问题，
   两件事互不替代。

#### B3 `lz_rmsnorm_gated` + `lz_rmsnorm_gated_out_fixed`（`ops_norm.c`）

1. **操作序列**（定点档，CE 走这条）：4 次 `q8_amax` + 4 次 `expof` +
   5 次 `pow2f` + 5 次 fmul（前奏，每次调用一次）；然后每元素：
   3 次 `fmul`（各乘一个 `pow2f` 的结果）+ 3 次 `q8_round` + 3 次钳位 +
   1 次 `sigmoid_q15` + int64 链（gcc 发 `smull`/`mul`/`mla`/`umull`/
   `lsr`/`orr` 各三组，约 24 条）+ `__aeabi_l2f` + `pow2f` + `fmul`。
2. **可用指令核对**：
   - 三次 `q8_round(x[i]*s)` 里的 `s` 全是 `pow2f` 的返回值 → **M1 + A1**，
     每个从 ~70 条降到 ~18 条。
   - int64 链：`t = (int64)qx*qw` 两个操作数都已钳到 ±32767 →
     **`SMULBB` 一条**（gcc 发了 64 位的 `smull`，因为源码写的是 int64）。
     后两步 `(t*qg)>>15`、`(t*qinv)>>15` 的被乘数是 31 位、乘数 16 位，
     `SMULL` + `lsr`/`orr` 三条即可（gcc 因为不知道量程发了六条）。
     `SMULWB`（32×16 取高 32，即 `>>16`）能把其中一步做成两条，但
     `>>16` 与 `>>15` 差一位、会丢最低位，**不逐位一致**，所以只作为
     候选记下、不作为写法。
   - 末尾 `(float)t * pow2f(...)` → A6 + M1。
   - `sigmoid_q15` → C1。
3. **判定**：**有独立内容（该写）**。

#### B4 `lz_l2norm`（`ops_norm.c`）

1. **操作序列**：定点档下平方和走 `norm_ss_fixed`（B1），然后
   `lz_rsqrt(ss + eps)`，输出环每元素一次 `bl __aeabi_fmul`。
2. **可用指令核对**：平方和那半完全等同 B1。输出环乘数 `inv` 循环不变
   但不是 2 的幂 → 只有 M5 与 `LDM`。
3. **判定**：**有独立内容（该写）**，内容与 B1/B2 同源，实现时共享。

#### B5 `lz_rmsnorm_int` / `lz_rmsnorm_gated_int` / `lz_rmsnorm_out_fixed_int` / `lz_rmsnorm_gated_out_fixed_int`

1. **操作序列**：与 B3 同形，区别是不做每元素解量化，把 Q15 结果写成 int
   并把 `deq` 带出来。
2. **可用指令核对**：与 B3 逐条相同（M1 的 `pow2f` 乘数、A1 的 `q8_round`、
   `SMULBB` 顶掉 `smull`、`LDM`）。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`lz_norm_int()` 要
   显式 `--norm-write int`，`auto` 到不了，见 §1）。

#### B6 `lz_softmax`（`ops_norm.c`）

1. **操作序列**：最大值扫描（`lz_vmax_pick` 在 ARM 上返回 NULL，走标量
   `bl __aeabi_fcmpgt` 每元素）→ 每元素 `bl __aeabi_fsub` + `lz_exp`
   + `bl __aeabi_fadd` → 一次 fdiv → 每元素 `bl __aeabi_fmul`。
2. **可用指令核对**：最大值扫描是 M2 的正例——比较可以整体搬进整数域
   （引擎在 `q8_amax` 已有先例；这里带符号，用符号-幅值到二补的标准
   翻转，2 条额外指令，之后 `cmp` 即可），~30 条/元素 → ~5 条。
   `x[i] - mx` 与求和是真浮点，只有 M5。末尾 `x[i] *= inv` 的 `inv`
   循环不变 → M5 + `LDM`。`lz_exp` 归 C2。
3. **判定**：**有独立内容（该写）**。

### C. 超越函数

#### C1 `sigmoid_q15` / `sigmoid_q15_t`（`ops_quant.c`）

1. **操作序列**（`sigmoid_q15`，gcc 的 48 条本地 + 8 次库调用）：
   两次浮点比较（钳位到表两端）→ `t = (x - LZ_SIG_LO) * LZ_SIG_STEP`
   （fsub + fmul）→ `idx = (int)t`（f2iz）→
   `frac = (t - (float)idx) * 32768.0f`（i2f + fsub + fmul）→
   两次查表 + `a + (((b-a)*frac)>>15)`。整条 ≈ 250 条指令。
   `sigmoid_q15_t` 省掉入口那对 fsub+fmul，≈ 180 条。
2. **可用指令核对**：
   - `LZ_SIG_STEP` 是 **16.0f**，`* 32768.0f` 是 **2^15**，两个乘法全是
     2 的幂 → **M1**，各 28 条降到 5 条。
   - 两次入口比较 → **M2**，各 ~30 条降到 3 条。
   - `(int)t` / `(float)idx`：`t` 的量程被前面两次钳位限死在 `[0, 384)`，
     所以 `f2iz`/`i2f` 都可以用 `CLZ` + 移位内联（A6 的同一条内容），
     各 ~20 条降到 ~8 条。
   - `(b-a)*frac`：`|b-a| ≤ 32768`、`|frac| < 32768` → **`SMULBB` 一条**，
     gcc 发的是 32×32 的 `mul`。
   - 表项载入：`g_sigtab[idx]` 与 `g_sigtab[idx+1]` 相邻 → **`LDRD` 一条**
     顶两条 `ldr`（表是 `int32_t[385]`，8 字节对齐需要在表声明上加对齐属性，
     属实现细节）。
3. **判定**：**有独立内容（该写）**。250 条 → 约 35 条是可达的，而且
   每一步的逐位一致都是可证的（2 的幂乘法精确、比较语义不变、
   `SMULBB` 与 `mul` 在该量程下同值）。

#### C2 `lz_exp_fixed`（`ops_quant.c`）

1. **操作序列**：两次浮点比较（越界护栏）→ `y32 = x * LZ_EXP_LOG2E32`（fmul）
   → 魔数加取 `floor`（fadd + 位操作）→ `r = y32 - (float)t`（i2f + fsub）
   → `q8_round(r * LZ_EXP_QSCALE)`（fmul + `q8_round`）→ Q20 泰勒
   （gcc 发 `smlal` ×2 + `smull`）→ `(float)prod * pow2f(n - 20)`
   （i2f + fmul）。gcc 计 108 条本地 + 12 次库调用。
2. **可用指令核对**：护栏比较 → M2。`LZ_EXP_QSCALE = 2^20` 与
   `pow2f(...)` 都是 2 的幂 → **M1** ×2。魔数加 → A1。两次 i2f → A6。
   `x * LZ_EXP_LOG2E32` 是唯一的真乘法（M5）。泰勒项：`s` 有界
   （`|s| < 2^20`），`LZ_EXP2_LN2` 是 20 位常数，`SMULL`+移位就够；
   `(int64)LZ_EXP2_LN2SQ * s * s` 是两次 32×32，gcc 已发 `smlal`/`smull`，
   这一段**没有**额外内容。
3. **判定**：**有独立内容（该写）**，内容集中在四处 2 的幂乘法、
   两次比较、两次转换与 `q8_round`，泰勒本体不动。

#### C3 `lz_exp_float_body`（`ops_quant.c`）

1. **操作序列**：6 次 fmul + 6 次 fadd/fsub + 1 次 i2f（该函数自己的
   `LZ_FCX` 计账就是这个），全是真浮点。CE 上只有 `sig_build`（建表，
   每模型一次）和 `--scalar-fixed float` 时才走到。
2. **可用指令核对**：`w = 1.0f + r * 0.03125f` 里 `0.03125 = 2^-5` 是
   2 的幂 → M1；`p2n.i = (n+127) << 23` 已经是位操作；其余乘加是
   二阶多项式的真浮点 → 只有 M5。i2f → A6。
3. **判定**：**有独立内容（该写）**，权重接近零（CE 默认走 C2）。

#### C4 `lz_rsqrt_fixed`（`ops_quant.c`）

1. **操作序列**：位域取 `e`/`m23`/`idx` → 查 Q23 种子表 → 两轮整数牛顿
   （gcc 发 `umull` ×4 + `mla` ×4 + `smlal` ×2 + `mul` ×2）→
   `(float)y23 * pow2f(-(e>>1)-23)`，奇 `e` 再乘 `LZ_RSQRT_SQRT2`。
   gcc 统计：181 条本地 + **18 次 `__aeabi_fmul`**（跨分支重复，实际
   执行路径 2–3 次）+ 若干 i2f/fsub。
2. **可用指令核对**：牛顿本体是纯整数 64 位乘加，gcc 已经在用
   `umull`/`mla`/`smlal`，**这一段没有独立内容**——`SMLAL` 就是它该发的
   指令，它发了。独立内容全在尾巴：`(float)y23` → A6；`* pow2f(...)`
   → M1；`* LZ_RSQRT_SQRT2` 不是 2 的幂 → M5。另外 `e < -126 || e > 126`
   的护栏是整数比较，已最优。
3. **判定**：**有独立内容（该写）**，但**范围要写清楚**：只在解量化尾巴，
   牛顿迭代本体是 IMPOSSIBLE 的（`SMLAL`/`UMULL` 已被 gcc 用满）。

#### C5 `lz_rsqrt_float_body`（`ops_quant.c`）

1. **操作序列**：一次位操作种子 + 两轮 `y*(1.5f - h*y*y)`，共 7 次 fmul +
   2 次 fsub，全是真浮点。CE 上只有 `rsqrt_tab_build`（建表）与
   `subn_fwht_quant` 里的 `lz_rsqrt_float_body((float)blk)` 走到，后者
   每次调用一次（`blk` 是常数，结果可缓存，但那是 C 层的事）。
2. **可用指令核对**：`0.5f`/`1.5f` 不是能省掉乘法的形式（`0.5f * x`
   **是** 2 的幂 → M1 可用于 `h = 0.5f*x` 那一步）；其余 6 次乘法与
   2 次减法是真浮点 → M5。
3. **判定**：**有独立内容（该写）**，权重接近零。

#### C6 `lz_softplus`（`ops.c`）

1. **操作序列**：`x > 20.0f` 早退；否则 `(float)log1p(exp((double)x))`
   —— ARM 上是 `__aeabi_f2d` + glibc `exp`（double）+ glibc `log1p`（double）
   + `__aeabi_d2f`。这两个 libm 本体的静态量（`ar x
   /usr/arm-linux-gnueabi/lib/libm.a e_exp.o s_log1p.o`）：`__exp` 327 条
   本体 + 37 次 `__aeabi_d*`；`__log1p` 459 条 + 57 次 + 4 次 `ddiv`。
   被调的各是 `__aeabi_dmul` 155 条、`__adddf3` 171 条、`__aeabi_ddiv` 129 条。

   **这些是静态数，取用路径是其子集，所以这一条用执行指令计数钉死了**
   （方法见 §0 第 4 条；驱动 `.prof/arm_sp_drv.c` 链接的是真的
   `src/ops.c`，不是它的抄本）：

   | 输入集 | `lz_softplus` | `lz_sigmoid` | 对照臂 |
   |---|---|---|---|
   | **真实 KDA 衰减门实参** | **3,440** | **321** | 50.0 |
   | 合成均匀 [-20, 20] | 3,370 | 228 | 50.2 |
   | 合成全 25.0（走早退） | 39 | 130 | 48.0 |

   单位是每次调用的执行指令数，含 `bl`/返回。三条判据都过了：
   **零对照** —— 九个二进制在 `N=0` 上都是 21,344 条，逐条相同；
   **线性** —— 真实输入集上 `X(512)-X(256)` 与 `X(256)-X(0)` 相差
   135 条 / 256 次调用（0.02%）；**真实分布** —— 见下。
   `lz_sigmoid` 的两个窗口相差 335,979 条，那是 `sig_build()`
   一次性建 385 项表的钱，三个输入集上都是同一个数（±400），
   所以它是每进程一次而不是每次调用——**用第一个窗口测就会把它
   摊进每次调用，虚报 5 倍**。表里取的是 256→512 那个稳态窗口。

   **真实分布是量出来的，不是设的**（`.prof/arm_softplus_dist.py`）：
   在 `E:/LLM/models/kunkun-ce` 上挂 `f_b_proj` 的前向钩子、跑 220 个
   真 token、加上 `dt_bias`，得到 450,560 个实参
   （= 220 token × 4 个 KDA 层 × 512 通道）：

   ```
   min -13.08   1% -13.04   中位 -0.56   99% 8.34   max 8.38
   mean -1.17   std 4.07          frac(x > 20.0f) = 0.000000
   ```

   **`x > 20.0f` 那条早退在真实数据上一次都不触发**，所以 2,048 次
   调用全部走完整的 double libm 路径。这一条比分布的形状重要得多：
   成本是双峰的（39 或 3,440，差 88 倍），而真实分布 100% 落在慢的那一峰。
   合成均匀 [-20,20] 只低 2.0%，因为它同样几乎没有质量落在 20 以上——
   也就是说这次「合成会偏向快路径」的担心没有兑现，但**那是量过之后
   才知道的**，而它本可以完全兑现：任何把采样范围放宽到 20 以上的
   合成集都会按落在上面的比例线性地把数拉向 39。

   **在定案配方下这个函数每 token 调用 0 次。** `forward.c` 的
   `if (c->kda_has_gate_lower_bound)` 选的是另一支，而
   `kda_gate_lower_bound` 已定案为 **-5.0**
   （`docs/kunkun-ce-params.md` §1；`E:/LLM/models/kunkun-ce/config.json`
   同步为 `-5.0`，2026-08-21 复核）。衰减门那 2,048 次/token 现在
   全部走 `lz_sigmoid`，即 **0.66 M 指令/token**，而不是 softplus 的
   **7.05 M**。
   （第二条臂的实参严格说是 `exp(A_log)·x` 而不是 `x`；四个 KDA 层的
   `exp(A_log)` 均值是 0.27 / 1.28 / 0.30 / 0.27，即同量级，
   而 `lz_sigmoid` 在 [-20,20] 与真实集之间只差 93 条，
   对上面这个差不构成威胁。）
2. **可用指令核对**：把 double libm 换成手写 ARM 汇编不是「填一个 ISA 格」，
   是重写 libm——那超出本条铁律的范围。ARMv5TE 在这个序列上能作用的只有
   `x > 20.0f` 那一次比较（M2）与 f2d/d2f 两次转换（`CLZ` + 移位）。
   double 的乘加本身：`UMULL`/`UMLAL` 正是 `__aeabi_dmul` 已经在用的指令。
3. **判定**：**不在 CE 配方的路径上，这一格作废**（§3.H 已登记）。
   即使它在，判定也只会是「有独立内容，但内容只有护栏与两次转换」：
   `exp`/`log1p` 本体归 IMPOSSIBLE，能动的只有那条 39 条的早退路径，
   而真实分布上它一次都不触发。这一格从来就收不到东西。

**这一节留着，因为它是本次核对里唯一一处「量出来的数反过来改了配方」。**
把区间（当时是 2,000–5,000 条/次的上下界猜测）换成实测的 3,440 条之后，
`kda_gate_lower_bound` 这个字段本身的代价成了一个确定的数 ——
**6.39 M 指令/token**（`2,048 × (3,440 − 321)`），
比本文档里任何一格汇编能收的量都大，而 C6 这一格本身收不到东西。
这个数是定案 -5.0 的依据之一；另两条不在本文档范围内
（K3 生产配置就是 -5.0；「打开它是 identity break」的顾虑只对
转换已训权重成立，CE 从零训不适用）。

留下的方法论是：**一个跨越决策边界的区间不能带着往下走**。
当时 softplus 在 `T`=0 上排第 1、在 `T`=1536 上排第 4~6，
区间的两端指向相反的动作；量到 ±0.02% 之后，指向的动作
既不是「优先写它的汇编」也不是「不管它」，而是删掉这条路径。

#### C7 `lz_sigmoid` / `lz_silu` / `lz_exp` / `lz_rsqrt`（派发壳，`ops.c`）

1. **操作序列**：读一个档位全局量（`bl lz_sig_mode` / `bl lz_scalar_mode`）
   然后分支到 C1/C2/C3/C4/C5。`lz_silu` 额外一次 fmul。
2. **可用指令核对**：一次载入、一次比较、一次分支；`LDM` 无从下手，
   `SMLA*` 无乘法，条件执行已用。`lz_silu` 的那次 fmul 归 M5。
3. **判定**：**IMPOSSIBLE**（壳本身）。内容全在被派发的本体里。

### D. 递推（KDA / GDN）

#### D1 `gdn_build_table`（`ops_gdn.c`，static）

1. **操作序列**：每 gg 两趟遍历 kd。第一趟每 kk：2 次 fmul + 2 次浮点
   取绝对值 + 2 次浮点比较（求 amk/amq）。第二趟每 kk：4 次 fmul +
   2 次 `q8_round` + 4 次钳位 + 4 次 `strh`。CE 上 kd=32、ng=1，
   每头 64 次 `q8_round`、128 次 fmul。
2. **可用指令核对**：取绝对值与比较全是 M2（`bic` 清符号位后整数比，
   引擎在 `q8_amax` 已有先例），~30 条 → ~4 条。`q8_round` → A1。
   `ik`/`iq` 循环不变 → M5。四个 `strh` 写的是 `slot[0],slot[2]` /
   `slot[4],slot[6]` —— 不相邻，`STM` 用不上；但把同一个 int16 复制成
   一个 32 位字（`orr r, r, r, lsl #16`）再一次 `str` 可行，前提是
   `slot[0..1]` 与 `slot[2..3]` 各自 4 字节对齐（`LZGdnTable.t` 是
   `int16_t[..][8]`，成立）。
3. **判定**：**有独立内容（该写）**。

#### D2 `gdn_sum_gg` 标量分支（`ops_gdn.c`）—— pass 1 定点内核

1. **操作序列**（gcc 的 `.L8` 环，逐条数过，16 条覆盖一个 vv 车道
   的一对 kk，产出 4 次 MAC）：

   ```
   ldrsb r9,[r0,#1]!   ldrsb fp,[r1,#1]!   smulbb / smlabb
   ldr / add / str                       (au 累加)
   ldrsb r9,[r0]       ldrsb fp,[r1]      smulbb / smlabb   ← 重新载入同两个元素
   ldr / add / str                       (aw 累加)   cmp / bne
   ```

   即 **4 条指令换 1 次 MAC**。CE 上每 token 262,144 次 MAC（16 头 ×
   16 kk 对 × 32 车道 × 4 MAC × 2 个状态平面 × 4 个 KDA 层）。
3. **可用指令核对**：
   - **重复载入**：`rowA[j]`/`rowB[j]` 被各载入两次（一次给 au、一次给 aw），
     这是寄存器分配失手，汇编格直接消掉——**减 4 条中的 2 条**。
   - **`LDM`/打包载入（M3）**：`rowA` 是 `const int8_t*`，gcc 只能发
     `ldrsb`；状态行基址是 32 的倍数、来自 `calloc`，4 字节对齐，
     一条 `ldr` 取 4 个车道，配合 `sxtb`（v5TE 无 `SXTB`，用
     `mov r,r,lsl #24; mov r,r,asr #24` 两条，或直接让 `SMLA*` 吃
     未扩展的半字并在系数侧补偿）。
   - **系数半字选择（M4）**：`tab[kp][0..1]` 是 `ckA,ckB`、`[4..5]` 是
     `cqA,cqB`，**相邻的 int16 对** —— 一条 `ldr` 装两个系数，
     `SMLABB`/`SMLABT` 各取一半。gcc 发的是两条 `ldrsh`。
     这与 `ops_t2_arm.c` 汇编格注释里第 2 条（0x00030003 掩码一次抬两个码）
     是同一形状的机会。
   - **累加器搬进寄存器**：`au_gg[j]`/`aw_gg[j]` 每次迭代都 `ldr`+`str`。
     把 kk 环放到内层、j 环放外层，两个累加器整段留在寄存器里，
     `ldr`/`add`/`str` 三条降到一条 `SMLABB`。整数和在重结合下精确
     （该函数注释自己写了「integer sums are exact under reassociation，
     与 MMX 版逐位一致」），所以这个换序**不改一个位**。
   - `SMLAL`：不需要，int32 累加器的界是 127×32767×256 ≈ 1.06e9，
     在 int32 内（该文件注释已给出）。
   - `PLD`：状态行是顺序流式访问，但按铁律八它是正交档位，不进内核。
3. **判定**：**有独立内容（该写）**。16 条/车道对 → 目标 4–5 条。

#### D3 pass-1 尾声（`gdn_pass1_fixed_head` 的解量化环，`ops_gdn.c`）

1. **操作序列**（两平面，CE 默认）：每 vv
   `u[vi] = (lz_i32f(au) + lz_i32f(aul) * (1/256)) * su[gg]`，w 同理。
   即每 vv **4 次 `lz_i32f`**（每次 4 库调用，见 A6）+ 2 次
   `* (1/256)` + 2 次 `* su[gg]` + 2 次 fadd。CE 上每 token
   8,192 次 `lz_i32f`（32 vv × 4 × 16 头 × 4 个 KDA 层）。
2. **可用指令核对**：`1.0f / LZ_GDN_LO_SCALE` = 2^-8，**2 的幂** → M1。
   `su[gg]`/`sw[gg]` 在 32 个车道上循环不变 → M5（解包一次）。
   四次 `lz_i32f` → A6（每次 105 条 → 12 条）。
   更强的一步：`au` 与 `aul` 都是 int32，`au + aul/256` 可以先在**整数域**
   完成（`(au << 8) + aul` 再整体乘 `su/256`），把两次转换合成一次——
   但那**会改舍入**（现行是两次独立转换后浮点相加），所以只作为候选
   记下、不作为写法。
3. **判定**：**有独立内容（该写）**。

#### D4 `gdn_p2_group_fixed` 的车道环（`ops_gdn.c`）—— pass 2 定点内核

1. **操作序列**：每 32 车道两趟。第一趟：`H = (ph[j] << 8) + pl[j]`（两平面）
   → `A[j] = m1*H + m2*dq[j]` → 追踪 `amax`（两次比较+取反）。
   第二趟：带舍入右移 `(A[j] + (1<<(sh-1))) >> sh` → `h = (hn+128)>>8` →
   钳位 ±127 → `l = hn - (h<<8)` → 钳位 ±127 → 两次 `strb`。
   CE 上每 token 65,536 个车道更新（16 头 × 32 行 × 32 车道 × 4 个 KDA 层），
   每个约 12 条整数指令。
2. **可用指令核对**：
   - `ph[j]`/`pl[j]` 是 int8 → **`LDM` 一条取 4 车道**（M3），gcc 发 `ldrsb`。
   - `m1*H`：`|m1| ≤ 32767`，`|H| ≤ 127*256+127 ≈ 2^15` → 两个 16 位量，
     **`SMULBB`** 一条；`m2*dq[j]` 两个都是 int16 → **`SMLABB`** 一条，
     于是 `A[j] = m1*H + m2*dq[j]` 是两条而不是 `mul`+`mla`（32×32）。
     `m1,m2` 存在相邻的 `mul4[0..1]`，一条 `ldr` 装两个（M4）。
   - `amax` 追踪：`cmp`/`movgt` + `rsb`/`cmp`/`movgt`，gcc 已用条件执行，
     **无内容**；但换成「先算 `A[j]` 的绝对值再比」可以用
     `eor r,a,a,asr #31; sub r,r,a,asr #31`（2 条，无分支），与
     gcc 的 4 条持平偏优——记为边际，不作为理由。
   - 带舍入右移与 `(hn+128)>>8`：桶形移位器已被 gcc 用上，`add`+`asr`
     两条，**无内容**。
   - 钳位 ±127：`QADD` 饱和点不对，`SSAT` 是 v6，**无内容**。
   - `strb` ×2 每车道 → 攒成 `str` 一条写 4 个（M3）。
   - `p2_shift_of(amax)`：`ops_kernel_shared.h:108` 是
     `while ((amax >> sh) > 32512) sh++;` —— 一个**比较循环**，最坏 16 次
     迭代。一条 `CLZ` 直接给出前导零数，减去常数即得 `sh`（32512 不是
     2 的幂，要多一条修正比较）。**一条顶一整个循环**，是本条目里
     落差最大的一处。
3. **判定**：**有独立内容（该写）**。

#### D5 `p2_group_coef` / `p2_group_norm` / `p2_group_scale`（`ops_gdn.c`）

1. **操作序列**（每（行，组）一次，CE 上每 token 2,048 次 = 32 行 × 16 头 × 4 层）：
   `coef` 3 次 fmul；`norm` 两次浮点取绝对值 + 两次比较 + `expof` +
   两次 `pow2f` + 3 次 fmul + 2 次 `q8_round` + 4 次钳位；
   `scale` 一次比较 + `pow2f` + 2 次 fmul。
2. **可用指令核对**：`p = pow2f(14-e)`、`sacc = pow2f(e-14)`、
   `pow2f(sh)`、`1.0f/LZ_GDN_LO_SCALE`（2^-8）—— **五处乘 2 的幂**，
   全归 M1。取绝对值与三次比较 → M2。两次 `q8_round` → A1。
   `expof` 本身 IMPOSSIBLE（A5）但**调用点**可以内联那三条。
   钳位无内容。
3. **判定**：**有独立内容（该写）**。

#### D6 `gdn_step_impl` 的标量尾声（`ops_gdn.c`）

1. **操作序列**：每 vv `u[vv] = (v[vv] - gt*u[vv]) * beta`；
   `out[vv] = gt*w[vv] + u[vv]*kq`。4 次 fmul + 2 次 fadd/fsub，
   `gt`/`beta`/`kq` 三个乘数全部循环不变。CE 上每 token
   16 头 × 32 vv × 4 个 KDA 层 = 2,048 个 vv。
2. **可用指令核对**：三个循环不变乘数 → M5（各解包一次；`gt` 在 KDA
   家族恒为 `1.0f`，`gt*u` 与 `gt*w` 因此是恒等——**但删掉它会改哪一位
   都不改**，因为乘 1.0f 精确，这是 `lz_kda_step` 注释自己写的论据，
   所以汇编格可以整支砍掉两次乘法）。`LDM`/`STM` 打包 u/w/v/out（M3）。
3. **判定**：**有独立内容（该写）**。

#### D7 pass 1 / pass 2 的浮点回退（`gdn_step_impl` 内）

1. **操作序列**：pass 1 每元素 2 次 fmul + 2 次 fadd（两平面下另加
   一次 i2f 与一次 2^-8 乘法）；pass 2 每元素 `bb[j] = (…)*scg + kc*dd[j]`
   后整行 `lz_quantize_q8` / `lz_gdn_quantize_2p`。
2. **可用指令核对**：与 D3/A11 同族（M1 的 2^-8、A6 的 i2f、M5 的
   循环不变 `scg`/`kc`）。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`lz_gdn_mode()`
   与 `lz_gdn_p2_mode()` 在 ARM 上都恒为 fixed）。

### E. 因果卷积

#### E1 `lz_causal_conv1d_step_fixed` 的抽头环（`ops.c`）

1. **操作序列**（gcc 的 `.L649`，5 条/抽头）：
   `ldrh ip,[r1,#2]!` / `ldrh r2,[r0,#2]!` / `cmp` / `smlabb r7,ip,r2,r7` / `bne`。
   CE 上每 token 24,576 个抽头（1536 通道 × k=4 × 4 个 KDA 层）。
2. **可用指令核对**：
   - **打包载入（M3）**：`mw`（int16 抽头）与 `sh_in`（int16 历史）都是
     连续 int16，一条 `ldr` 取两个；`LDRD`/`LDM` 取四个。
   - **半字选择（M4）**：装进一个 32 位字的两个 int16，
     `SMLABB` + `SMLATT` 各取一半 —— **两次 MAC 只花 2 条载入 + 2 条 MAC**，
     现行是 4 条载入 + 2 条 MAC + 循环开销。k=4 时整环可完全展开成
     2 条 `ldr` + 2 条 `ldr` + 4 条 `SMLA*`。
   - **`SMLAL` 不需要**：`lz_conv_accum_bound` 的推导（`ops.c:55` 附近）
     把界压在 int32 内，而且那段注释自己就点了「ARMv5TE 有 SMLAL，
     但 MMX 的 pmaddwd 会同样溢出，所以两边共用一个界」。
3. **判定**：**有独立内容（该写）**。

#### E2 `lz_causal_conv1d_step_fixed` 的每通道尾声（`ops.c`）

1. **操作序列**（gcc 的 `.L648` 之后，逐条数过）：
   `bl __aeabi_i2f` / `bl __aeabi_fmul` / `bl __aeabi_i2f` / `bl __aeabi_fadd`
   （这四条是 `lz_i32f(acc)`）→ `bl __aeabi_fmul`（`*sig_k1[c]`）→
   `bl __aeabi_fadd`（`+sig_k2`）→ `bl sigmoid_q15_t` → `bl __aeabi_fmul`
   （`*sig_oscale2[c]`）→ `bl __aeabi_i2f`（`(float)s15`）→
   `bl __aeabi_fmul`。**每通道 10 次库调用**；再加入口的
   `x[c]*in_scale` + 比较 + `f2iz`。CE 上每 token 6,144 个通道
   （1536 × 4 个 KDA 层）。
2. **可用指令核对**：`lz_i32f` → A6（105 条 → 12 条）。
   `in_scale = pow2f(es)` 是 **2 的幂** → M1。
   `(float)s15`（`s15 ∈ [0, 32768]`）→ A6，量程小到可以只用移位。
   `sig_k1[c]`/`sig_oscale2[c]` 是每通道常数、在整个 token 上不变 → M5。
   入口的 `t < 0.0f` 比较 → M2；`(int)(t±0.5f)` 那次取整 → 与 A1 同族的
   位操作。`sigmoid_q15_t` → C1。
3. **判定**：**有独立内容（该写）**。

#### E3 `lz_causal_conv1d_step`（浮点档，`ops.c`）

1. **操作序列**：每抽头 fmul + fadd；每通道一次 `lz_silu`；历史搬移。
2. **可用指令核对**：`wr[j]` 与 `st_in[j]` 是 `float*` → `LDM`（M3）；
   两个乘数都随 j 变，M1 用不上，只有 M5。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`--conv-write auto`
   在 ARM 上恒为 fixed）。

#### E4 `lz_conv_norm_pow2` / `lz_conv_build_scales` / `lz_conv_accum_bound`（`ops.c`，装载期）

1. **操作序列**：`norm_pow2` 是一次浮点 amax + 一个 `while` 找指数 +
   每元素 fmul/取整/钳位；`build_scales` 是每通道一次 `pow2f`；
   `accum_bound` 是一个 double 比较的倍增循环。三者都在
   `conv_fixed_build` 里，**每模型一次**，不在 token 路径上。
2. **可用指令核对**：`norm_pow2` 的 amax 与钳位归 M2/A1；
   `v[i] * (float)(1 << e)` 是 2 的幂 → M1；`accum_bound` 的 double
   乘与比较（`__aeabi_dmul` 155 条 ×2、`__aeabi_dcmple`）在整数域是
   一次 `umull` + 比较。
3. **判定**：**有独立内容**，**权重为零级**（每模型一次，不随 token 增长）。
   排队时放在最后。

### F. 注意力

#### F1 `lz_attn_score_q8`（`ops_gdn.c`）

1. **操作序列**：每头一次 `lz_quantize_q8(hd, 32)` + `hd` 次 int8→int16 拓宽；
   然后每 t：`ng` 次 `LZ_ATTN_DOT32`（ARM 上是 `dot32_x16_attn_ref`，
   32 次 `w[k]*256*x[k]`）+ 每组 `lz_i32f(rr[g]) * (qs[g]*kts[g])`
   （i2f 链 + 2 次 fmul + 1 次 fadd）+ 收尾 `((a0+a2)+(a1+a3)) * (1/256) * scale`。
   CE 上每 token 512·T 次 MAC 与 16·T 次 `lz_i32f`（2 层合计 1024·T / 32·T）。
2. **可用指令核对**：
   - 内积本体 `dot32_x16_attn_ref` 与 matmul 的行内核同形（`w` 是 int8、
     `x` 是 int16）→ `LDM` 打包 + `SMLABB`/`SMLATT` 半字选择（M3+M4），
     与 `lz_dot32_t2_arm_asm` 已经写下的三条理由逐条对应。
     **这一段应与 matmul 任务共享实现**（§6）。
   - `* 256` 是 2^8 → 在整数域是一条 `lsl`，gcc 已经这么发（`mov r,r,lsl #8`
     形式），**无额外内容**；末尾 `* (1.0f/256.0f)` 是 2 的幂 → **M1**。
   - `lz_i32f` → A6。`qs[g]*kts[g]` 两个都随 g 变 → 只有 M5。
   - `qw[i] = (int16_t)qq[i]` 的 int8→int16 拓宽环：`LDM` 取 4 个 int8、
     移位拆开、`orr` 成两个 32 位字、`STM` 写回（M3），gcc 发的是
     每元素 `ldrsb`+`strh`。
3. **判定**：**有独立内容（该写）**。

#### F2 `lz_attn_wsum_q8`（`ops_gdn.c`）

1. **操作序列**：每组两趟。第一趟每 t：`att[t] * vs[…]`（fmul）+
   取绝对值 + 比较（求 cmax）。然后一次 fdiv，每 t 一次
   `q8_round(cbuf[t]*inv)` + 钳位。第二趟每对行 × 32 车道（gcc 的 `.L79`，
   9 条覆盖 2 次 MAC）：

   ```
   ldrsb / ldrsb / smulbb / smlabb / ldr / add / cmp / str / bne
   ```

   最后每 512 行块 32 次 `lz_i32f` + 32 次 fmul 收尾。
   CE 上每 token 512·T 次 MAC（2 层合计 1024·T），16·T 次 `q8_round`（合计 32·T）。
2. **可用指令核对**：
   - 第二趟：`rowA`/`rowB` 是 `const int8_t*` → `LDM` 打包（M3）；
     `coef[0..1]` 是相邻 int16 → 一条 `ldr` + `SMLABB`/`SMLABT`（M4）；
     `acc32[d]` 的 `ldr`/`add`/`str` 可以整段留在寄存器里（32 个车道
     分 4 组、每组 8 个寄存器），整数和重结合精确，**不改一个位**
     （该函数注释已经据此论证过 MMX 与标量的逐位一致）。
     9 条/2 MAC → 目标 2.5 条/2 MAC。
   - 第一趟：取绝对值与 cmax 比较 → M2；`att[t]` 循环不变 → M5。
   - `q8_round` → A1；钳位 ±32767 无内容。
   - `lz_i32f` → A6；`sscale` 收尾乘法循环不变 → M5。
   - `LZ_WSUM_CHUNK` 分块与 `LZ_SLOT_NEXT` 的环形游走是纯整数
     `cmp`/`mov`，条件执行已用，**无内容**。
3. **判定**：**有独立内容（该写）**。

#### F3 `lz_attn_wsum_q8_int`（`ops_gdn.c`）

1. **操作序列**：与 F2 相同，只是把每块 int32 累加器加进 int64 而不是
   转成 float。
2. **可用指令核对**：与 F2 逐条相同；额外一处 `acc64[d] += acc32[d]`
   是 `adds`/`adc`，gcc 已最优。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`lz_attn_int()`
   要显式 `--attn-write int`）。

### G. 变换与位置编码

#### G1 `lz_fwht_i32`（`fwht.c`）—— SubLN 的定点 Hadamard

1. **操作序列**（gcc 全函数 45 条，内环 `.L4` **8 条/蝶形**）：

   ```
   ldr r3,[r1,#4]!   ldr r0,[r2]   add ip,r3,r0   sub r3,r3,r0
   str ip,[r1]       str r3,[r2],#4   cmp r2,lr   bne .L4
   ```

   外加入口一次 `bl memcpy`。CE 上每 token **129,024 次蝶形**
   （KDA o_proj：4 层 × 512·log2(512) = 18,432；dense down_proj：
   6 层 × 4 块 × 512·9 = 110,592）。
   **down_proj 那 110,592 与 8 层 / inter 1536 的几何完全相同**——
   蝶形数是 `n_layers × inter × log2(blk)`，而 `n_layers × inter` 守恒
   （`6×2048 == 8×1536 == 12,288`）。这是本表里收缩最少的一行，
   也是它在 §5 里往上挪的原因。
2. **可用指令核对**：
   - **`LDM`/`STM`（M3）**：`tmp[i+j .. i+j+3]` 连续，`tmp[i+j+len .. +3]`
     也连续，缓冲区是 `int32_t*`（`forward.c` 的 `fwht_scratch`，来自
     `xcalloc`，4 字节对齐）。`len ≥ 4` 的阶段一条
     `ldmia r1!,{r4-r7}` + 一条 `ldmia r2,{r8-r11}` + 8 条 add/sub +
     两条 `stm` ≈ 14 条做 4 个蝶形，即 **3.5 条/蝶形**，对 8 条。
     gcc 发不出这个形状：它偏好后变址 `ldr`，`LDM` 的启发式只覆盖
     结构体拷贝与寄存器保存。
   - `len < 4` 的最后两阶段：`u` 与 `v` 相邻，一条 `LDRD` 取两个，
     `add`/`sub` 后一条 `STRD` 写回。
   - `SMLA*`/`SMLAL`/`MUL`：变换体内**一次乘法都没有**（Sylvester 蝶形
     只有加减），全族用不上。
   - `QADD`/`QSUB`：int32 饱和，而这里要的是回绕；该函数的界由
     `subn_fwht_quant` 的 `2^14` 规格化保证（`fwht.h` 的注释写了
     `blk*32767 ≈ 1.68e7 < 2^31`），**不需要也不允许饱和**。
   - `CLZ`：无归约。`PLD`：正交档位。
   - 入口的 `memcpy`：`y != x` 时的整块拷贝，`LDM`/`STM` 已是 libc 的做法，
     但汇编格可以把**第一阶段的蝶形与拷贝合并**（读 x 直接写 y 的
     和差），省掉整整一趟 n 元素的读写。这是纯算法重排，逐位不变。
3. **判定**：**有独立内容（该写）**。8 条/蝶形 → 3.5 条/蝶形，外加
   省掉入口那趟 `memcpy`。
   **敏感性**：这一条整个建立在 `use_subn=1` 上；`use_subn=0` 时
   `subn_fwht_quant` 不被调用，本算子每 token 0 次。CE 定案与盘上的
   `kunkun-ce` 检查点都是 `use_subn:True`，所以这条计数按 1 取。

#### G2 `lz_fwht`（浮点，`ops.c`）

1. **操作序列**：每蝶形 `bl __addsf3`（加）+ `bl __addsf3`（减），
   约 80 条。调用点只有 `rot_head`，由 `--kv-rot` 控制，**默认 off**
   （`cli_main.c:224` 写的是 `default OFF`），CE 上每 token 0 次。
2. **可用指令核对**：`a+b` 与 `a-b` 共用同一份对阶工作（比较指数、
   移位对齐尾数），而两次 `__addsf3` 调用把这份工作做了两遍——**一次
   解包+对阶服务两个结果**，这是 M5 的最强形式。`LDM` 打包（M3）。
   无乘法，`SMLA*` 族全不适用。
3. **判定**：**有独立内容**，但**不在 CE 默认路径上**（`--kv-rot` 默认关）。

#### G3 `lz_rope`（`ops.c`）

1. **操作序列**（ARM 上 `LZ_ROPE_SSE2_EXTERN` 未定义，走标量环）：
   每个 (i, h) 对 4 次 `bl __aeabi_fmul` + 1 次 `bl __aeabi_fsub` +
   1 次 `bl __aeabi_fadd`，约 200 条。CE 上每 token 640 次
   （2 层 ×（8 q 头 + 2 kv 头）× 32 个频率对）。
2. **可用指令核对**：`c` 与 `s` 在**整个 h 环上循环不变**（源码结构就是
   i 在外、h 在内，注释没写但代码如此）→ M5 的最强适用点：
   两个乘数各解包一次，内环只剩尾数 `UMULL` + 规格化。
   `x0*c - x1*s` 与 `x1*c + x0*s` 共用 `x0`、`x1` 的解包，
   四次乘法只需两次操作数解包而不是八次。
   `LDM`/`STM` 打包 `base[i]`、`base[i+half]`（`float*`，4 字节对齐，M3）。
   `SMLA*`：这是浮点，用不上；除非引入定点 RoPE 档，而
   **那会破坏该函数注释里明写的「每个旋转是独立的 IEEE 乘加」逐位契约**，
   属于另开档位，不是这一格。
3. **判定**：**有独立内容（该写）**。

### H. CE 默认配方里跑不到的算子（登记，不排序）

每条都给出「怎么确认的」，而不是「看起来不会跑」。

| 算子 | 为什么跑不到 | 确认方式 |
|---|---|---|
| `lz_gdn_step` | CE 是 4 KDA + 2 全注意力，0 个 `LZ_LT_LINEAR` 层 | `kunkun-ce-params.md` §1 的 `LLALLA`；`forward_chunk` 按 `L->type` 派发，`LZ_LT_KDA` 走 `forward_kda` |
| `lz_moe_route` / `lz_moe_hits_add` | `num_experts=0` + `first_k_dense_replace = n_layers` → 每层 `ffn_moe=0` | `kunkun-ce-params.md` §1；`forward_chunk` 的 `if (L->ffn_moe)` |
| `lz_fwht`（浮点） | `--kv-rot` 默认 off → `s->kv_rot_k = s->kv_rot_v = 0` | `forward.c:891`；`cli_main.c:224` `default OFF` |
| `lz_kv4_quantize` | 只有 `--kv q4` 分配 `k4`/`v4` 平面 | `forward.c:1351/1360` 的 `if (v4)` / `if (k4)` |
| `lz_gdn_quantize_2p` | 两个调用点分别在 pass 2 的**浮点**回退与 `LZ_KV_2PLANE`（默认 0） | `ops_gdn.c:1871`（在 `lz_gdn_p2_mode()` 的 `return` 之后）；`ops.h:1293` |
| `lz_quantize_q8_int` / `_tok_int` / `_int64`、`lz_rmsnorm_int` / `_gated_int` / `_out_fixed_int` / `_gated_out_fixed_int` | `auto` 解析不到 `int`：`lz_norm_int()` 要 `g_norm==LZ_NORM_INT`，`lz_attn_int()` 要 `g_attn==4`，两者的 auto 值分别是 -1 | `ops_sched.c` 的 `lz_norm_mode`/`lz_norm_int`/`lz_attn_int`；`forward.c` 每个调用点的 `if (lz_norm_int() && …)` |
| `lz_attn_wsum_q8_int` | 同上，且额外要求 `gso==32 && !v4 && !vf && !LZ_KV_2PLANE` | `forward.c:1435` 的 `attn_int[tk]` 表达式 |
| `lz_causal_conv1d_step`（浮点）、GDN pass 1/2 的浮点回退 | `lz_conv_mode()` / `lz_gdn_mode()` / `lz_gdn_p2_mode()` 在 `LZ_NO_FPU` 下恒为 fixed | `ops_sched.c:127` 的 `lz_soft_float_target()` |
| `lz_quantize_q8_silu` | 接进 MoE SwiGLU 的改动已回退，forward 里没有调用点 | `grep lz_quantize_q8_silu src/forward.c` 无命中；提交 `58fc52e` |
| `lz_softplus` | `kda_gate_lower_bound` 定案 -5.0 → 衰减门走 `lz_sigmoid` 那一支 | `forward.c` 的 `if (c->kda_has_gate_lower_bound)`；`kunkun-ce/config.json` 是 `-5.0`（2026-08-21 复核）→ **不跑** |

这些算子的判定见各自条目（A9/A10/A11/A12、B5、C6、D7、E3、F3、G2）。
它们**不是 IMPOSSIBLE**，只是不进 §5 的排序。

---

## 4. 汇总表

| 算子 | 判定 | 一句话理由 |
|---|---|---|
| `q8_round` | 有独立内容 | 魔数加是为躲 x87 控制字而生；ARM 无 FPU，位域舍入 10–14 条替掉 `bl q8_round`+`bl __addsf3` 的 48 条 |
| `q8_amax` | 有独立内容 | `LDM` 一条顶四条 `ldr`；比较已在整数域 |
| `q8_group_scale` | 有独立内容（仅比较部分） | 三次 fcmp 走整数域；那次 fdiv 无独立内容（无除法指令，逐位一致堵死 Newton） |
| `pow2f` | **IMPOSSIBLE** | 7 条 `cmn/blt/cmp/movge/add/lsl/bx`；乘加族无乘法、饱和族点不对、`CLZ` 无零可数、不访存、移位与条件执行已用满 |
| `expof` | **IMPOSSIBLE** | 3 条 `lsr/and/sub`；`UXTB` 是 ARMv6（汇编器已核实拒绝），其余指令族均不适用 |
| `lz_i32f` | 有独立内容 | 二段拆分只为 x87/SSE 一致，ARM 上等价于一次正确舍入转换，`CLZ` 内联 12 条替掉四次库调用的 105 条 |
| `lz_quantize_q8` / `_tok` | 有独立内容 | `inv` 循环不变（解包一次）+ `LDM`/`STM` 打包 + `q8_round` 内联 |
| `lz_quantize_q8_silu` | 有独立内容（CE 路径外） | 末乘 2^-15、`(float)sig[k]` 转换 |
| `lz_quantize_q8_int` / `_tok_int` / `_int64` | 有独立内容（CE 路径外） | 同 `lz_quantize_q8`，外加 int64→float 的 `CLZ` 内联 |
| `lz_gdn_quantize_2p` | 有独立内容（CE 路径外） | `*256` 是 2 的幂、`(float)h` 量程极小 |
| `lz_kv4_quantize` | 有独立内容（CE 路径外） | 15 次浮点比较可整体进整数域并二分 |
| `norm_ss_fixed` | 有独立内容 | `sc` 是 2 的幂、魔数加可内联、`v*v` 累加是 `SMLALBB` 一条；88 条/元素 → ~18 |
| `lz_rmsnorm` 输出环 | 有独立内容 | 三次库调用可内联并共用 `inv` 的解包；无整数化路径（那是档位问题） |
| `lz_rmsnorm_gated` + `_out_fixed` | 有独立内容 | 三处 2 的幂乘法、`SMULBB` 顶掉 `smull`、`q8_round` 内联 |
| `lz_l2norm` | 有独立内容 | 与 `norm_ss_fixed` 同源 |
| `lz_rmsnorm_*_int` 四支 | 有独立内容（CE 路径外） | 同 `_gated_out_fixed` |
| `lz_softmax` | 有独立内容 | 最大值扫描的浮点比较进整数域；末尾 `*inv` 循环不变 |
| `sigmoid_q15` / `_t` | 有独立内容 | 两个乘数是 2^4 与 2^15、两次比较进整数域、`SMULBB` 顶 `mul`、`LDRD` 取相邻表项；250 条 → ~35 |
| `lz_exp_fixed` | 有独立内容 | 四处 2 的幂乘法 + 两次转换 + `q8_round`；泰勒本体无内容 |
| `lz_exp_float_body` | 有独立内容（权重近零） | `0.03125f` 是 2 的幂；其余只有解包复用 |
| `lz_rsqrt_fixed` | 有独立内容（仅解量化尾巴） | 牛顿本体已被 gcc 用满 `UMULL`/`SMLAL`，是 IMPOSSIBLE 的那一半 |
| `lz_rsqrt_float_body` | 有独立内容（权重近零） | `0.5f*x` 是 2 的幂；其余解包复用 |
| `lz_softplus` | **这一格作废（CE 路径外）** | `kda_gate_lower_bound` 定案 -5.0，衰减门走 `lz_sigmoid`；即使在，能动的也只有那条真实分布上从不触发的早退路径 |
| `lz_sigmoid`/`lz_silu`/`lz_exp`/`lz_rsqrt` 派发壳 | **IMPOSSIBLE** | 一次载入、一次比较、一次分支；内容全在被派发的本体 |
| `gdn_build_table` | 有独立内容 | 取绝对值与比较进整数域、`q8_round` 内联、int16 对可 `orr`+`str` 一次写 |
| `gdn_sum_gg`（pass 1） | 有独立内容 | 元素被重复载入、`ldrsb` 可换 `LDM`、相邻系数对可半字选择、累加器可留寄存器（整数重结合精确）；16 条/车道对 → 4–5 |
| pass-1 尾声 | 有独立内容 | 四次 `lz_i32f` + `2^-8` 乘法 + 组内不变的 `su`/`sw` |
| `gdn_p2_group_fixed` | 有独立内容 | `LDM` 取 int8、`SMULBB`/`SMLABB` 顶 32×32、`str` 打包写回；移位与钳位无内容 |
| `p2_group_coef`/`_norm`/`_scale` | 有独立内容 | 五处乘 2 的幂 + 三次比较 + 两次 `q8_round` |
| `gdn_step_impl` 标量尾声 | 有独立内容 | 三个循环不变乘数，且 KDA 的 `gt==1.0f` 让两次乘法可整支删（乘 1.0f 精确） |
| GDN 浮点 pass 1/2 | 有独立内容（CE 路径外） | 同上族 |
| `lz_causal_conv1d_step_fixed` 抽头环 | 有独立内容 | 相邻 int16 打包 + `SMLABB`/`SMLATT` 半字选择；5 条/抽头 → 2 |
| `lz_causal_conv1d_step_fixed` 尾声 | 有独立内容 | 每通道 10 次库调用里 `lz_i32f`(4) 与 `in_scale`/`(float)s15` 全可内联 |
| `lz_causal_conv1d_step`（浮点） | 有独立内容（CE 路径外） | `LDM` + 解包复用 |
| conv 装载期三函数 | 有独立内容（权重零级） | 每模型一次 |
| `lz_attn_score_q8` | 有独立内容 | 内积体与 matmul 行内核同形（`LDM` + 半字选择），末乘 2^-8，`lz_i32f`，int8→int16 拓宽环 |
| `lz_attn_wsum_q8` | 有独立内容 | 9 条/2 MAC → 2.5：`LDM` + 相邻系数对 + 累加器留寄存器 |
| `lz_attn_wsum_q8_int` | 有独立内容（CE 路径外） | 同上 |
| `lz_fwht_i32` | 有独立内容 | `LDM`/`STM` 让 8 条/蝶形降到 3.5，`LDRD` 收尾两阶段，首阶段可与入口 `memcpy` 合并；乘加族与饱和族全不适用（变换体无乘法、要回绕不要饱和） |
| `lz_fwht`（浮点） | 有独立内容（CE 路径外） | 一次解包+对阶服务和与差两个结果 |
| `lz_rope` | 有独立内容 | `c`/`s` 在 h 环上不变，四次乘法只需两次操作数解包 |
| `lz_moe_route` | 有独立内容（CE 路径外） | 排序与 top-k 的浮点比较进整数域 |
| `lz_moe_hits_add` | **IMPOSSIBLE** | `ldrb`/`orr`/`strb`；ARMv5TE 无位置位指令，其余指令族不适用 |

判定分布：**IMPOSSIBLE 4 条**（`pow2f`、`expof`、四个派发壳、`lz_moe_hits_add`），
其余全部「有独立内容」。方向如此一边倒的原因是单一的、可以写下来的：
**CE 目标机没有 FPU**，所以每个浮点运算都是一次 28–100 条的库调用，
而引擎里大量乘数是精确的 2 的幂、大量比较可以在整数域完成、
大量转换可以用 `CLZ` 内联。这不是「什么都值得写」，是「这台机器上
C 那一格与硬件之间隔着一整个软浮点库」。

---

## 5. 排序：按 CE 出货配方的每 token 调用量

**排序依据与它的可靠性。** 不用 `.prof/` 的 f32 普查，也不用
`docs/operator-audit.md` 的那张表：那张表每行单位不同（有的是操作数，
有的是调用次数），而且注意力打分整行缺席。这里用的量是
**从 CE 几何与 forward.c 的循环结构直接数出来的每 token 调用次数**，
以及**从 §0 的 `.s` 里数出来的每次调用指令数**。两个因子都可复核：
前者重数一遍循环，后者重编一次 `.s`。乘积是每 token 静态指令数——
铁律三点名的、唯一能跨机器迁移的量。**不给周期数。**

每 token 计数的推导（`T` = 上下文长度，nt=1 解码）：

```
KDA 层 ×4，每层：
  SubLN 输入 norm      4 × lz_rmsnorm(512)              → 2048 元素
  激活量化             5 × lz_quantize_q8_tok(512)      → 2560 元素
  conv                 3 × step_fixed(n_ch=512, k=4)    → 1536 通道 / 6144 抽头
  衰减门               nv·kd = 512 元素                 → 512 exp + 512 sigmoid
  递推 16 头：
    l2norm             32 × lz_l2norm(32)               → 1024 元素
    pass 1             16 kk 对 × 32 车道 × 4 MAC × 2 平面 × 16 头 = 65,536 MAC
    pass 1 尾声        32 vv × 4 × 16 头 = 2048 次 lz_i32f
    pass 2             32 行 × 32 车道 × 16 头 = 16,384 车道
    build_table        64 次 q8_round × 16 头 = 1024
  o_norm(gated)        16 × lz_rmsnorm_gated(32)        → 512 元素
  o_subn_norm          lz_rmsnorm(512)
  subn_fwht_quant      512·log2(512) = 4608 蝶形 + 1024 次 q8_round

全注意力层 ×2，每层：
  norm                 8×rmsnorm(64) + 2×rmsnorm(64)    → 640 元素
  rope                 (8+2) 头 × 32 频率对 = 320 次旋转
  KV 写                2 × lz_quantize_q8(128)          → 256 元素
  打分                 8 头 × 2 组 × 32 × T = 512·T MAC，16·T 次 lz_i32f
  softmax              8 × lz_softmax(T)                → 8·T 次 exp
  wsum                 8 头 × 2 组 × 32 × T = 512·T MAC，16·T 次 q8_round
  输出门 + o_proj 量化 512 sigmoid + 512 元素量化

dense FFN ×6（每层都有，first_k_dense_replace = n_layers）：
  gate/up norm+量化    2×rmsnorm(512) + 2×q8_tok(512)   → 各 1024 元素
  SwiGLU               2048 × lz_silu                   → 2048 sigmoid
  down_norm            lz_rmsnorm(2048)
  subn_fwht_quant      4 块 × 512·9 = 18,432 蝶形 + 4096 次 q8_round

末端：lz_rmsnorm(512) + lz_quantize_q8(512)
```

`q8_round` 的总计要单独列，因为它散在十处：

```
q8_tok(KDA)  4×5×512 = 10,240      norm_ss_fixed          36,608
q8_tok(FFN)  6×2×512 =  6,144      gated_out_fixed         6,208
quantize_q8（注意力 3,584 + 末端 512 + f_b 128） = 4,224
subn_fwht_quant  4×1024 + 6×4096 = 28,672
gdn_build_table  4×16×64 = 4,096   pass 2  4×16×96 = 6,144
lz_exp_fixed 2,112 + 16·T          wsum 系数量化 32·T
                                   ------------------------------
                                   合计 104,448 + 48·T
```

汇总。两列 `T`，因为**名次本身随上下文变**：`T=0` 是纯常数项，
`T=1536` 是 CE 的实训上下文（`max_position 8192`，实训 1536 + 5120）。

| 算子 / 叶子 | 每 token 次数 | 每次指令 | 来源 | 常数项 | @`T`=1536 |
|---|---|---|---|---|---|
| **`lz_exp_fixed`（C2）**※ | 2,112 + 16·T | **431.5** | **实测** | 0.91 M | **11.52 M** |
| `lz_i32f`（A6）※ | 28,672 + 32·T | **112.8** | **实测** | 3.23 M | **8.78 M** |
| `q8_round`（A1）※ | 104,448 + 48·T | **48.0** | **实测** | 5.01 M | **8.55 M** |
| **注意力打分+wsum 的 MAC（F1/F2）** | 1024·T | 4.5 | `.s` 逐条数 | 0 | **7.08 M** |
| **`lz_sigmoid` / `sigmoid_q15_t`（C1）** | 17,472 + 6,144 `_t` | **322.4** / **201.0** | **实测** | **6.87 M** | 6.87 M |
| `lz_rmsnorm` 的浮点输出环（B2） | 30,464 元素 | ~100 | `.s` 估计 | 3.05 M | 3.05 M |
| `lz_softmax` 的浮点部分（B6） | 16·T 元素 | ~100 | `.s` 估计 | 0 | 2.46 M |
| `lz_causal_conv1d_step_fixed` 尾声（E2） | 6,144 通道 | ~350 | `.s` 估计 | 2.15 M | 2.15 M |
| `norm_ss_fixed` 元素环（B1，除 `q8_round` 外） | 36,608 元素 | ~40 | `.s` 估计 | 1.46 M | 1.46 M |
| `gdn_sum_gg` pass 1（D2） | 65,536 车道对 | 16 | `.s` 逐条数 | 1.05 M | 1.05 M |
| **`lz_fwht_i32`（G1）** | 129,024 蝶形 | 8 | `.s` 逐条数 | **1.03 M** | 1.03 M |
| `gdn_p2_group_fixed`（D4） | 65,536 车道 | ~12 | `.s` 估计 | 0.79 M | 0.79 M |
| 量化元素环的 `__aeabi_fmul`（A7/A8） | 20,608 元素 | 32 | `.s` 估计 | 0.66 M | 0.66 M |
| `lz_rmsnorm_gated_out_fixed` 元素环（B3） | 2,048 元素 | ~180 | `.s` 估计 | 0.37 M | 0.37 M |
| `q8_amax`（A2） | 77,760 元素（norm 36,608 + 量化 20,608 + gated 6,208 + fwht 14,336） | 4 | `.s` 逐条数 | 0.31 M | 0.31 M |
| `lz_rope`（G3） | 640 次旋转 | ~200 | `.s` 估计 | 0.13 M | 0.13 M |
| `lz_l2norm` 输出环（B4） | 4,096 元素 | ~32 | `.s` 估计 | 0.13 M | 0.13 M |
| conv 抽头环（E1） | 24,576 抽头 | 5 | `.s` 逐条数 | 0.12 M | 0.12 M |
| `lz_rsqrt_fixed`（C4） | 251 | **153.1** | **实测** | 0.04 M | 0.04 M |
| conv 装载期三函数（E4） | 每模型 1 次 | — | — | 0 | 0 |

#### `.s` 估计能不能信 —— 六个叶子的标定

六个叶子都用 §0 第 4 条那套装置量过，每个配自己的**真实输入分布**
（`.prof/arm_bias_dist.py` 从 `E:/LLM/models/kunkun-ce` 钩出来：
`sigmoid_q15_t` 取 conv1d 前激活折出的 `t`、`q8_round` 取
`norm_ss_fixed` 的 `x[i]·2^(14−expof(amax))`、`lz_exp` 取衰减门的 `g`、
`lz_rsqrt` 取 `ss/n+eps`。`lz_i32f` 的输入不是钩的，是按引擎自己写下的
三个累加器界取的 2^20..2^30 对数均匀带，那份脚本的注释里说明了）。

| 叶子 | 每次库调用 | `.s` 估计 | **实测** | 误差 |
|---|---|---|---|---|
| `q8_round` | 1 | 48 | **48.0** | **0.0%** |
| `lz_i32f` | 4 | 105 | **112.8** | 估低 6.9% |
| `sigmoid_q15_t` | 6 | 250 | **201.0** | 估**高** 24.4% |
| `lz_sigmoid` | 8 | 250 | **322.4** | 估低 22.5% |
| `lz_rsqrt_fixed` | 9 | 250 | **153.1** | 估**高** 63.3% |
| `lz_exp_fixed` | 12 | 200 | **431.5** | 估低 53.6% |

**没有可用的修正因子，误差也不随库调用密度变。** 六个误差是
0% / −7% / +24% / −23% / +63% / −54% —— **两个方向都有，最大约 ±60%，
且与库调用数不单调**：最大的高估在 9 次调用那行，最大的低估在 12 次
那行，唯一准确的那行只有 1 次。拟不出关系式，**不要凑**。
拟不出的原因是下面三种机制混在一起，只有第二种是方法本身的问题：

1. **直线代码、所有调用都执行 → 逐条加是对的。** `q8_round`（0%）与
   `lz_i32f`（−7%）属于这一类。
2. **分支让调用互斥 → 逐条加会高估。** `lz_rsqrt_fixed` 的 `.s` 里有
   **18 个 `bl __aeabi_fmul` 分布在 7 个基本块**（奇/偶指数、subnormal
   回退），取用路径只跑 2–3 个。+63% 就是这么来的，这是**方法失效**。
3. **那个数根本不是加出来的，是猜的。** `sigmoid_q15_t` 的 250 是拿量到的
   `lz_sigmoid` 减一个猜的差额（真差额 121 条，猜成 ~70）；`lz_exp_fixed`
   的 200 是随手写的 —— 把 `.s` 里 14 处 `bl` 认真加一遍能得到 465，
   与实测 431.5 只差 7%。这两条不是方法失效，是没真去数。

**可操作的划分就是表里「来源」那一列：**

- **`.s` 逐条数**（`gdn_sum_gg` 16、FWHT 8、conv 抽头 5、`q8_amax` 4、
  注意力 MAC 4.5、量化环 32）—— 数的是**没有分支、没有调用的循环体**，
  属机制 1，可以直接信。
- **`~` 开头的**（`lz_rmsnorm` 输出环 ~100、`lz_softmax` ~100、
  conv 尾声 ~350、`gdn_p2` ~12、`gated_out` ~180、`lz_rope` ~200）——
  与刚失手的那两条同类，**可能偏 ±50%，方向不可预测**，
  拿它们排序之前先量。

**还该量哪几行**（按「量错了会不会改动作」排，不按密度）：
`lz_causal_conv1d_step_fixed` 尾声（~350，2.15 M；单次库调用最密**且**
带分支，机制 2 与 3 同时具备）> `lz_rmsnorm` 输出环（~100，3.05 M）>
`lz_softmax` 浮点部分（~100，2.46 M @`T`=1536）。
**不该再花力气的是 `lz_softplus`**：已量到 ±0.02%，且在定案配方下
每 token 跑 0 次。

※ 这三行与别的行**重叠**：`q8_round` 被 `norm_ss_fixed` / `gdn_p2` /
`subn_fwht_quant` / 量化环 / **`lz_exp_fixed`** 内含，`lz_i32f` 被
conv 尾声 / pass-1 尾声 / `subn_fwht_quant` / 注意力内含，而
`lz_exp_fixed` 自己那 431.5 条里就含着一次 `q8_round`（48）与一次
`pow2f`。前两行单列是因为**一份汇编实现同时服务所有调用点**，
在队列里算一个条目而不是四个。**表里的数不能相加。**

**排队结果**（按「写一份能收多少指令」，取 `T`=1536 那一列）：

1. **`lz_exp_fixed`**（C2）。**11.52 M @`T`=1536，全表第一**，其中 **92%
   来自 softmax 的 16·T 项** —— 它随上下文长得比注意力还快
   （每 `T` 6,904 条 对 4,608 条），所以上下文越长它越靠前，
   与注意力永不交叉。C2 判定里能收的是四处乘 2 的幂、两次比较、
   两次转换与那次嵌套的 `q8_round`。
   **它的常数项只有 0.91 M**，旧表按常数项把它排在末段，那是失真。
2. **`q8_round` + `lz_i32f` + 「乘 2 的幂」这三个原语**（A1 + A6 + §2 M1）。
   前两项在表里单列，合计 **17.33 M @`T`=1536**；第三项散在几乎每一行里，
   没有单独的行可数。一份实现同时压掉上面第 1 项与下面每一行的一部分
   （`lz_exp_fixed` 那 431.5 条里就含一次 `q8_round`），所以
   写完它们，后面每个算子的收益都要按修正后的基线重算。
   两者内部谁先谁后不影响动作 —— 都要写。
3. **注意力 `lz_attn_score_q8` / `lz_attn_wsum_q8`**（F1/F2）。7.08 M
   @`T`=1536，**常数项是 0，全部收益都长在上下文上**。
   它在 **`T ≈ 1490`** 越过 `lz_sigmoid` 那一行（实测把后者从 7.14 M
   改到 6.87 M，交叉点因此从 1551 落到 1536 以下），也就是说
   **在 CE 实训长度上它已经在 `lz_sigmoid` 前面**；但它追不上
   `lz_exp_fixed`，后者的 `T` 系数更大。
   唯一的排序保留是它的内积体与 matmul 行内核同形，应等 matmul
   那一格出来后共享（§6），否则会写出两份一样的东西。
4. **`lz_sigmoid` / `sigmoid_q15` / `sigmoid_q15_t`**（C1）。**6.87 M**，
   **最大的常数项行**，两半都是实测（322.4 与 201.0）。
   两个乘数是 2 的幂、两次入口比较可整数化、`(b-a)*frac` 可用
   `SMULBB`、相邻表项可 `LDRD` —— 收得最干净的一条。
   `kda_gate_lower_bound` 定案 -5.0 之后，衰减门那 2,048 次/token
   记在这一行头上。
5. **`lz_rmsnorm` 的浮点输出环**（B2，3.05 M）与
   **`lz_softmax` 的比较扫描**（B6，2.46 M @`T`=1536）。
   **两行的每次成本都还是 `~100` 的估计**，而刚刚的标定说这类数
   可能偏 ±50% —— 排在这里是暂定的，量过再定。
6. **`lz_causal_conv1d_step_fixed` 尾声**（E2，2.15 M 中扣掉已被第 1 项
   收走的 `lz_i32f`）与 **`norm_ss_fixed` 元素环**（B1，1.46 M）。
7. **`gdn_sum_gg`**（D2，1.05 M）、**`lz_fwht_i32`**（G1，1.03 M）、
   **`gdn_p2_group_fixed`**（D4，0.79 M）。三者现在是同一档，
   见下面「与旧排序的差」第 1 条。
8. 量化元素环的 fmul（A7/A8）、`gated_out_fixed`（B3）、`q8_amax`（A2）、
   `lz_rope`（G3）、`lz_l2norm`（B4）、conv 抽头环（E1）、
   `gdn_build_table`（D1）、`p2_group_*`（D5）、`gdn_step_impl` 尾声（D6）。
9. `lz_rsqrt_fixed` 的尾巴（C4，实测 153.1 条/次 × 251 次 = 0.04 M ——
   估计的 250 高了 63%，它比原先以为的更不值得动）、
   两个 float body（C3/C5）。
10. CE 路径外的那批（§3.H）与装载期三函数（E4）。

**与 8 层 / inter 1536 那份几何相比，哪几行换位、为什么**：

| 行 | 旧（8L/1536） | 新（6L/2048） | 比 | 机制 |
|---|---|---|---|---|
| 注意力 MAC | 4.6k·T | 4.6k·T | **1.00** | 全注意力层两种几何都是 2 个 |
| `lz_rope` | 0.13 M | 0.13 M | 1.00 | 同上 |
| `lz_fwht_i32` | 1.11 M | 1.03 M | **0.93** | FFN 那 110,592 蝶形**完全不变**（`n_layers × inter` 守恒），只有 KDA 半边按 4/6 缩 |
| `sigmoid_q15` | 6.55 M | 5.47 M | 0.82 | SwiGLU 的 12,288 次**完全不变**（同守恒），缩的只有 KDA 侧的 conv / 衰减门 / o_norm |
| `lz_rmsnorm` 输出环 | 3.76 M | 3.05 M | 0.81 | FFN 侧 18,432 不变，KDA 侧 ×4/6 |
| `q8_round` | 6.40 M | 5.01 M | 0.78 | 同上，两半混合 |
| `lz_i32f` | 3.87 M | 3.01 M | 0.78 | 同上 |
| `norm_ss_fixed` | 1.87 M | 1.46 M | 0.78 | 同上 |
| conv 尾声 / 抽头 | 3.23 M | 2.15 M | **0.67** | 纯 KDA，4/6 |
| `gdn_sum_gg` | 1.57 M | 1.05 M | **0.67** | 纯 KDA |
| `gdn_p2_group_fixed` | 1.18 M | 0.79 M | **0.67** | 纯 KDA |

（这张表两列都按**当时那份**每次成本算，为的是把几何的影响单独隔出来。
后来的实测把 `sigmoid_q15`、`lz_i32f` 等几行的每次成本改了，
两列同比变化，**比值与这三条换位结论都不受影响**。）

三处真换位：

1. **`lz_fwht_i32` 越过 `gdn_p2_group_fixed`，并追平 `gdn_sum_gg`。**
   旧几何下递推那对是 1.57 / 1.18，FWHT 是 1.11，差距 42%；
   新几何下是 1.05 / 0.79 对 1.03 —— **`gdn_sum_gg` 与 `lz_fwht_i32`
   差 2%**。原因是 FWHT 的大头（`hadamard_down`）挂在守恒的
   `n_layers × inter` 上，而递推纯按 KDA 层数缩。**「先写递推、
   FWHT 排后面」这个旧结论不再成立**，三者是同一批。
2. **注意力的交叉点前移**：注意力那一行两种几何完全相同（4.6k·T），
   而它要越过的最大常数项行缩了，所以交叉点往前走。按**实测**的
   每次 322.4/201.0 算，`lz_sigmoid` 那一行旧几何是 8.28 M、
   新几何 6.87 M，交叉点从 `T ≈ 1796` 移到 **`T ≈ 1490`** ——
   **越过了 CE 的实训上下文 1536**，所以在部署点上注意力已经在
   `lz_sigmoid` 前面。旧排序把 F1/F2 放在第 5 位
   并注了「部署上下文上它排第一」；它现在排第 3，
   而那句话的准确形式是「它在实训长度上已经超过最大的常数项行，
   但超不过 `lz_exp_fixed`」。
3. **`lz_exp_fixed` 从第 12 升到第 1**。它的常数项微不足道（0.91 M），
   但 `16·T` 项在 `T`=1536 上是 10.60 M。旧表把它按常数项排在末段，
   那是旧表**只按常数项排序**造成的失真，不是几何变化造成的——
   这一条是排序方法的修正，不是配方的后果。

**这个排序对哪些假设敏感，各自的方向：**

- **`T`（上下文）**：五行随 `T` 长，其余不随。按实测的每次成本，
  系数是 **`lz_exp_fixed` 6.90k·T**（最大）、注意力 MAC 4.6k·T、
  `lz_i32f` 3.61k·T、`q8_round` 2.30k·T、`lz_softmax` 1.6k·T。
  表给了 `T`=0 与 `T`=1536 两列，换别的部署长度只需重算这五个系数。
  **`lz_exp_fixed` 的系数比注意力大**，所以上下文越长它越拉开 ——
  这一条是这轮实测改掉的最大结论。
- **`use_subn`**：定案与盘上检查点现在都是 `True`，本表按 1 取，所以它不再是
  一处不确定性——留在这里是因为它的杠杆大，关掉会重排整张表。
  为 0 时 `subn_fwht_quant` 不被调用 → `lz_fwht_i32` 掉到 **0**
  （整条从队列消失），KDA 的 4 个投影 norm 与 FFN 的 3 个投影 norm 换成
  每层 2 个 pre-LN → `lz_rmsnorm` 输出环从 30,464 元素降到约 8,700，
  `q8_round` 降约 3 万。**这是本表里最大的单一不确定性**，
  比几何本身的影响大。
- **`kda_gate_lower_bound`**：已定案 **-5.0**，检查点同步，本表按它取。
  与 `use_subn` 同样，它不再是一处不确定性；留在这里是因为杠杆大。
  **关掉（回到 `null`）会怎样**：衰减门那 2,048 次/token 从 `lz_sigmoid`
  换成 `lz_softplus`，两条臂在同一批真实实参上量过（§3 C6）——
  **322 条/次 变 3,441 条/次**，`lz_sigmoid` 那一行从 6.87 M 掉到
  6.21 M，同时多出一行 7.05 M 的 `lz_softplus`，净增
  **6.39 M 指令/token**，并在 `T`=0 上把 `lz_softplus` 顶成全表第一。
  **这一格汇编收不到它**（§3 C6 的判定），所以那 6.39 M 只能靠配方拿，
  而配方已经拿了。

---

## 6. 边界：属于 matmul 任务的格

不在本文档里，登记以免两边都以为对方管：

- `lz_matmul` / `lz_matmul_w` / `lz_matmul_xq` / `lz_matmul_xq_nt`
  与 `ops_matmul.c` 的全部行内核。
- 行内积族：`dot32_*`、`dot128_*`、`lz_dot32_t2_scalar` /
  `lz_dot32_t2_arm` / `lz_dot32_t2_arm_asm`（**ARM 侧唯一已填的汇编格**）。
- 解量化尾声：`epi_q8` / `epi_q41` / `epi_fixed` / `epi_pack_act` /
  `lz_epi_prep` / `lz_epi_mac`（`--epi-write` 档，ARM 上 auto 解析为 fixed）。
- `lz_t_f32` / `lz_t_row_f32` / `lz_act_gs`：权重解量化与组宽决策，
  只被 matmul 与它的调用方使用。

两处必须协调、否则会写出两份同样的东西：

1. `lz_attn_score_q8` 的 `dot32_x16_attn_ref`（F1）与 matmul 的
   int8×int16 行内核是同一个操作序列，`LDM` + `SMLA<x><y>` 半字选择的
   写法应当共用。
2. `--kernel arm-c` / `arm-asm` 这个选择器目前**只**接 T2 matmul 行
   （`ops_matmul.c:750` 是唯一读它的地方）。本文档判定为「该写」的每一格
   都需要挂到同一个选择器上，否则铁律八的覆盖断言（两个构建的并集
   必须恰好覆盖 格式 × 两种）在 ARM 侧仍然只有一格可报。
   `lz_kernel_name()` / `lz_kernel_matrix()` 的 ARM 行要随之扩展成
   按算子报告，而不是按整个构建报告一个值。
