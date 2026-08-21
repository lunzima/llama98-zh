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
hidden 512  layers 8 = LLLALLLA（6 KDA + 2 全注意力）  inter 1536  vocab 6144
KDA        16 头 × (kd=vd=32) → kdim=vdim=512，gate_rank=32，conv k=4，conv_dim=1536
全注意力    8 q 头 / 2 kv 头 / head_dim 64 → qd=512, kvd=128, rotary_dim=64
SubLN 开，hadamard_o=512，hadamard_down=512（1536 % 512 == 0）
```

> 盘上的 `E:/LLM/models/kunkun-ce` 是**先导/消融**检查点，不是这份配方：
> 它是 6 层 / `intermediate_size` 2048 / `use_subn:false` / `kda_gate_lower_bound:null`。
> 前两项只按比例改 §5 的数、不改名次；后两项改结构，各自在
> §3 的 G1（`lz_fwht_i32` 的敏感性一段）与 C6（`lz_softplus`）里单独写了。

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
   —— ARM 上是 `__aeabi_f2d` + libm `exp`（double）+ libm `log1p`（double）
   + `__aeabi_d2f`。`__aeabi_dmul` 155 条、`__adddf3` 171 条，而 libm 的
   `exp`/`log1p` 本身各是几百条 double 运算。
   **它在 CE 上跑不跑取决于一个 config 字段**：`kda_has_gate_lower_bound`
   为假时，KDA 衰减门每个通道调用一次（CE 定案几何下是每层每 token 512 次）；
   为真时改走 `lz_sigmoid`，一次都不调用。盘上的 `kunkun-ce` 检查点是
   `"kda_gate_lower_bound": null`，即**走 softplus**；
   `docs/kunkun-ce-params.md` §1 没有为这个字段定案。
2. **可用指令核对**：把 double libm 换成手写 ARM 汇编不是「填一个 ISA 格」，
   是重写 libm——那超出本条铁律的范围。ARMv5TE 在这个序列上能作用的只有
   `x > 20.0f` 那一次比较（M2）与 f2d/d2f 两次转换（`CLZ` + 移位）。
   double 的乘加本身：`UMULL`/`UMLAL` 正是 `__aeabi_dmul` 已经在用的指令。
3. **判定**：**有独立内容（该写），但内容只有护栏与两次转换**；
   `exp`/`log1p` 本体归 IMPOSSIBLE（重写 libm 不是这一格）。
   **真正的动作是让 CE 配方把 `kda_gate_lower_bound` 定下来**，
   那是配方问题不是内核问题，登记在这里以免它继续没人认领。

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

   即 **4 条指令换 1 次 MAC**。CE 上每 token 393,216 次 MAC（16 头 ×
   16 kk 对 × 32 车道 × 4 MAC × 2 个状态平面 × 6 层）。
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
   12,288 次 `lz_i32f`。
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
   CE 上每 token 98,304 个车道更新（16 头 × 32 行 × 32 车道 × 6 层），
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

1. **操作序列**（每（行，组）一次，CE 上每 token 3,072 次）：
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
   16 头 × 32 vv × 6 层 = 3,072 个 vv。
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
   CE 上每 token 6,144 个抽头（1536 通道 × k=4 × 6 层）。
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
   `x[c]*in_scale` + 比较 + `f2iz`。CE 上每 token 9,216 个通道。
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

   外加入口一次 `bl memcpy`。CE 上每 token **138,240 次蝶形**
   （KDA o_proj：6 层 × 512·log2(512) = 27,648；dense down_proj：
   8 层 × 3 块 × 512·9 = 110,592）。
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
   `subn_fwht_quant` 不被调用，本算子每 token 0 次（盘上的
   `kunkun-ce` 检查点正是 `use_subn:false`）。

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
| `lz_gdn_step` | CE 是 6 KDA + 2 全注意力，0 个 `LZ_LT_LINEAR` 层 | `kunkun-ce-params.md` §1 的 `LLLALLLA`；`forward_chunk` 按 `L->type` 派发，`LZ_LT_KDA` 走 `forward_kda` |
| `lz_moe_route` / `lz_moe_hits_add` | `num_experts=0` + `first_k_dense_replace = n_layers` → 每层 `ffn_moe=0` | `kunkun-ce-params.md` §1；`forward_chunk` 的 `if (L->ffn_moe)` |
| `lz_fwht`（浮点） | `--kv-rot` 默认 off → `s->kv_rot_k = s->kv_rot_v = 0` | `forward.c:891`；`cli_main.c:224` `default OFF` |
| `lz_kv4_quantize` | 只有 `--kv q4` 分配 `k4`/`v4` 平面 | `forward.c:1351/1360` 的 `if (v4)` / `if (k4)` |
| `lz_gdn_quantize_2p` | 两个调用点分别在 pass 2 的**浮点**回退与 `LZ_KV_2PLANE`（默认 0） | `ops_gdn.c:1871`（在 `lz_gdn_p2_mode()` 的 `return` 之后）；`ops.h:1293` |
| `lz_quantize_q8_int` / `_tok_int` / `_int64`、`lz_rmsnorm_int` / `_gated_int` / `_out_fixed_int` / `_gated_out_fixed_int` | `auto` 解析不到 `int`：`lz_norm_int()` 要 `g_norm==LZ_NORM_INT`，`lz_attn_int()` 要 `g_attn==4`，两者的 auto 值分别是 -1 | `ops_sched.c` 的 `lz_norm_mode`/`lz_norm_int`/`lz_attn_int`；`forward.c` 每个调用点的 `if (lz_norm_int() && …)` |
| `lz_attn_wsum_q8_int` | 同上，且额外要求 `gso==32 && !v4 && !vf && !LZ_KV_2PLANE` | `forward.c:1435` 的 `attn_int[tk]` 表达式 |
| `lz_causal_conv1d_step`（浮点）、GDN pass 1/2 的浮点回退 | `lz_conv_mode()` / `lz_gdn_mode()` / `lz_gdn_p2_mode()` 在 `LZ_NO_FPU` 下恒为 fixed | `ops_sched.c:127` 的 `lz_soft_float_target()` |
| `lz_quantize_q8_silu` | 接进 MoE SwiGLU 的改动已回退，forward 里没有调用点 | `grep lz_quantize_q8_silu src/forward.c` 无命中；提交 `58fc52e` |
| `lz_softplus` | **条件性**：`kda_gate_lower_bound` 存在时不跑，为 null 时每通道跑一次 | `forward.c` 的 `if (c->kda_has_gate_lower_bound)`；盘上 `kunkun-ce/config.json` 是 `null` → **会跑** |

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
| `lz_softplus` | 有独立内容（仅护栏与两次转换） | `exp`/`log1p` 本体是 libm，重写 libm 不是这一格 |
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
KDA 层 ×6，每层：
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

dense FFN ×8，每层：
  gate/up norm+量化    2×rmsnorm(512) + 2×q8_tok(512)   → 各 1024 元素
  SwiGLU               1536 × lz_silu                   → 1536 sigmoid
  down_norm            lz_rmsnorm(1536)
  subn_fwht_quant      3 块 × 512·9 = 13,824 蝶形 + 3072 次 q8_round

末端：lz_rmsnorm(512) + lz_quantize_q8(512)
```

汇总（常数项 + `T` 项）：

| 排名 | 算子 / 叶子 | 每 token 次数 | 每次指令数（`.s`） | 每 token 指令数 |
|---|---|---|---|---|
| 1 | `sigmoid_q15` / `_t`（C1） | 28,768 | 250 / 180 | ≈ 6.5 M |
| 2 | `q8_round`（A1，被下面几乎所有条目内含） | 133,248 + 48·T | 48 | ≈ 6.4 M + 2.3k·T |
| 3 | `lz_rmsnorm` 的浮点输出环（B2） | 37,632 元素 | ~100 | ≈ 3.8 M |
| 4 | `lz_causal_conv1d_step_fixed` 尾声（E2） | 9,216 通道 | ~350 | ≈ 3.2 M |
| 5 | `norm_ss_fixed` 的元素环（B1，`q8_round` 已计入第 2 行，此处只算其余） | 46,848 元素 | ~40 | ≈ 1.9 M |
| 5b | `lz_rmsnorm_gated_out_fixed` 的元素环（B3，`q8_round`/`sigmoid_q15` 已分别计入第 2、1 行） | 3,072 元素 | ~180 | ≈ 0.55 M |
| 5c | `lz_l2norm` 的输出环（B4） | 6,144 元素 | ~32 | ≈ 0.2 M |
| 6 | `gdn_sum_gg` pass 1（D2） | 98,304 车道对 | 16 | ≈ 1.6 M |
| 7 | `gdn_p2_group_fixed`（D4） | 98,304 车道 | ~12 | ≈ 1.2 M |
| 8 | `lz_fwht_i32`（G1） | 138,240 蝶形 | 8 | ≈ 1.1 M |
| 9 | 量化元素环的 `__aeabi_fmul`（A7/A8，`q8_round` 已计入第 2 行） | 27,648 元素 | 32 | ≈ 0.9 M |
| 10 | `lz_i32f`（A6，散落在 4/6/F1/F2 里） | 36,864 + 32·T | 105 | ≈ 3.9 M + 3.4k·T ※ |
| 11 | **注意力打分 + wsum 的 MAC（F1/F2）** | 1024·T | 4.5 | **4.6k·T** |
| 12 | `lz_exp_fixed`（C2） | 3,168 + 16·T | ~200 | ≈ 0.6 M + 3.2k·T |
| 13 | `lz_softmax` 的浮点部分（B6） | 16·T 元素 | ~100 | 1.6k·T |
| 14 | conv 抽头环（E1） | 6,144 抽头 | 5 | ≈ 0.03 M |
| 15 | `lz_rope`（G3） | 640 次旋转 | ~200 | ≈ 0.13 M |
| 16 | `lz_rsqrt_fixed`（C4） | 363 | ~250 | ≈ 0.09 M |
| 17 | `q8_amax`（A2） | ≈100k 元素（norm 46,848 + 量化 27,648 + gated 9,312 + fwht 15,360） | 4 | ≈ 0.4 M |
| — | conv 装载期三函数（E4） | 每模型 1 次 | — | 0 |

※ 第 10 行与第 4、6、11 行有重叠（`lz_i32f` 的调用点就在那几个算子里）。
它单列，是因为**一份汇编实现同时服务所有调用点**，所以它在队列里是
一个条目而不是四个。同理，第 2 行的 `q8_round` 被第 3、5、7、8 行内含。

**排队结果（去掉重叠、按「写一份能收多少指令」排）：**

1. **`q8_round` + `lz_i32f` + `pow2f` 乘法这三个叶子**（A1 + A6 + M1 的
   通用实现）。它们不是算子而是被到处内含的原语，一份实现同时压掉
   表里第 2、3、4、5、10、12 行的很大一块。先写它们，后面每个算子的
   收益都要按修正后的基线重算。
2. **`sigmoid_q15` / `sigmoid_q15_t`**（C1）。单点最大，且两个乘数是
   2 的幂、两次比较可整数化，收得最干净。
3. **`gdn_sum_gg`**（D2）与 **`gdn_p2_group_fixed`**（D4）。两个递推内核，
   合计 2.8 M，且是纯整数、逐位一致最容易证。
4. **`lz_fwht_i32`**（G1）。1.1 M，`LDM`/`STM` 是教科书用法，
   而且它已经有一份可对拍的 C 参考。
5. **注意力 `lz_attn_score_q8` / `lz_attn_wsum_q8`**（F1/F2）。
   常数项小但**随上下文线性增长**：`T = 512` 时 2.4 M，
   `T = 2048` 时 9.4 M——在部署上下文上它排到第一。
   排在这里而不是更前，是因为它的内积体与 matmul 行内核同形，
   应该等 matmul 那一格的实现出来后共享（§6）。
6. **`lz_rmsnorm` 的浮点输出环**（B2）与 **`lz_l2norm`**（B4）。
7. **`lz_causal_conv1d_step_fixed` 的抽头环**（E1）与尾声（E2 的
   剩余部分——它的大头已经被第 1 项的 `lz_i32f` 收掉）。
8. **`lz_rope`**（G3）、**`lz_softmax` 的比较扫描**（B6）、
   **`q8_amax`**（A2）、**`gdn_build_table`**（D1）、
   **`p2_group_*`**（D5）、**`gdn_step_impl` 尾声**（D6）。
9. **`lz_exp_fixed`**（C2）、**`lz_rsqrt_fixed` 的尾巴**（C4）、
   两个 float body（C3/C5）。
10. CE 路径外的那批（§3.H）与装载期三函数（E4）。

**这个排序对哪些假设敏感，各自的方向：**

- `use_subn`：为 0 时第 4 项（`lz_fwht_i32`）掉到 0，且 KDA/FFN 的
  SubLN norm 少掉 6 个 → 第 1、3、6 项的常数项按比例缩。
- `kda_gate_lower_bound`：为 null 时每 token 多 3,072 次 `lz_softplus`
  （double libm），会挤进前三——但正确动作是把配方定下来，不是写内核。
- `T`：第 5 项随 `T` 线性增长，其余不随。表里所有 `T` 项都单列了，
  换一个部署上下文重排只需要改一个数。
- 层数 / `intermediate_size`：盘上先导检查点是 6 层 / 2048，定案是
  8 层 / 1536。这只按比例改常数项，**不改名次**（逐项算过：
  FFN 相关行 ×(8·1536)/(6·2048) = 1.0，KDA 相关行 ×6/4 = 1.5）。

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
