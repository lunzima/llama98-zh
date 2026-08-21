# ARM 三值内核档位审计：手写汇编格是否兑现了铁律八

审计对象：`feat/attn-conv-fixed-tiers` 分支，`3f36224`（手写 ARMv5TE 三值
内核）、`309e710`（非 matmul 算子手写汇编格核对）、`4d8e1c9`（内核覆盖矩阵
变成闸门）。审计时 HEAD = `4d8e1c9d59a74bc547a1edbd922d7ebb99b3ae5e`。

`parity_gate.sh` 自己的头注释写着「ARM 三档互相欠的逐位一致，靠三次单独
跑 `--kernel ref/arm-c/arm-asm` 来查，不是这个脚本」——**这句话在写下它的
同一个提交（`3f36224`）里已经不成立**：那次提交同时给脚本加了一段循环，
在跨目标对拍之后用同一个 ARM 二进制的 `ref` 转储分别核对 `arm-c` 与
`arm-asm`。下面第 2 节独立验证了这段循环，第 4 节额外做了脚本没做的事——
证明它真的会红。

## 1. 拿到一个真正带 T2 权重的模型

`/tmp/kmr20` 与 `kunkun98-pilot-20260806` 都不含 T2 张量（前者是 WSL
`/tmp`，本次会话里已不存在——tmpfs 不跨会话持久；后者是稠密 GDN 架构，
没有可以标 T2 的注意力头）。磁盘上没有现成 T2 模型，必须现导出。

**导出过程中踩到一个与本次审计无关的既有缺陷，如实记录、未修复**：
`tools/export_q8.py`（训练树 `E:\LLM\llama98-zh`）对 `kunmoe-v2` 的**任何**
导出——不带 `--recipe`、不带 `--allow-lossy-t2`，纯默认调用也一样——都在
`expect()` 阶段抛
`KeyError: 'model.layers.0.linear_attn.q_proj.norm.weight'`。根因：
`kunmoe-v2/config.json` 写着 `"use_subn": true`，导出器据此要求五个
SubLN 输入norm张量（`linear_attn.{q,k,v,g,o}_proj.norm.weight`），但
`model.safetensors`（与 config.json 同一时刻 08-20 21:33 写入）里**没有**
这些张量。**这不影响引擎侧**——`llama98 kunmoe-v2 --check`（直接读
safetensors，不经导出器）正常给出 `Structure OK`，说明 C 侧加载器对这份
checkpoint 没有同样的假设。缺陷严格在训练树的 Python 导出器里，与本次
审计的 ARM 内核无关，未做任何修改。

为了拿到 T2 文件，在一份**私有的、未提交的**导出源目录里把 `use_subn`
临时置 `false`（`model.safetensors` 用软链接指向原文件，不复制 533 MB）：

```
/tmp/kmv2_export_src/config.json         # kunmoe-v2/config.json 的副本，use_subn: true -> false
/tmp/kmv2_export_src/model.safetensors -> /mnt/e/LLM/models/kunmoe-v2/model.safetensors
```

这个改动只是让导出器的期望匹配 checkpoint 里**实际存在**的张量——SubLN
张量确实不存在，禁用对它们的期望不是伪造数据。据此导出：

```
python3 tools/export_q8.py /tmp/kmv2_export_src \
    /mnt/e/LLM/models/kunmoe-v2-t2-probe/model.bin \
    --recipe "self_attn.q_proj=t2:128,self_attn.k_proj=t2:128,self_attn.v_proj=t2:128,self_attn.o_proj=t2:128" \
    --allow-lossy-t2
```

```
WARNING: layers.3.self_attn.{q,k,v,o}_proj.weight is 0.5000 off the ternary grid (tolerance 0.0010), exported anyway
WARNING: layers.7.self_attn.{q,k,v,o}_proj.weight is 0.5000 off the ternary grid (tolerance 0.0010), exported anyway
export done: .../model.bin (bin v4)
  tensors: F32 65 / Q8_0 481 / Q4_1 0 / Q6_1 0 / Q16_0 0 / T2 8
  params: 133.4M, size: 147.3 MB (3.62x compression, 8.830 bit/param)
```

`kunmoe-v2` 只有两层 `full_attention`（第 3、7 层，`LLLALLLA`），`self_attn`
只在这两层存在，四个投影各一份 = 8 个 T2 张量。文本质量因此很差
（0.5 grid 偏差远超 0.001 容忍度，权重根本不是 QAT 出来的三值），这条闸门
问的是加载器与内核，符合 `--allow-lossy-t2` 自己文档写的用途。`config.json`
/`tokenizer.json`/`tokenizer_config.json`/`generation_config.json` 从
`kunmoe-v2` 原样复制到 `kunmoe-v2-t2-probe/`，模型目录可独立加载。

这份探针模型留在磁盘上（`E:\LLM\models\kunmoe-v2-t2-probe`），供以后复用。

## 2. 落地证据：T2 权重确实存在、确实到达内核选择

三条独立信号，不依赖单一工具：

1. **导出器自己的清点**：`tensors: ... T2 8`（见上）。
2. **加载器接受它**：`llama98 kunmoe-v2-t2-probe --check` 打出
   `Structure OK   833 ms`，无报错——`src/model.c` 对 `LZ_FMT_T2` 的分支
   （`dtype == 0` 时才是 f32，且它必须认识 `5` 这个 dtype 才能不报错地
   往下走）被走通。
3. **`--stats` 反着证实位置**：`llama98 ... --stats --layer 3` 对
   `self_attn.{q,k,v,o}_proj.weight` 打印 `(not loaded)`，同层的
   `q_norm`/`k_norm`（f32）正常打印 `n=... rms=... absmax=...`。
   **这本身是一处报告盲区，顺带记下**：`cli_main.c` 的 `tensor_stats_t`
   只认 `dtype==0`（f32）和 `dtype==1`（Q8_0），Q4_1/Q6_1/Q16_0/T2 一律
   落到 `else` 分支打印 `(not loaded)`——它不区分"没读到"和"格式我不
   认识"。但恰好因为这样，`(not loaded)` 精确圈出了 recipe 点名的那
   4×2=8 个张量，间接确认它们不是 f32/Q8_0，与第 1 条互证。

三条合起来：T2 权重真实存在于文件里，加载器真的解析并归类了它们，位置
与导出时指定的完全对应。`--check` 本身不单独打印"T2 张量数"这一行——
这是本仓库现有工具的一个空缺，第 9 节记录。

## 3. 交叉编译与工具链

`arm-linux-gnueabi-gcc-14` + `qemu-arm-static`（WSL，`/usr/bin`）。
`build/arm/build_engine.sh` 与 `run_qemu.sh` 未改动。x86 侧对照二进制是
`make native` 在 WSL 里编出的原生 Linux ELF（`build/native/` 里有一批
Windows 侧遗留的 `.pdata`/`__ImageBase` 目标文件，与 Linux `ld` 混链会报
`dangerous relocation`——先 `rm -rf build/native build/gui` 再重建即可，
不是代码缺陷，是两个工具链共用同一 `build/` 目录又不受 Make 依赖跟踪的
老问题（[[kernel-header-not-in-make-deps]] 的同型）。

## 4. 三向 ARM 档位逐位一致（ref / arm-c / arm-asm）

`build/arm/run_qemu.sh`，`--pin-tiers` 给出的全部 pin，`--kernel` 分别
指定 `ref`/`arm-c`/`arm-asm`，10 token，探针模型：

```
kernel  exit  dump size  md5
ref     0     131072     a9792bb0abacb04cff778795770921a4
arm-c   0     131072     a9792bb0abacb04cff778795770921a4
arm-asm 0     131072     a9792bb0abacb04cff778795770921a4
```

三个 `cmp` 两两 `IDENTICAL`，md5 三者相同。三次日志里都没有出现
`[kernel: requested X, running Y]`——`--kernel` 请求的档位都被真实采纳，
没有静默 clamp 回 ref。

## 5. x86 ↔ ARM parity_gate.sh，两个模型分别跑

**T2 探针模型**（`kunmoe-v2-t2-probe`）：

```
-- x86 native: /mnt/e/llama98-zh/llama98 --
-- ARM under QEMU (arm926): /tmp/armbuild/llama98-arm --
PASS: x86 and ARM --dump-logits are bit-identical (131072 bytes)
PASS: ARM tier arm-c is bit-identical to ref (131072 bytes)
PASS: ARM tier arm-asm is bit-identical to ref (131072 bytes)
EXIT=0
```

**对照：`kunkun98-pilot-20260806`**（稠密 GDN，零 T2 张量；`/tmp/kmr20`
本会话里已不存在，用它替代承担"不带 T2"的对照角色）：

```
PASS: x86 and ARM --dump-logits are bit-identical (131072 bytes)
PASS: ARM tier arm-c is bit-identical to ref (131072 bytes)
PASS: ARM tier arm-asm is bit-identical to ref (131072 bytes)
EXIT=0
```

**两次 PASS 分量不同，必须分开读**：pilot 模型上的三行 PASS 不能说明
`arm-asm` 对——`lz_matmul_xq` 把每个非 T2 格式都送去标量参考（`3f36224`
提交信息原话），这个模型没有任何 T2 张量，三档从头到尾执行的是同一份
标量代码，PASS 只证明"三条路径在没有三值张量时退化成同一条路径"，这正是
交接消息点名的假绿陷阱。**T2 探针模型上的 PASS 才是有信息量的那一次**
——它是唯一真正走过 `matmul_t2_impl` 的一次。

## 6. `arm-asm` 单独对 `ref`，不只是对 `arm-c`

铁律二要求手写汇编对标量参考验证，不是对另一份手写实现互证。第 4 节的
三向比较已经把这一条单独列出（`arm-asm` vs `ref` 的 md5 相同独立于
`arm-c` vs `ref`），`parity_gate.sh` 自己的循环同理——两行 PASS
（`arm-c is bit-identical to ref`、`arm-asm is bit-identical to ref`）
共享同一个 `$ARM_DUMP`（用 `--kernel ref` 生成）作比较基准，不是互相比较。

## 7. 变异测试：这条闸门真的会红吗

在 `git archive HEAD | tar -x` 出的一次性目录（`/tmp/mutant_tree`，**不是
真实工作树**）里，改了 `src/ops_t2_arm.c` 里 `lz_dot32_t2_arm_asm` 的一
处立即数：

```diff
-        "mov    r10, #3\n\t"
+        "mov    r10, #1\n\t"
```

`r10` 是逐 2 位提取三值编码用的掩码（`0x00030003`，来自 `#3` 再
`orr r10,r10,r10,lsl #16` 广播），这一改把掩码变成 `0x00010001`——每个
三值码的高位被吃掉，计算结果错但不崩溃（不是空指针/越界那类会在断言
之前就让程序死掉的变异，见 [[mutation-red-is-not-that-assertion-red]]）。

用这份被改过的源码重新跑 `build/arm/build_engine.sh`（`arm-c` 完全不受
影响，因为它是另一份独立的 C 源码，只有 `arm-asm` 被动了）：

```
== mutated arm-asm ==
cmp arm.ref.logits arm.arm-asm.MUTANT.logits
  differ: byte 1, line 1 (125339 of 131072 bytes differ, 95.6%)

== mutated arm-c (control, unmutated source) ==
cmp arm.ref.logits arm.arm-c.MUTANT.logits
  IDENTICAL
```

对照组（同一次被"污染"的构建里的 `arm-c`）仍然和 `ref` 逐位相同，说明
分歧确实只来自被改的那个函数，不是构建环境本身出了问题。

**再用真实的 `build/arm/parity_gate.sh`（同一份变异树里的那一份，不是我
手搭的比较）复核一遍**（`git archive` 生成的 tar 没保留 `build/arm/*.sh`
的可执行位，运行前需要 `chmod +x`——操作性的坑，记一笔，不是仓库缺陷）：

```
PASS: x86 and ARM --dump-logits are bit-identical (131072 bytes)
PASS: ARM tier arm-c is bit-identical to ref (131072 bytes)
FAIL: ARM tier arm-asm differs from ref - 125339 of 131072 bytes
EXIT=1
```

差异字节数（125339）与手工比较完全一致，脚本本身的判据是敏感的——它
不是一条从没红过的闸门。

## 8. `missing-facilities.md` H4 是否过期

**是，已过期，需要更新**（本次审计不改文档正文，只在这里记录裁定所需的
证据；是否回填由后续会话处理）。H4 写于 2026-08-18，早于 `3f36224`
（2026-08-20），原文：

> 仍未做：ARM 侧两格里汇编那格是空的，原因记作「还没写」而不是「用不上」

`lz_dot32_t2_arm_asm` 现在存在、可通过 `--kernel arm-asm` 选中、被
`lz_kernel_name`/`lz_kernel_tier` 报告、并且如第 4-6 节所证，对标量参考
逐位一致——"还没写"这句话不再是事实。

`--kernel-matrix`（`4d8e1c9` 新加的闸门，比 H4 当时描述的"两格"模型更细，
是 5 格式 × 2 ARM 档的矩阵）已经反映了这个变化：

```
cell t2    armC  o --A row -
cell t2    armA  o --A row -
```

`o`（其他构建里覆盖）而非 `t`（待办）。但**范围是有限定的**——同一份
`--kernel-matrix` 输出里，`q8_0`/`q4_1`/`q6_1`/`q16_0` 在 `armC`/`armA`
两列全部仍是 `t`，理由写在 row 里：

> not written: the ARM build has one row kernel, ternary, and
> lz_matmul_xq_nt sends every other dtype to matmul_scalar_ref

这是按设计的范围限定（`3f36224` 提交信息原话），不是遗漏——ARM 手写汇编
格目前**只**覆盖 T2 这一种权重格式，其余四种在 ARM 上永远走标量参考。
H4 回填时这一点要写清楚，不能只说"汇编格已补"就完事。

## 9. 发现但未修复的问题（如实报告，未改动）

1. **`tools/export_q8.py` 对 `kunmoe-v2` 的默认导出已损坏**（第 1 节）：
   `use_subn: true` 与实际张量集不一致，任何 recipe（含不带 recipe 的
   默认调用）都会在 `expect()` 处 `KeyError`。不影响引擎 C 侧加载器
   （`llama98 kunmoe-v2 --check` 正常）。`build/ci.sh` 的
   `CHECK_MODELS` 列表里点名了 `kunmoe-v2`，但它跑的是直接
   `--check`（C 加载器路径），不经过导出器，所以目前不受这个缺陷影响
   ——影响面仅限"需要先导出 `.bin` 再跑"的场景。
2. **`cli_main.c` 的 `tensor_stats_t` 认不出 Q4_1/Q6_1/Q16_0/T2**：全部
   打印 `(not loaded)`，无法区分"这个格式我没实现统计"与"这个张量真的
   没加载成功"。本次审计靠"就是 recipe 点名的那几个张量"这个旁证绕过去
   了，但如果以后要一条闸门直接断言"T2 张量数 = N"，现有 `--check`/
   `--tensors`/`--stats` 都不提供这个数字——`--tensors` 只在 safetensors
   源上工作（`m.st.n_tensors`），从 `model.bin` 加载时该计数为 0，
   连文件名列表都不会打印，尽管 `--check` 本身完全正常。
3. **`git archive HEAD | tar -x` 不保留 `.sh` 的可执行位**：一次性目录里
   跑 `build/arm/*.sh` 之前要手动 `chmod +x`，否则第一步就是
   `Permission denied`。操作细节，非缺陷。

## 10. 结论

**本次审计证实的**：`arm-asm`（`lz_dot32_t2_arm_asm`）在一个真实带 8 个
T2 张量的模型上，对标量参考 `ref` 逐位一致；与 `arm-c` 也逐位一致；三者
互证不依赖脚本自身（本文档独立于 `parity_gate.sh` 重新跑了一遍三向比较）。
这条闸门不是摆设——变异测试证明它在真实分歧存在时会报 `FAIL` 并给出
差异字节数，而不是一条从未红过的绿灯。`missing-facilities.md` H4"汇编格
还没写"的判断需要更新为"已写，且仅覆盖 T2 一种格式，其余四种格式在 ARM
上没有独立汇编实现（by design，`--kernel-matrix` 已经登记了理由）"。

**本次审计没有覆盖、仍然缺证据的**：
- 真实 AK7802 硬件——全部验证都在 QEMU `-cpu arm926` 用户态模拟下完成，
  `run_qemu.sh` 自己的头注释说得很清楚：这给出正确性，不给出周期数；
- `--kernel-matrix` 标记为 `t`（TODO）的四个格式 × 两个 ARM 档
  （q8_0/q4_1/q6_1/q16_0 的 armC/armA）——按设计范围之外，未评估是否
  应该扩大范围；
- Watcom/DOS 目标上的 T2——ARM 之外的目标机没有出现在这次审计里；
- 探针模型的质量——`--allow-lossy-t2` 的警告已经说明这份 checkpoint
  不是 QAT 出来的真三值权重，本次只问加载器与内核对不对，不问困惑度。
