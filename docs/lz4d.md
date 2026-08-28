# lz4d - model.bin.lz4 流式解压（整活）

引擎可以加载 LZ4 压缩的模型权重：目录里有 `model.bin.lz4` 时，`lz_open_bin`
优先用它，通过同一个读取器把解压后的字节流喂给原有的逐字段加载路径。没有
`model.bin.lz4` 时行为与之前逐位相同。

## 文件

- `src/lz4d.c` / `src/lz4d.h` - LZ4 frame 解压器 + 流式读取器（~300 行纯 C）
- `make_lz4.py` - 开发机压缩工具：`model.bin` → `model.bin.lz4`
- `src/model.c` - 读取器抽象：`LZModel.rd`/`rd_ctx`，`file_reader`（普通文件）
  与 `lz4d_read_void`（LZ4 流）两个实现。所有 bin 路径读取（config 头、group
  size、张量平面）都走 `m->rd`，所以是否压缩对加载路径不可见。

## 用法

```sh
python make_lz4.py MODEL/model.bin      # 生成 MODEL/model.bin.lz4
llama98 MODEL                            # 引擎优先读 model.bin.lz4
```

**识别靠 magic 不靠扩展名**（2026-08-28）：加载器读 `model.bin` 前 4 字节，
`04 22 4d 18`（LE）即 LZ4 frame——**无论文件叫什么**，是压缩的就自动解压。
所以压缩结果直接命名为 `model.bin` 也能加载（`make_lz4.py` 支持：
`python make_lz4.py MODEL/model.bin MODEL/model.bin` 原地覆盖）。旧
`model.bin.lz4` 在 `model.bin` 缺失时仍兼容。

## 格式（标准 LZ4 frame，lz4d.c 接受的子集）

- magic `04 22 4d 18`，FLG 要求 version 01 + block independence + 可选
  content size；**不接受 block checksum 与 linked blocks**。
- **block header 4 字节 LE：bit31 = 未压缩标志**（`LZ4F_BLOCKUNCOMPRESSED_FLAG`，
  该位被置时内容为原始字节），bits 0-30 = 块大小。**没有 last-block 标志**：
  帧在 content size 到达时结束。
- **block 大小 MCU 级（64 KiB）**：解压缓冲固定 64 KiB，无论模型多大，流式
  解压只多占 ~64 KiB 内存。压缩侧必须保证每块 ≤ 64 KiB（`make_lz4.py` 保证）。
  更大的块是硬错误而非静默截断。
- 解压器逐位对照 lz4 官方源码（`lz4.c` 的 `LZ4_decompress_generic` /
  `lz4frame.c` 的 `LZ4F_decompress`）。

## 平台

- 纯字节级访问，无端序假设，无 `long long`（用 `lz_int.h` 的 `lz_i64`），
  Watcom / VC++ 4.0（MIPS 大端）/ dos32a 均兼容。
- **已实测三工具链**（2026-08-28）：
  - 解压后与原始 `model.bin` 逐位相同：45,816,469 字节，`cmp` 无差异
    （2026-08-28 复测，`_armgate2` 的 model.bin，700 x 64KiB block）。
    此处原记「103MB」，而下文「如实说明」记「108MB」——同一件事两个数，
    至少一个是错的；现在写实测到的那个。
  - `make native`、`make watcom-cli`（`llama98-w.exe`）、`make watcom-dos`
    （`llama98-dos.exe`，dos32a）三目标均编译通过（含 lz4d.c 0 errors）。
  - Watcom 版从 `model.bin.lz4` 加载与它自己从未压缩 `model.bin` 加载
    **逐位相同**（chk 1536798181578518320，两次一致）。这就是 lz4 该证明的
    全部：解压出的字节流与原文件无别。

    **这里原本写的是「与 gcc 版逐字相同……跨编译器 + lz4 双变量的逐位一致」，
    那是错的**（2026-08-28 复测）。gcc 版同一模型是 1708445672595413405，
    两者从来就不相等，而且与 lz4 无关——它是 x87 与 SSE 的既有差异，本树
    别处已记为已知可接受项。压缩与否各自对照才是有效实验：变量只有一个。
  - VC++ 4.0（MIPS）无编译器，代码层面验证（字节访问 / `lz_i64`）。

## 如实说明

`model.bin` 是量化权重，熵高、重复少，LZ4 压缩率只有 ~3%
（实测 45,816,469 → 44,578,396 字节，0.973）。
这个特性是整活（在 dos32a/Watcom/VC++ 4.0 上实现 LZ4 解压、真机加载压缩模型），
不是实用优化。约束：block checksum / linked blocks 的帧会拒绝；block > 64 KiB
会报错（提示用 `make_lz4.py` 重新压缩）。
