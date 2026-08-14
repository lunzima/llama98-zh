# -*- coding: utf-8 -*-
"""从 llama98.def（PE 标准格式）生成两种导出清单——单一事实来源：

1. wlink export 段（默认）：`export <external>=<internal>`（Watcom 内部
   符号带尾下划线，PE 导出名不带——这里做映射）。
2. gcc 版本脚本（--map）：`<name>;` 列表 + `local: *;`。

llama98.def 是唯一事实来源（docs/dll-cli-gui-spec.md §3.8）。
"""
import io
import sys


def _syms(def_path):
    syms = []
    data = set()
    in_exports = False
    for raw in io.open(def_path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.lower().startswith("exports"):
            in_exports = True
            continue
        if in_exports:
            name = line.split()[0]
            if name and name[0].isalnum():
                syms.append(name)
                # "name ; data" marks a DATA symbol (float/int global).
                # Watcom 32-bit OMF names data with a LEADING underscore
                # and functions with a TRAILING one - see the .def file.
                if "data" in line.lower():
                    data.add(name)
    return syms, data


def main(def_path, out_path, map_mode):
    syms, data = _syms(def_path)
    with io.open(out_path, "w", encoding="utf-8") as f:
        if map_mode:
            f.write("/* generated from llama98.def - do not edit */\n")
            f.write("LLAMA98_1.0 {\n  global:\n")
            for s in syms:
                f.write("    %s;\n" % s)
            f.write("  local:\n    *;\n};\n")
        else:
            for s in syms:
                if s in data:
                    f.write("export %s=_%s\n" % (s, s))
                else:
                    f.write("export %s=%s_\n" % (s, s))
    print("%s 符号 %d 个（含数据 %d） -> %s" % (
        "版本脚本" if map_mode else "wlink 导出", len(syms), len(data),
        out_path))


if __name__ == "__main__":
    map_mode = len(sys.argv) > 3 and sys.argv[3] == "--map"
    main(sys.argv[1], sys.argv[2], map_mode)
