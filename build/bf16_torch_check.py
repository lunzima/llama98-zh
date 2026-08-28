"""Check src/lz_bf16.c's answers against PyTorch's bfloat16.

Run by build/bf16_torch_gate.sh; reads the flat binary
build/bf16_torch_probe.c writes.

WHY PYTORCH. lz_bf16.c already has two checks - its own C core, and the
compiler's __bf16 - but both of those are on this machine and share a
toolchain, so a misreading of the FORMAT rather than of the arithmetic
could survive them both. PyTorch's c10::BFloat16 is a separate
implementation by separate people, and it is what the rest of the world
runs. It widens to f32, computes, and rounds back, which is the same
contract lz_bf16.h states.

NaN IS COMPARED BY CLASS, not by bits. bf16 is not an IEEE format, so
nothing standardises a generated NaN's payload, and implementations
genuinely disagree there - Eigen and Intel did, on record. Everything
else is compared bit for bit.
"""
import struct
import sys

REC = struct.Struct("<6H")
OPS = ("add", "sub", "mul", "div")


def is_nan(u):
    return (u & 0x7F80) == 0x7F80 and (u & 0x7F) != 0


def main():
    if len(sys.argv) < 2:
        print("usage: bf16_torch_check.py cases.bin", file=sys.stderr)
        return 2
    try:
        import torch
    except ImportError:
        print("bf16-torch: SKIP - no torch")
        return 2

    with open(sys.argv[1], "rb") as f:
        raw = f.read()
    n = len(raw) // REC.size
    if n == 0:
        print("bf16-torch: FAIL - the probe wrote no cases")
        return 1

    a = torch.empty(n, dtype=torch.int16)
    b = torch.empty(n, dtype=torch.int16)
    mine = [[0] * n for _ in OPS]
    for i in range(n):
        ai, bi, ad, sb, ml, dv = REC.unpack_from(raw, i * REC.size)
        # int16 view: torch has no uint16, and the bit pattern is what
        # matters - the reinterpret below reads the same 16 bits either way.
        a[i] = ai - 0x10000 if ai > 0x7FFF else ai
        b[i] = bi - 0x10000 if bi > 0x7FFF else bi
        mine[0][i], mine[1][i], mine[2][i], mine[3][i] = ad, sb, ml, dv

    fa = a.view(torch.bfloat16)
    fb = b.view(torch.bfloat16)
    theirs = (fa + fb, fa - fb, fa * fb, fa / fb)

    bad = [0] * len(OPS)
    shown = 0
    for k, name in enumerate(OPS):
        tb = theirs[k].view(torch.int16).tolist()
        for i in range(n):
            t = tb[i] & 0xFFFF
            m = mine[k][i]
            if m == t:
                continue
            if is_nan(m) and is_nan(t):
                continue          # payload is unspecified for bf16
            bad[k] += 1
            if shown < 6:
                shown += 1
                print("  %s %04x,%04x: ours %04x  torch %04x"
                      % (name, a[i].item() & 0xFFFF, b[i].item() & 0xFFFF,
                         m, t))

    fail = 0
    for k, name in enumerate(OPS):
        print("  %-4s %8d / %8d differ" % (name, bad[k], n))
        if bad[k]:
            fail = 1
    if fail:
        print("bf16-torch: FAIL")
        return 1
    print("bf16-torch: PASS (lz_bf16.c agrees with PyTorch's bfloat16 "
          "on every case)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
