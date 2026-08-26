# armv5telint

An ARMv5TE peephole linter. Finds suboptimal instruction sequences in
ARMv5TE assembly, in the spirit of [gaul/armlint](https://github.com/gaul/armlint)
and [CSharperMantle/loonglint](https://github.com/CSharperMantle/loonglint).

The linter targets the one place this project's suboptimal sequences
actually are: the **gcc-compiled C tier**. The hand-written asm cells
(`src/ops_arm.c`'s `lz_dot32_*_arm_asm`) are already optimal, with
per-instruction counts in their comments. gcc cannot match them because
`int16_t` promises only 2-byte alignment and gcc cannot prove the
4-byte contract the asm cells assert (`src/ops_arm.h`).

## Rules

| rule | what it flags | the rewrite (under the 4-byte contract) |
|---|---|---|
| `ldrh-run` | 4+ consecutive `ldrh` (16-bit loads) | one `ldm` (4 words = 8 halfwords), a 4:1 load saving |
| `smlabb-alone` | `smlabb` with no `smlatt`/`smlabt`/`smlatb` | pair BB with TT so one loaded word feeds two accumulates |

**Soundness-first.** Every rewrite is only correct when the flags it
changes are dead. The linter flags candidates; the reader proves. This
is armlint's principle, not a convenience.

## Usage

```bash
# Cross-compile a C tier for ARMv5TE and disassemble:
arm-linux-gnueabi-gcc-14 -march=armv5te -O2 -std=c99 -Isrc -Icommon \
    -c -o /tmp/ops_arm.o src/ops_arm.c
arm-linux-gnueabi-objdump -d /tmp/ops_arm.o > /tmp/ops_arm.dis

# Lint it:
python3 armv5telint.py /tmp/ops_arm.dis
```

## Tests

```bash
python3 test_armv5telint.py
```

The unit tests are fabricated objdump fixtures with a positive control:
a rule that cannot see its own planted pattern is a rule that never
fires.

## Adding a rule

Add a `def <name>_rule()` that returns a `Rule(name, apply)`, register
it in `RULES`, and write a `test_<name>_fires` in the test file. The
`Rule.apply` takes `(func_name, [mnemonics])` and returns
`[(finding, func_name)]`.
