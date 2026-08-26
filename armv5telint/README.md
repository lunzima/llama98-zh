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
| `mul-family` | `mul`/`mla`/`smull`/`umull`/`smlal`/`umlal` (2-3× on ARMv5TE vs ARMv7-A) | shift/add for small constants, or re-schedule into the dot-product's paired MACs |
| `ldst-cost` | single `ldr`/`str`/`ldrb`/`ldrh`/`ldrsb`/`ldrsh` (2×) and `ldrd`/`strd` (3×) | group consecutive loads into `ldm`/`stm` |

### `mul-family` ratios (from gcc, not memory)

gcc's ARM cost table, `gcc/config/arm/aarch-cost-tables.h`, maps every
ARMv5TE core (`arm9tdmi`, `arm9e`, `arm10e`, `arm926ej-s`, `arm1026ej-s`,
`xscale`, `strongarm`) to `generic_extra_costs`, and the ARMv7-A baseline
Cortex-A8 to `cortexa8_extra_costs`. The MULT SImode sub-table, in
`COSTS_N_INSNS` units:

| field | mnemonic | generic (ARMv5TE) | cortex-a8 | ratio |
|---|---|---|---|---|
| `simple` | `mul` | 2 | 1 | **2×** |
| `flag_setting` | `muls` | 1 | 1 | equal (not flagged) |
| `extend` | `smull`/`umull` | 2 | 1 | **2×** |
| `add` | `mla` | 3 | 1 | **3×** |
| `extend_add` | `smlal`/`umlal` | 3 | 1 | **3×** |

The previous stand-in ("2-5 cycles, ARM9TDMI TRM") is superseded by this
table: the TRM's data-dependent early-termination multiplier is real, but
gcc's own cost model is the authority the linter keys on, and it says
2-3×, not 2-5×.

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
