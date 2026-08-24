# The recur and epi rows, determined from the code

`km_reg` in `src/ops.c` has 15 units. The two largest float census sites
have no row at all: RECUR at 26.7% of per-token float and EPI at 10.0%.
This file is the cell-by-cell determination so the edit is mechanical
rather than another round of reading.

Prerequisite already answered: `.prof/km_initializer_limit_probe.c`
(commit e9bacdd) shows Open Watcom takes 17 x 10 in both the flat and
nested shapes, with a 130-row control proving the green is not blindness.
The initializer count does not block this.

## The headline: these are two different jobs

**recur is a registration problem.** Three x86 tiers of hand-written
kernels already ship and no cell reports any of them, so every gate that
reads the matrix is blind to work that is done.

**epi is six owed kernels.** Nothing but C exists, and the ISAs can all
express what it does.

## recur - where the code is

| symbol | ISA | file |
|---|---|---|
| `lz_gdn_sum_gg_mmx` | MMX | `ops_mmx.c` |
| `lz_gdn1_x4_asm` | MMX, Watcom `#pragma aux` | `ops_mmx.c` |
| `lz_p2_rows_mmx` | MMX | reached from `ops_gdn.c:1331` |
| `lz_p2_rows_sse` | SSE1 | `ops_mmx_sse.c`, `ops_gdn.c:1331` |
| `lz_p2_rows_sse2` | SSE2 | `ops_gdn.c:1388` |

Gated by `LZ_GDN_HAVE_MMX`, which `ops_gdn.c:672` sets from
`__WATCOMC__ || LZ_GDN1_MMX_EXTERN`. `ops_mmx.h:40` notes
`LZ_GDN1_MMX_EXTERN` is unconditional on both compilers, unlike the four
dot externs.

Row:

| col | value | evidence |
|---|---|---|
| ref | `KM_H(KM_ALL)` | shared C body, as every row |
| mmxI | `KM_H(LZ_KMB_GCC)` | `lz_p2_rows_mmx` via `LZ_GDN1_MMX_EXTERN` |
| sseI | `KM_H(LZ_KMB_GCC)` | `lz_p2_rows_sse` - see warning below |
| sse2I | `KM_H(LZ_KMB_GCC)` | `lz_p2_rows_sse2` |
| mmxA | `KM_H(LZ_KMB_WAT)` | `lz_gdn1_x4_asm`, `lz_gdn_sum_gg_mmx` |
| sseA | CONFIRM | is `ops_mmx_sse.c` in the Watcom TU set? |
| sse2A | CONFIRM | same question for the SSE2 body |
| armC | `KM_H(LZ_KMB_ARM)` | the shared C body runs there |
| armA | `KM_T` | no gdn/p2_rows symbol in `ops_arm.c/.h`; ARMv5TE can do this, so owed, not impossible |
| g128 | `KM_N` | zero `g128` references in `ops_gdn.c` - different axis, same as the other operator rows |

**Do not paste `KM_I(km_why_sse1)` into recur's sseI cell.** That string
fills the SSE1 column for every weight-format row and the reflex is to
carry it down. `lz_p2_rows_sse` exists, so that entry would be false -
and false-absent is worse than empty, because the gate would then be
enforcing the lie.

## epi - where the code is

`lz_epi_mac_grp`, `lz_epi_mac_i16`, `lz_epi_prep`, `lz_epi_prep_zero`,
`lz_epi_post_e` in `ops.c`; `epi_q41_fixed`, `epi_q41_join`,
`epi_q41_align_i16`, `epi_q41_row` in `ops.c` / `ops_matmul.c`.

No `_mmx`, `_sse`, `_sse2`, `_asm` or ARM variant exists anywhere.

`lz_epi_mac_i16` is an int16 multiply-accumulate. MMX (`pmaddwd`), SSE1
and SSE2 all express that directly, so the empty x86 cells are TODO.
Only "the ISA cannot do it at all" earns an impossible.

| col | value |
|---|---|
| ref | `KM_H(KM_ALL)` |
| mmxI, sseI, sse2I | `KM_T` x3 |
| mmxA, sseA, sse2A | `KM_T` x3 |
| armC | `KM_H(LZ_KMB_ARM)` |
| armA | `KM_T` |
| g128 | `KM_N` |

## The edit

1. `LZ_KM_NUNIT` 15 -> 17 (`ops.c:3780`)
2. `km_unit[]` += `{ "recur", LZ_KM_OP }, { "epi", LZ_KM_OP }` (:3822)
3. `km_reg[]` += the two rows above, after f32mm (:4202)
4. why-strings for the `KM_T` and `KM_N` cells
5. `km_op_present()` probes (:4305) - `#if defined(LZ_GDN1_MMX_EXTERN)`
   for recur's gcc cells, `#if defined(__WATCOMC__)` for mmxA, chosen by
   build class the way FWHTI is at :4312, or a gcc build reports the
   Watcom cell present and the gate fails with "present but not
   registered for this build class"

Verify on all three build classes. A row registered but not probeable
fails `build/kernel_matrix_gate.sh`, and it fails for the next person in
a way that looks like their bug.
