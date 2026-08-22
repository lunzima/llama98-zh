# f32 census audit: what the LZ_FCX sites miss, measured

The census (`-DLZ_F32COUNT=1`) is a set of hand-written formulas at
`LZ_FCX` call sites. This directory holds an instrument that counts the
same quantity **without** formulas, so the two can be compared, and the
result of running it on kmr20.

## Before re-running: two things that make it lie

**`covbuild.sh`, `measure.sh` and `genrawfix.sh` do not run from this
directory.** Each does `cd "$(dirname "$0")"` and then uses repo-root
relative paths (`src/$tu.c`, `build/cov`), so from `.prof/census_audit/`
the first dies with `cc1: fatal error: src/err.c: No such file or
directory`. Copy them to the root of a throwaway `git archive` checkout
and run them there. Not fixed in place because validating the fix means
re-running the whole measurement.

**`linemap.tsv` keys 36 rows to `src/forward.c` LINE NUMBERS**, so any
edit to that file moves work between sites without failing. It has
happened: a commit landing between the measurement and the publication
shifted `forward.c` and moved 6,240 ops/token out of the NOSITE rows
into SIGMOID. `compare.sh` now refuses when `src/forward.c` is not the
blob the map was built against, warns when it cannot tell, and takes
`CENSUS_AUDIT_SKIP_BLOBCHECK=1` for the regenerate-then-rerun case. The
recorded blob is `b143c6c5` (`e65f181`'s); **the map is stale for
anything later and needs regenerating, not shifting by hand.**
Regenerate it by matching each mapped line's source TEXT into the new
file, not by shifting the numbers: 34 of the 36 land on a unique line
and the two that do not (`s->qn[i] *= scale`, the residual add) are
placed by which of their two occurrences the old map named.

**`slotmap.tsv` and `rawcounters.diff` are still `0283fa0`'s** and were
not regenerated with the map. Whoever needs the full 26-slot override
table again has to redo them. The conversion measurement below did not:
only two of its lines are UNEQUAL and both were resolved by the
leaf-total constraint (below).

## Method

One run of one binary produces both numbers.

1. `covbuild.sh` compiles every engine TU `-O0 -g --coverage` (the three
   intrinsic TUs keep `-mmmx -msse -msse2`) and links `llama98_covall.exe`
   with `-DLZ_F32COUNT=1` still on.
2. `measure.sh` reads that run back:
   - `objdump -dlr --inlines` gives, per source line, how many float
     instructions the compiler emitted there. `-O0` means one source
     operation is one instruction, which is the census's own unit.
     `--inlines` charges an always_inline intrinsic body to the `.c` line
     that called it, not to `xmmintrin.h`.
   - `gcov -a` gives how many times each line ran.
   - `dis2lines.awk` + `joinall.awk` multiply them, per
     (TU, file, line, function, mnemonic), and also record every call
     site so an unbilled leaf helper can be charged to its caller.
3. `classify.awk` splits the result into the census's convention: `f32`
   is mul/add/sub/div plus int<->float converts; `cmp` (compares, sign
   masks) and `f64` (doubles, including the census counters' own
   accumulation) are counted separately and excluded. A packed op counts
   as 4 (`ps`) or 2 (`pd`) element operations.
4. `rollup.awk` maps functions to census sites (`sitemap.tsv`,
   `linemap.tsv` for the files that hold work belonging to several
   sites) and charges leaf helpers to their call sites.
5. `compare.sh` prints census vs measured per site.

## Two things the instrument had to get right

**gcov's printed line count is not an execution count.** It is the sum
of the counts of every basic block starting on that line, so a line
holding three inlined intrinsics prints 3x, and a `for (...) body;` line
prints the loop-condition block's count. `joinall.awk` therefore reads
the `-a` block detail and uses the common value when all blocks agree,
flagging the line `UNEQUAL` when they do not.

**The flagged lines are resolved by counting, not by reasoning.**
`rawcounters.diff` patches a throwaway checkout with `lz_rawc[]`, one
slot per flagged line plus seven controls on lines gcov reports
unambiguously. `genrawfix.sh` turns that run's output into the override
table. The controls all agreed with gcov exactly; the flagged lines did
not, and neither the min nor the max block count was right in general
(`for (e = 0; e < 32; e++)` has blocks C, 32C, 33C and the answer is the
middle one).

## Validation

- The counting build is **bit-identical** to the shipping build: same
  logits, same census, in both tier arms and both fixture corpora.
- The raw-counter build is bit-identical too.
- In the float arm (`--fixed off`), five sites reproduce the census's own
  number exactly - conv1d 147,456 (delta 0), attention wsum 2,473,984
  (delta 0), rsqrt 2,925 (delta 0), attention scoring (delta 4), softmax
  (delta 16), exp (delta -174). Where the census formulas describe a
  plain float loop, an independent measurement lands on the same digit;
  the disagreement is concentrated in the int/fixed paths.
- Leaf per-call costs come out as whole numbers that match the source:
  `q8_round` 1.000, `lz_i32f` 4.000, `lz_q8round32_simd` 64.000,
  `pow2f` 0.000 (it is a bit trick, and bills nothing correctly).

## Result (kmr20, `build/norm_int_gate_tokens_zh.txt`, 603 tokens)

Shipping default. Per token.

| site | census | measured | ratio |
|---|---|---|---|
| recurrence write | 110,592 | 419,122 | 3.79x |
| sigmoid | 158,144 | 265,980 | 1.68x |
| conv1d | 129,024 | 259,781 | 2.01x |
| dequant epilogue | 201,020 | 236,076 | 1.17x |
| norms | 168,020 | 198,808 | 1.18x |
| attention scoring | 101,472 | 159,460 | 1.57x |
| quantize | 101,532 | 102,684 | 1.01x |
| exp | 77,512 | 99,654 | 1.29x |
| attention wsum | 64,874 | 71,942 | 1.11x |
| (no site) | 0 | 50,611 | - |
| softmax | 14,512 | 14,528 | 1.00x |
| rsqrt | 1,112 | 794 | 0.71x |
| **total** | **1,127,814** | **1,879,329** | **1.67x** |

The English fixture gives the same ratios (1.66x total). The float arm
(`--fixed off`) is 14,678,287 billed against 23,018,697 measured, 1.57x.

Three mechanisms account for nearly all of the gap:

- `lz_i32f` is billed as one convert and costs four operations
  (`(float)(v>>12) * 4096.0f + (float)(v & 0xFFF)`). 65,077 calls/token,
  so 195,231 ops/token are invisible, spread over conv1d, the
  recurrence, and both attention sites.
- `sigmoid_q15` (6.94 ops/call) and `sigmoid_q15_t` (4.99) are leaves
  whose callers bill a smaller constant than they cost.
- `forward.c`'s own element-wise loops have no site at all: the MoE
  SwiGLU multiply and expert combine, the two residual adds, the
  q-scale loop, the attention output gate, RoPE, `lz_t_row_f32`.

## Conversion-ranked view (`cvtroll.awk`)

`cvtroll.awk` is the same collection filtered to int<->float converts,
which is the ranking that matters when the target has no FPU: a convert
is the boundary, and removing one is what lets a chain stay integer.
Same attribution as `rollup.awk` (validated: its `f32` column reproduces
`rollup.awk`'s on all 18 sites that carry work, exactly).

    awk -v tokens=603 -v sitemap=sitemap.tsv -v linemap=linemap.tsv \
        -v rawfix=<f> -v magic=magic.tsv -f cvtroll.awk measured.tsv

kmr20, zh fixture, shipping default, `e65f181`. 514,226 hardware
converts/token against 263,928 billed - the census sees 51.3%. Adding
the conversions performed with no cvt instruction (`magic.tsv`:
`q8_round`'s magic add, `lz_exp`'s floor) takes the true count to
664,183 and the census's share to 39.7%.

Three things the instruction count gets right that a formula cannot:

- `lz_i32f` measures 2.000 converts per call, so its 130,154/token show
  up whether or not the caller's formula says `cvt`.
- **int<->double is not counted, and that is not an assumption.** Every
  executed `cvtsi2sd`/`cvttsd2si` in the run was checked: all of them
  are `LZ_FCX` counter arguments, which exist only in the counting
  build, or one-time table setup.
- `q8_round` emits no cvt instruction at all. It is 138,884 float->int
  conversions per token that an instruction count scores 0, which is
  why `magic.tsv` declares them instead.

The two UNEQUAL lines that carry conversions were resolved without the
raw-counter build, by the constraint that the parts sum to the leaf's
own unambiguous instruction count:

- `ops_gdn.c:336` (`lz_attn_wsum_q8` -> `lz_i32f`): `lz_i32f`'s own
  78,482,944 converts / 2 minus the 37,820,160 unambiguous calls leaves
  1,421,312, i.e. 2,357.07/token - the number the census bills there.
- `ops_gdn.c:1196/1197` (`p2_group_norm` -> `q8_round`): 83,746,958
  minus 68,943,764 leaves 14,803,194. Only the sum is measured; both
  lines are RECUR and `q8_round` emits no cvt, so the split is inert.

Everything else is bracketed by running `basis=hi` and `basis=lo`:
512,743..514,226, 0.29%. The `f32` denominators are looser (2.3%
overall, 11% on EPI) because they include lines the conversions do not.

## Not covered

- libm calls the engine still makes on this path (`ldexp` in the fixed
  conv build, `pow`/`sin`/`cos` in the RoPE table) - under 80 calls per
  token, their internals are not counted.
- The front end (`cli_main.c`), which is not part of the forward pass.
- Compares and absolute values, 715,658/token measured here, excluded by
  the census's stated convention rather than by oversight.
