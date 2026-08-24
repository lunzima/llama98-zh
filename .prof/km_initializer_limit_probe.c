/* Does Open Watcom accept the km_reg initializer count that adding the
   recur and epi units would need?

   ops.c carries a note that the nested [NUNIT][NCOL] form drew
   "E1175: Too many initializers" on the fifteenth row, and that the flat
   form is what bought the room. The table is at 15 units x 10 columns =
   150 today. recur and epi take it to 17 x 10 = 170. Whether flat still
   holds at 170 is the question, and finding out by editing ops.c would
   put the answer behind a file another session has checked out.

   Build:  wcc386 -6r -zq .prof/km_initializer_limit_probe.c
   PASS is a clean compile. E1175 here means the flat form has its own
   ceiling and the two new units need a different container.

   RESULT: recur and epi are not blocked by the initializer count. The
   installed Open Watcom compiles 17 x 10 clean, and it does so in BOTH
   shapes - including the nested one ops.c says drew E1175 on its
   fifteenth row.

   That took two controls to establish honestly. The first (nested at
   17 x 10) would not go red, which meant the probe might simply be
   unable to see the diagnostic. The second pushes to 130 rows / 1300
   cells, nested, with the string literals the real table carries. It
   compiles too. So this wcc386 has no ceiling anywhere near the sizes
   in play, and the greens above are real rather than blind.

   Which leaves ops.c's own comment: it justifies the flat shape as
   forced by a compiler limit, and that limit does not reproduce here.
   It may have been true of a different Open Watcom, or the original
   E1175 may have had another cause. Either way it now reads as a hard
   constraint on the table's shape and is not one for this toolchain.
   NOT acted on: flat works, ops.c is checked out elsewhere, and a
   comment that was true once is a different problem from a wrong one. */

typedef struct { unsigned char have; unsigned char why; const char *text; } LZKmReg;
#define KM_H(mask) { (unsigned char)(mask), 0, (const char *)0 }

#define KM_I(text) { 0, 1, (text) }
#define KM_N(text) { 0, 2, (text) }

/* The four why-macros put a string literal in the initializer. km_reg's
   operator rows carry three of them each, so a faithful row has to as
   well - the count alone was not the constraint. */
static const char why_a[] =
    "impossible: MMX is integer-only and this is a float multiply-add. "
    "SSE1 is the first tier that can express it";
static const char why_b[] =
    "this column is the 128-ELEMENT quantization group; this operator's "
    "unit of work is a whole row, a different axis";

/* One grid line, ten columns - the shape of a real operator row. */
#define R10 KM_H(7), KM_H(1), KM_I(why_a), KM_H(1), KM_H(2), \
            KM_I(why_a), KM_H(2), KM_H(4), KM_H(4), KM_N(why_b)

/* Seventeen of them: the fifteen that exist plus recur and epi. */
static const LZKmReg km_reg_probe[17 * 10] = {
    R10, R10, R10, R10, R10, R10, R10, R10, R10,
    R10, R10, R10, R10, R10, R10, R10, R10
};

/* Control. A probe that only ever compiles proves nothing about the
   limit it claims to test, so this is the shape ops.c says DOES fail:
   the same 170 cells with the row braces put back.

   Build:  wcc386 -6r -zq -DKM_PROBE_NESTED .prof/...
   Expected: E1175. If this compiles too, the flat/nested distinction in
   ops.c's comment is not what is holding the table together and the
   real constraint is still unidentified. */
#ifdef KM_PROBE_NESTED
static const LZKmReg km_reg_nested[17][10] = {
    { R10 }, { R10 }, { R10 }, { R10 }, { R10 }, { R10 }, { R10 },
    { R10 }, { R10 }, { R10 }, { R10 }, { R10 }, { R10 }, { R10 },
    { R10 }, { R10 }, { R10 }
};
#endif

/* Second control, because the first one would not go red either. If the
   compiler never reports E1175 at any size this probe can express, then
   it is not detecting the diagnostic at all and both greens above are
   worthless. 130 rows / 1300 cells.

   Build:  wcc386 -6r -zq -DKM_PROBE_HUGE .prof/... */
#ifdef KM_PROBE_HUGE
#define R100 { R10 }, { R10 }, { R10 }, { R10 }, { R10 }, \
             { R10 }, { R10 }, { R10 }, { R10 }, { R10 }
static const LZKmReg km_reg_huge[130][10] = {
    R100, R100, R100, R100, R100, R100, R100, R100, R100, R100,
    R100, R100, R100
};
#endif

int main(void) {
#ifdef KM_PROBE_HUGE
    if (km_reg_huge[129][9].why != 2) return 3;
#endif
    /* Touch the last cell so nothing is optimised away before the
       initializer count is checked. */
#ifdef KM_PROBE_NESTED
    if (km_reg_nested[16][9].have != 1) return 2;
#endif
    return km_reg_probe[169].have == 1 ? 0 : 1;
}
