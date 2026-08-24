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

   RESULT SO FAR: INCONCLUSIVE, and the control is why. Both forms
   compile clean at 17 x 10 under wcc386 -6r - flat AND the nested shape
   ops.c says drew E1175 on its fifteenth row. A control that will not
   go red means this probe is not reproducing the real constraint, so
   its green says nothing about whether recur and epi fit.

   What this file leaves out that the real table has: every row here is
   KM_H with a null pointer, while km_reg mixes KM_H, KM_I, KM_N, KM_T
   and KM_D, and the last four each put a string literal into the
   initializer. String literals are the obvious next suspect - reproduce
   with those before trusting any answer here. */

typedef struct { unsigned char have; unsigned char why; const char *text; } LZKmReg;
#define KM_H(mask) { (unsigned char)(mask), 0, (const char *)0 }

/* One grid line, ten columns - the shape of a real row. */
#define R10 KM_H(7), KM_H(1), KM_H(0), KM_H(1), KM_H(2), \
            KM_H(0), KM_H(2), KM_H(4), KM_H(4), KM_H(1)

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

int main(void) {
    /* Touch the last cell so nothing is optimised away before the
       initializer count is checked. */
#ifdef KM_PROBE_NESTED
    if (km_reg_nested[16][9].have != 1) return 2;
#endif
    return km_reg_probe[169].have == 1 ? 0 : 1;
}
