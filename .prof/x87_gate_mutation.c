/* Mutation test for the LZ_X87_STACK_CHECK gate (see ops.h and
   docs/x87-gcc-reg-stack-leak.md). Not a build target - compiled and run
   ad hoc:

     gcc -m32 -march=i686 -mmmx -mfpmath=387 -O2 -std=c99 -Isrc \
         -DLZ_X87_STACK_GATE=1 .prof/x87_gate_mutation.c src/ops.c -lm \
         -o /tmp/x87_gate_mutation
     /tmp/x87_gate_mutation

   Two things this proves, that neither the real bug reproduction nor a
   clean run can: (1) the gate genuinely reddens on a deliberately
   dirtied stack, not just coincidentally on this one bug, and (2) it
   does NOT flag ordinary balanced x87 code as dirty (the "before"
   check, immediately after lz_x87_stack_check's own fnstenv, which
   itself does not push anything - a gate that always fires would pass
   this test's mutant half and still be useless). */
#include <stdio.h>
#include "ops.h"

int main(void) {
    const char *where;
    unsigned tag;

    /* Baseline: nothing has touched the FPU yet in this process, so the
       stack should read clean. If this already reports dirty, the gate
       itself is broken and the rest of the test is moot. */
    LZ_X87_STACK_CHECK("baseline");
    if (lz_x87_stack_dirty(&where, &tag)) {
        printf("FAIL: baseline already dirty at %s tag=%04x\n", where, tag);
        return 1;
    }
    printf("ok: baseline clean\n");

    /* The mutation: one fld with no matching pop, emulating exactly the
       leaked-push shape this whole investigation chased. */
    __asm__ volatile ("fld1");
    LZ_X87_STACK_CHECK("mutation");
    if (!lz_x87_stack_dirty(&where, &tag)) {
        printf("FAIL: gate did not fire after a deliberate unmatched fld\n");
        return 1;
    }
    printf("ok: gate fired at %s tag=%04x (expect one non-empty register)\n",
           where, tag);

    /* Clean up the FPU stack so the process can exit normally. */
    __asm__ volatile ("fstp %%st(0)" ::: "st");
    return 0;
}
