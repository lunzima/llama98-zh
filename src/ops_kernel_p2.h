/* Algorithm documentation and the shared scratch-block layout check for
   the fixed-point pass 2 of the GDN/KDA recurrence. NOT a standalone
   translation unit, ops.c #includes it once at the point the code
   sits, no include guards.

   The Watcom #pragma aux bodies live in src/ops_mmx.c /
   src/ops_mmx_sse.c / src/ops_sse2.c - see this file's tail for why
   (the same -Nr/-Ns ceiling argument found on cpucheck.c). This file
   stays the authority on the algorithm itself; only the code is
   elsewhere.

   The gcc intrinsics twins of the MMX/SSE1 kernels (lz_p2_mul32_mmx,
   lz_p2_split32_mmx, lz_p2_split32_sse) live in src/ops_mmx.c too -
   they write %mm registers, so they belong in the one translation unit
   built with -mmmx (see src/ops_mmx.h). Both toolchains' bodies live
   in the same file, for different reasons (register-file aliasing for
   gcc, the CPU-generation ceiling for Watcom).

   WHAT THE OPERATOR IS. Per (row, 32-group), with the float prologue
   already reduced to two int16 multipliers (p2_group_norm in ops.c):

     H[j]  = hi[j]*256 + lo[j]                      |H| <= 32639, int16
     A[j]  = m1*H[j] + m2*dq[j]                     int32
     amax  = max |A[j]|
     sh    = smallest shift with (amax >> sh) <= 32512
     hn[j] = (A[j] + (1 << (sh-1))) >> sh           |hn| <= 32513, int16
     hi[j] = (hn[j] + 128) >> 8                     int8
     lo[j] = hn[j] - hi[j]*256, clamped to >= -127  int8

   Every line is an integer operation MMX has: the A term is ONE pmaddwd
   (H and dq interleave into the two int16 lanes of a dword, the
   multiplier pair broadcasts), the rescale is psrad, the split is
   paddw/psraw and psllw/psubw, the narrowing is packssdw/packsswb.
   That is the whole point of the two changes that preceded this file:
   LZ_GDN_LO_SCALE = 256 turned the split from a divide into a shift
   (ops.h has the exhaustive search that closed the 254 door), and
   deriving the rescale from a shift instead of A*32512/amax removed the
   last divide (ops.c, gdn_p2_group_fixed).

   ISA INVENTORY - what is here and what is deliberately not:

     A + absmax     MMX  <- src/ops_mmx.c, and the absmax is the awkward
                          half: MMX has no pmaxsd, so a max of int32
                          costs pcmpgtd + pand + pandn + por, five
                          instructions per two lanes counting the copy,
                          against one for a hypothetical pmaxsd. An OR
                          of the |A| would be four times cheaper and is
                          NOT usable: it has the same bit length as the
                          max but not the same value, and sh is decided
                          against 32512, which is not a power of two.
                          An sh one too large is one bit of state range
                          silently lost on some groups and nothing red
                          anywhere, so the exact max it is.
                    SSE  for the A term and the absmax: NO CONTENT, and
                          this one is structural. SSE1's additions to the
                          MMX register file are pshufw pinsrw pextrw
                          pmovmskb pmulhuw pavgb pavgw pmaxsw pminsw
                          pmaxub pminub psadbw maskmovq movntq. None
                          contracts int16 pairs (that is pmaddwd), none
                          maximizes int32 (pmaxsw is 16-bit and A is
                          32), none sign-extends bytes to words
                          (punpck), and movntq on the A buffer would be
                          a pessimization since the split reads it back
                          immediately. So the SSE1 cell is empty for
                          THIS kernel by enumeration, not by size.
                    SSE  for the split: one instruction of content, which
                          is not a reason to leave the cell empty
                          either. pmaxsw does the low
                          plane's clamp to -127 directly, replacing the
                          psubsw/paddsw pair - 8 instructions saved per
                          32 elements out of ~400, so ~2%, on the PIII,
                          K7 Palomino and Athlon XP (a PII and a K6-2
                          have no SSE1 and stay on the MMX pair). See
                          lz_p2_split32_sse_asm, src/ops_mmx_sse.c.
                    SSE2 <- src/ops_sse2.c, the same sequence 128 bits
                          wide. Real content rather than a copy: every
                          one of pmaddwd/psrad/packssdw/packsswb/pcmpgtd
                          exists on xmm, so the instruction count halves.
                          Still no pmaxsd (SSE4.1), so the absmax stays
                          the expensive half at both widths.

   THIS ONE DOES HELP A PENTIUM II, which is the point. The pass-2
   write-back was measured at 83% + 11% of lz_gdn_step at the all-scalar
   tier, and MMX is the one SIMD the PII has.

   NO emms HERE - the caller emits one after the row loop, and the
   precondition is met by CONSTRUCTION rather than by inspection: the
   caller (gdn_p2_row_simd)
   is phase-split so that every float computation of the row happens
   before the MMX phase or after it, never between two groups. The
   integer part in between - the group's absmax fold and p2_shift_of -
   touches no x87. Putting the emms back per group would be one per 32
   elements, the exact ratio that costs 16% of the compute budget for
   the matmul sub-blocks. */

/* The scratch block every tier of this operator reads and writes. It
   exists because Watcom's #pragma aux passes at most a handful of
   register arguments comfortably and these kernels need eight operands
   between them; packing the small ones into one aligned block leaves
   four pointers per call.

   EVERY FIELD IS 16 BYTES WIDE AND THE BLOCK IS 16-BYTE ALIGNED, which
   is more than MMX needs and exactly what SSE2 needs:

     - SSE2's shift-count and constant operands are m128, and every SSE2
       memory operand except the movdqu family FAULTS on a misaligned
       address. Not "runs slower" - #GP.
     - MMX reads the low 8 bytes of the same fields and gets the same
       values, because each constant is stored replicated. One layout,
       two tiers, no second struct to keep in step.

   The alignment cannot come from the type. Watcom is built here with
   -zp4, which caps member alignment at 4, so even a union with a double
   would not guarantee 8 - let alone 16. p2_blk() (src/ops_mmx.c, both
   toolchains) does it by hand on a static buffer. The
   MMX kernels tolerated the misalignment (x86 movq to an unaligned
   address is legal, just slower); movdqa would not have, and finding
   that out at runtime is a fault, not a wrong number. */
/* The lz_p2_blk type lives in src/ops_p2_blk.h, not here: this file,
   src/ops_mmx.c and src/ops_sse2.c all need it, and ops_p2_blk.h is the
   one with a real include guard - the guard-less "included exactly
   once" contract above is ops.c's, about THIS file, and does not
   extend to a struct shared with several other translation units. */
#include "ops_p2_blk.h"

/* The assembly addresses those fields as literal byte offsets, so a
   field that moves is a wrong answer rather than a compile error.
   These make it a compile error. */
typedef char lz_p2_blk_layout_check[
    (offsetof(lz_p2_blk, mul)  ==  0 &&
     offsetof(lz_p2_blk, amax) == 16 &&
     offsetof(lz_p2_blk, rnd)  == 32 &&
     offsetof(lz_p2_blk, cnt)  == 48 &&
     offsetof(lz_p2_blk, k128) == 64 &&
     offsetof(lz_p2_blk, kclp) == 80 &&
     offsetof(lz_p2_blk, a)    == 96 &&
     /* kmin is APPENDED, after the 128-byte a[], rather than slotted in
        beside the other constants. Tidier grouping would have shifted
        a[] and every one of the 32 literal offsets in the assembly -
        offsets that a 10-million-element differential has already
        signed off on. Ugly layout beats re-verifying working assembly
        for the sake of field order. */
     offsetof(lz_p2_blk, kmin) == 224 &&
     sizeof(lz_p2_blk)         == 240) ? 1 : -1];

#if defined(__WATCOMC__) && defined(__MMX__)
/* The #pragma aux bodies (lz_p2_mul32_mmx_asm, lz_p2_split32_mmx_asm,
   lz_p2_split32_sse_asm, lz_p2_mul32_sse2_asm, lz_p2_split32_sse2_asm)
   live in src/ops_mmx.c (MMX tier and the SSE1-split's mul32 half),
   src/ops_mmx_sse.c (the SSE1 split itself) and src/ops_sse2.c (the
   SSE2 pair). Reason: Watcom's -Nr/-Ns CPU-generation flag is a hard
   ceiling its own per-function .586/.686 directives cannot exceed, so
   ops.c sitting at the i486/i386 floor and these kernels needing
   .586/.686 codegen are only compatible if the kernels live in a
   translation unit with its own, higher ceiling. The algorithm
   documentation above (WHAT THE OPERATOR IS, ISA INVENTORY) stays the
   authority every one of those files' comments points back to.

   These three flags mean "Watcom has a genuine #pragma aux
   implementation of this tier", which src/ops.c's own cascade (search
   LZ_HAVE_P2_MMX_ASM) and lz_gdn_p2_impl()'s asm-vs-intrin diagnostic
   both need to tell apart from LZ_P2_MMX_EXTERN/LZ_P2_SSE2_EXTERN's
   "a real extern function elsewhere has it" - true for gcc always, and
   also true for Watcom's row-granularity functions (lz_p2_rows_mmx/
   _sse/_sse2), which call the pragmas by their real names from another
   TU rather than aliasing a bare name to them. Nothing downstream
   depends on that: the diagnostic and the LZ_HAVE_P2_MMX/SSE/SSE2
   defines only ever test "is Watcom" via these three flags, never "is
   the code still physically here". */
#define LZ_HAVE_P2_MMX_ASM  1
#define LZ_HAVE_P2_SSE_ASM  1
#define LZ_HAVE_P2_SSE2_ASM 1
#endif /* __WATCOMC__ && __MMX__ */
