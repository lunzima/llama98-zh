/* Hand-written assembly for the fixed-point pass 2 of the GDN/KDA
   recurrence - the kernel task #25 spent its whole length making
   writable. NOT a standalone translation unit, ops.c #includes it once
   at the point the code sits, no include guards.

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

     A + absmax     MMX  <- this file, and the absmax is the awkward
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
                    SSE  for the split: one instruction of content, and
                          iron law 8 does not allow "one instruction" as
                          a reason to skip a cell. pmaxsw does the low
                          plane's clamp to -127 directly, replacing the
                          psubsw/paddsw pair - 8 instructions saved per
                          32 elements out of ~400, so ~2%, on the PIII,
                          K7 Palomino and Athlon XP (a PII and a K6-2
                          have no SSE1 and stay on the MMX pair). See
                          lz_p2_split32_sse_asm below.
                    SSE2 <- this file, the same sequence 128 bits wide.
                          Real content rather than a copy: every one of
                          pmaddwd/psrad/packssdw/packsswb/pcmpgtd exists
                          on xmm, so the instruction count halves. Still
                          no pmaxsd (SSE4.1), so the absmax stays the
                          expensive half at both widths.

   THIS ONE DOES HELP A PENTIUM II, which is the point. The pass-2
   write-back was measured at 83% + 11% of lz_gdn_step at the all-scalar
   tier, and MMX is the one SIMD the PII has.

   NO emms HERE - rule 6 item 6, and its precondition is met by
   CONSTRUCTION rather than by inspection: the caller (gdn_p2_row_simd)
   is phase-split so that every float computation of the row happens
   before the MMX phase or after it, never between two groups. The
   integer part in between - the group's absmax fold and p2_shift_of -
   touches no x87. Putting the emms back per group would be one per 32
   elements, the exact ratio iron law 6 costs out at 16% of the compute
   budget for the matmul sub-blocks. */

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
   would not guarantee 8 - let alone 16. p2_blk() below does it by hand
   on a static buffer. The MMX kernels tolerated the misalignment (x86
   movq to an unaligned address is legal, just slower); movdqa would not
   have, and finding that out at runtime is a fault, not a wrong
   number. */
typedef struct {
    int16_t mul[8];      /*   0  in   m1,m2 repeated 4x (pmaddwd operand) */
    int32_t amax[4];     /*  16  out  |A| accumulator lanes, caller folds */
    int32_t rnd[4];      /*  32  in   1 << (sh-1), or 0 when sh = 0       */
    int32_t cnt[4];      /*  48  in   sh; the shift count is the low 64   */
    int16_t k128[8];     /*  64  in   128, the split's rounding addend    */
    int16_t kclp[8];     /*  80  in   32641, the lo clamp for MMX/SSE2    */
    int32_t a[32];       /*  96  mid  A, written by one kernel, read by   */
    int16_t kmin[8];     /* 224  in   -127, the same clamp for SSE1       */
} lz_p2_blk;             /*           240 bytes total                     */

/* The assembly below addresses those fields as literal byte offsets, so
   a field that moves is a wrong answer rather than a compile error.
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
        a[] and every one of the 32 literal offsets in the assembly
        below - offsets that a 10-million-element differential has
        already signed off on. Ugly layout beats re-verifying working
        assembly for the sake of field order. */
     offsetof(lz_p2_blk, kmin) == 224 &&
     sizeof(lz_p2_blk)         == 240) ? 1 : -1];

#if defined(__WATCOMC__) && defined(__MMX__)

/* PASS A: blk->a[0..31] = m1*H + m2*dq, and blk->amax[0..3] = four
   partial maxima of |A| (the caller folds the four; doing it in MMX
   would cost more than the three scalar compares it saves).

   Per quad of elements:
     movd/punpcklbw(zero,hi)   hi<<8 without a psraw - putting a byte in
                               the high half of a word IS a multiply by
                               256, sign and all (byte 0xFF becomes
                               0xFF00 = -256). Same trick the Q8 matmul
                               kernels use on the weight side.
     punpcklbw(lo,lo)+psraw 8  lo sign-extended, the ordinary way
     paddw                     H, still int16 because |hi*256+lo| <=
                               32639 - which is why LZ_GDN_LO_SCALE had
                               to stay at 256 and the planes clamp at
                               +-127
     punpcklwd/punpckhwd       interleave H with dq so each dword holds
                               the pair (H, dq) pmaddwd will contract
     pmaddwd x2                A, four elements, no overflow: the bound
                               is 32767*32639 + 32767*32767 = 2.143e9,
                               inside int32 by 4e6. That margin is the
                               reason p2_group_norm clamps m to +-32767
                               rather than letting q8_round return 32768.
     psrad 31 / pxor / psubd   |A|, the branchless two's-complement form
     pcmpgtd/pand/pandn/por    the running max, five instructions per
                               two lanes because MMX has no pmaxsd

   eax=hi edx=lo ebx=dq ecx=blk. Single-plane callers pass a zeroed lo
   buffer rather than getting a second kernel: H = hi*256 + 0 is exactly
   the single-plane H, so one kernel serves both and cannot drift. */
extern void lz_p2_mul32_mmx_asm(const signed char *hi, const signed char *lo,
                                const short *dq, void *blk);
#pragma aux lz_p2_mul32_mmx_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm4, mm4" \
    "pxor      mm3, mm3" \
    "movq      mm5, [ecx]" \
    \
    "movd      mm0, [eax]" \
    "movd      mm1, [edx]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+96], mm0" \
    "movq      [ecx+104], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+4]" \
    "movd      mm1, [edx+4]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+8]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+112], mm0" \
    "movq      [ecx+120], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+8]" \
    "movd      mm1, [edx+8]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+16]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+128], mm0" \
    "movq      [ecx+136], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+12]" \
    "movd      mm1, [edx+12]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+24]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+144], mm0" \
    "movq      [ecx+152], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+16]" \
    "movd      mm1, [edx+16]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+32]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+160], mm0" \
    "movq      [ecx+168], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+20]" \
    "movd      mm1, [edx+20]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+40]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+176], mm0" \
    "movq      [ecx+184], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+24]" \
    "movd      mm1, [edx+24]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+48]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+192], mm0" \
    "movq      [ecx+200], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movd      mm0, [eax+28]" \
    "movd      mm1, [edx+28]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm0" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "paddw     mm2, mm1" \
    "movq      mm6, [ebx+56]" \
    "movq      mm0, mm2" \
    "punpcklwd mm0, mm6" \
    "punpckhwd mm2, mm6" \
    "pmaddwd   mm0, mm5" \
    "pmaddwd   mm2, mm5" \
    "movq      [ecx+208], mm0" \
    "movq      [ecx+216], mm2" \
    "movq      mm1, mm0" \
    "psrad     mm1, 31" \
    "pxor      mm0, mm1" \
    "psubd     mm0, mm1" \
    "movq      mm6, mm2" \
    "psrad     mm6, 31" \
    "pxor      mm2, mm6" \
    "psubd     mm2, mm6" \
    "movq      mm1, mm4" \
    "pcmpgtd   mm1, mm0" \
    "pand      mm4, mm1" \
    "pandn     mm1, mm0" \
    "por       mm4, mm1" \
    "movq      mm6, mm3" \
    "pcmpgtd   mm6, mm2" \
    "pand      mm3, mm6" \
    "pandn     mm6, mm2" \
    "por       mm3, mm6" \
    \
    "movq      [ecx+16], mm4" \
    "movq      [ecx+24], mm3" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087]

/* PASS B: the rescale, the split and the narrowing, blk->a[] -> two
   int8 planes. eax=blk edx=hi_out ebx=lo_out.

   The rescale is a shift with a rounding addend rather than a truncating
   one because truncation is a BIAS and bias adds linearly where
   quantization noise adds in quadrature - measured on this routine,
   5.88e-05 truncating against 4.16e-05 rounded. rnd and cnt are both
   read from memory, which costs nothing here and keeps sh out of a
   register the unrolled body has no room for.

   packssdw and packsswb both saturate, and neither one is ALLOWED to:
   |hn| <= 32513 and |hi| <= 127 are what the shift and the +-127 clamps
   guarantee. Saturation firing would mean the scalar path and this one
   disagree, which is exactly what the differential probe looks for.

   THE lo CLAMP IS THE ONE TRICK IN HERE. lo = hn - hi*256 lands in
   [-128, 127] by construction, and the scalar path clamps it to
   [-127, 127] to match what the float quantizer can emit. Only the
   single value -128 is affected, so instead of a four-instruction
   compare-and-select it is psubsw then paddsw by 32641: x - 32641
   saturates to -32768 only for x = -128 (x = -127 gives exactly -32768
   without saturating), and adding 32641 back returns x for every x >=
   -127 and -127 for x = -128. Two instructions, no mask register, and
   the high end is untouched because 127 - 32641 does not saturate. */
extern void lz_p2_split32_mmx_asm(const void *blk, signed char *hi_out,
                                  signed char *lo_out);
#pragma aux lz_p2_split32_mmx_asm = \
    ".586" \
    "movq      mm0, [eax+96]" \
    "movq      mm1, [eax+104]" \
    "movq      mm2, [eax+112]" \
    "movq      mm3, [eax+120]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx], mm0" \
    \
    "movq      mm0, [eax+128]" \
    "movq      mm1, [eax+136]" \
    "movq      mm2, [eax+144]" \
    "movq      mm3, [eax+152]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+8], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+8], mm0" \
    \
    "movq      mm0, [eax+160]" \
    "movq      mm1, [eax+168]" \
    "movq      mm2, [eax+176]" \
    "movq      mm3, [eax+184]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+16], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+16], mm0" \
    \
    "movq      mm0, [eax+192]" \
    "movq      mm1, [eax+200]" \
    "movq      mm2, [eax+208]" \
    "movq      mm3, [eax+216]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "psubsw    mm0, [eax+80]" \
    "paddsw    mm0, [eax+80]" \
    "psubsw    mm2, [eax+80]" \
    "paddsw    mm2, [eax+80]" \
    "packsswb  mm1, mm3" \
    "movq      [edx+24], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+24], mm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

#define LZ_HAVE_P2_MMX_ASM 1

/* The SSE1 split: the MMX one above with the low-plane clamp done by
   pmaxsw instead of the psubsw/paddsw pair. Everything else is
   identical, because everything else has no SSE1 form (the enumeration
   is in this file's header).

   pmaxsw is an SSE1 addition that operates on MMX REGISTERS, so this is
   still a 64-bit kernel and still leaves MMX state behind - the caller's
   emms rule is unchanged. What it needs is a machine with SSE1, which
   in iron law 3's table means a PIII, a K7 Palomino or an Athlon XP; a
   PII and a K6-2 have MMX only and take the kernel above.

   Naming: the tier is called "sse" because SSE1 is what a machine must
   HAVE to run it, which is the property that decides dispatch. Half of
   the pair it belongs to (lz_p2_mul32) is the MMX kernel unchanged,
   since that half has no SSE1 content at all.

   Two instructions saved per 8 elements, 8 per 32, ~2% of the operator.
   Iron law 8 forbids "too small" as a reason to leave a cell empty, and
   this is what that rule costs and buys, stated so the next reader can
   see both. */
extern void lz_p2_split32_sse_asm(const void *blk, signed char *hi_out,
                                  signed char *lo_out);
#pragma aux lz_p2_split32_sse_asm = \
    ".686" \
    "movq      mm7, [eax+224]" \
    "movq      mm0, [eax+96]" \
    "movq      mm1, [eax+104]" \
    "movq      mm2, [eax+112]" \
    "movq      mm3, [eax+120]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx], mm0" \
    \
    "movq      mm0, [eax+128]" \
    "movq      mm1, [eax+136]" \
    "movq      mm2, [eax+144]" \
    "movq      mm3, [eax+152]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+8], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+8], mm0" \
    \
    "movq      mm0, [eax+160]" \
    "movq      mm1, [eax+168]" \
    "movq      mm2, [eax+176]" \
    "movq      mm3, [eax+184]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+16], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+16], mm0" \
    \
    "movq      mm0, [eax+192]" \
    "movq      mm1, [eax+200]" \
    "movq      mm2, [eax+208]" \
    "movq      mm3, [eax+216]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+24], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+24], mm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

#define LZ_HAVE_P2_SSE_ASM 1

/* ---- the same two kernels, 128 bits wide ------------------------------

   Instruction for instruction the MMX pair above, on xmm: SSE2 has every
   one of pmaddwd, psrad, packssdw, packsswb, pcmpgtd, pand/pandn/por and
   the saturating pair. Half the instructions per element, and the
   machines it is for are the P4 Northwood and the Pentium M (iron law
   3's table: the only two rows with a check in the SSE2 column).

   STILL NO pmaxsd - that is SSE4.1, which none of the target machines
   have. So the int32 absmax remains four instructions plus a copy, the
   same shape as the MMX one, and it remains the expensive half.

   ALIGNMENT IS NOW A FAULT, NOT A SLOWDOWN. movdqa and the m128
   shift-count operand require 16 bytes; ops.c's p2_blk() provides it by
   hand because -zp4 caps what the type system can promise at 4. The
   caller's planes get movq and movdqu: their alignment is not ours.

   No emms anywhere near this: xmm is not MMX state. ops.c's row driver
   skips its emms entirely at this tier, which is worth a few cycles a
   row on the P4 - the machine where an unnecessary MMX/x87 transition
   costs the most. */
extern void lz_p2_mul32_sse2_asm(const signed char *hi, const signed char *lo,
                                 const short *dq, void *blk);
#pragma aux lz_p2_mul32_sse2_asm = \
    ".686" \
    "pxor      xmm7, xmm7" \
    "pxor      xmm5, xmm5" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm6, [ecx]" \
    \
    "movq      xmm0, [eax]" \
    "movq      xmm1, [edx]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+96], xmm0" \
    "movdqa    [ecx+112], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+8]" \
    "movq      xmm1, [edx+8]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+16]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+128], xmm0" \
    "movdqa    [ecx+144], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+16]" \
    "movq      xmm1, [edx+16]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+32]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+160], xmm0" \
    "movdqa    [ecx+176], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+24]" \
    "movq      xmm1, [edx+24]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+48]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+192], xmm0" \
    "movdqa    [ecx+208], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm4" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm4" \
    "por       xmm5, xmm1" \
    "movdqa    [ecx+16], xmm5" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087]

/* The 128-bit split. Every constant stays in memory - all four are
   16-byte aligned m128 operands, which frees all eight registers for the
   sixteen elements in flight and is why this needs no register budget
   note the way the MMX one does. */
extern void lz_p2_split32_sse2_asm(const void *blk, signed char *hi_out,
                                   signed char *lo_out);
#pragma aux lz_p2_split32_sse2_asm = \
    ".686" \
    "movdqa   xmm0, [eax+96]" \
    "movdqa   xmm1, [eax+112]" \
    "movdqa   xmm2, [eax+128]" \
    "movdqa   xmm3, [eax+144]" \
    "paddd    xmm0, [eax+32]" \
    "paddd    xmm1, [eax+32]" \
    "paddd    xmm2, [eax+32]" \
    "paddd    xmm3, [eax+32]" \
    "psrad    xmm0, [eax+48]" \
    "psrad    xmm1, [eax+48]" \
    "psrad    xmm2, [eax+48]" \
    "psrad    xmm3, [eax+48]" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "movdqa   xmm1, xmm0" \
    "paddw    xmm1, [eax+64]" \
    "psraw    xmm1, 8" \
    "movdqa   xmm3, xmm2" \
    "paddw    xmm3, [eax+64]" \
    "psraw    xmm3, 8" \
    "movdqa   xmm4, xmm1" \
    "psllw    xmm4, 8" \
    "psubw    xmm0, xmm4" \
    "movdqa   xmm5, xmm3" \
    "psllw    xmm5, 8" \
    "psubw    xmm2, xmm5" \
    "psubsw   xmm0, [eax+80]" \
    "paddsw   xmm0, [eax+80]" \
    "psubsw   xmm2, [eax+80]" \
    "paddsw   xmm2, [eax+80]" \
    "packsswb xmm1, xmm3" \
    "movdqu   [edx], xmm1" \
    "packsswb xmm0, xmm2" \
    "movdqu   [ebx], xmm0" \
    \
    "movdqa   xmm0, [eax+160]" \
    "movdqa   xmm1, [eax+176]" \
    "movdqa   xmm2, [eax+192]" \
    "movdqa   xmm3, [eax+208]" \
    "paddd    xmm0, [eax+32]" \
    "paddd    xmm1, [eax+32]" \
    "paddd    xmm2, [eax+32]" \
    "paddd    xmm3, [eax+32]" \
    "psrad    xmm0, [eax+48]" \
    "psrad    xmm1, [eax+48]" \
    "psrad    xmm2, [eax+48]" \
    "psrad    xmm3, [eax+48]" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "movdqa   xmm1, xmm0" \
    "paddw    xmm1, [eax+64]" \
    "psraw    xmm1, 8" \
    "movdqa   xmm3, xmm2" \
    "paddw    xmm3, [eax+64]" \
    "psraw    xmm3, 8" \
    "movdqa   xmm4, xmm1" \
    "psllw    xmm4, 8" \
    "psubw    xmm0, xmm4" \
    "movdqa   xmm5, xmm3" \
    "psllw    xmm5, 8" \
    "psubw    xmm2, xmm5" \
    "psubsw   xmm0, [eax+80]" \
    "paddsw   xmm0, [eax+80]" \
    "psubsw   xmm2, [eax+80]" \
    "paddsw   xmm2, [eax+80]" \
    "packsswb xmm1, xmm3" \
    "movdqu   [edx+16], xmm1" \
    "packsswb xmm0, xmm2" \
    "movdqu   [ebx+16], xmm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

#define LZ_HAVE_P2_SSE2_ASM 1

#endif /* __WATCOMC__ && __MMX__ */
