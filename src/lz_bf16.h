/* bfloat16 arithmetic: 1 sign + 8 exponent + 7 mantissa, i.e. the high
   half of an IEEE binary32.

   WHY THIS EXISTS AND WHAT IT IS WORTH, measured rather than assumed.
   On a target with no FPU every float operation is software, so the
   question is how many INSTRUCTIONS a narrower mantissa saves. Counted
   entry to return with the fast path taken, off the disassembly:

       multiply    f32 33   bf16 31      (-2)
       add         f32 64   bf16 58      (-6)

   THE SAVING IS SMALL AND THAT IS THE FINDING. A 24x24 product is 48
   bits and STRADDLES hi:lo, so f32 needs UMULL where bf16 needs MUL,
   and needs the low word's bits jammed into a sticky where bf16's
   16-bit product sits in one register with its round bits already in
   place. That is the entire difference. Everything else - unpacking two
   exponents, the special-case chain, sign, exponent arithmetic,
   normalisation, rounding, packing - is the SAME COUNT at either width.
   Floating point in software costs what it costs because of its FORMAT,
   not its precision. Hardware pays by bit (an 11x11 multiplier array is
   about a fifth of a 24x24 one, which is why TF32 buys 8x throughput);
   software pays by step, and the steps do not change.

   WHAT THE REST OF THE WORLD DOES, because this library is unusual and
   a reader should know it. PyTorch, Eigen, TensorFlow, ggml, the Rust
   `half` crate and gcc's own __bf16 lowering all treat bf16 as a
   STORAGE format: every operator widens to f32, computes, and rounds
   back. libgcc has no __addbf3 or __mulbf3 at all. Arm's FEAT_BF16,
   RISC-V's Zvfbfwma and Intel's AVX512_BF16 provide conversion and
   f32-accumulating dot products, and no bf16 scalar add or multiply;
   the RISC-V committee declined to add them, noting that widening is
   FAITHFUL - f32's 24-bit significand is at least 2*7+2, so widen,
   compute, narrow rounds exactly once and equals the native answer.
   Native bf16 arithmetic appears only in format-generic emulators
   (QEMU's softfloat, LLVM's APFloat), where it was one table row.

   That reasoning does not transfer here, which is why this file exists:
   everyone widens because f32 is one FPU instruction, and on ARM926EJ-S
   it is a call into lz_softfp.c. Measured, the conventional path costs
   6 instructions of glue plus __aeabi_fmul plus lz_f32_to_bf16's 17 -
   against 31 native.

   Even so, bf16 is not mainly a speed play here. It is worth having for
   half the memory traffic on a 248MHz part whose bottleneck is DDR, and
   for a dynamic range identical to f32 (8 exponent bits), which is what
   makes it a drop-in where f16's 5-bit exponent would need loss
   scaling.

   BF16 IS NOT AN IEEE FORMAT, so no standard fixes its denormal and NaN
   behaviour and implementations genuinely disagree (Arm's BFDOT flushes
   denormals and rounds to odd; Intel's AVX512_BF16 excludes denormals;
   Eigen and Intel disagreed on NaN conversion). This file follows the
   full-IEEE reading - denormals supported, round-to-nearest-even - which
   is QEMU's, the only complete and testable definition available.

   ROUNDING CONTRACT. Every operation here is defined as "compute in f32
   and round the result to bf16, round-to-nearest-even" - that is what
   the tests compare against, and what the rest of the world means by a
   bf16 op. LZ_BF16_FAST trades the tie-to-even for truncation, which
   is a different contract and says so.
*/
#ifndef LZ_BF16_H
#define LZ_BF16_H

#include "lz_int.h"

#if defined(__GNUC__) || defined(__WATCOMC__)

/* A bf16 lives in the low 16 bits of a word. Not a typedef'd uint16_t:
   every operation below takes and returns it in a register, and the
   ARM ABI would otherwise insist on truncating at each boundary. */

/* f32 <-> bf16. to_f32 is a shift; from_f32 is a shift plus the
   round-to-nearest-even that makes it the inverse and not just a
   truncation (safetensors.c does the shift-only direction on load). */
uint32_t lz_bf16_to_f32(uint32_t h);
uint32_t lz_f32_to_bf16(uint32_t f);

/* Arithmetic. Operands and result are bf16 bit patterns. */
uint32_t lz_bf16_add(uint32_t a, uint32_t b);
uint32_t lz_bf16_sub(uint32_t a, uint32_t b);
uint32_t lz_bf16_mul(uint32_t a, uint32_t b);
uint32_t lz_bf16_div(uint32_t a, uint32_t b);

/* Comparison: -1 / 0 / 1, and 2 when either operand is NaN (unordered),
   so a caller can tell unordered from equal without a second test. */
int lz_bf16_cmp(uint32_t a, uint32_t b);

/* int <-> bf16. */
uint32_t lz_bf16_from_i32(int32_t v);
int32_t lz_bf16_to_i32(uint32_t h);

#endif /* __GNUC__ || __WATCOMC__ */
#endif /* LZ_BF16_H */
