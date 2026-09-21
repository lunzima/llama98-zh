# gates.mk - local-private verification targets. NOT SHIPPED.
#
# The public Makefile carries no gates, so a fresh clone gets the engine
# and the build and nothing else. This file restores the development
# gates on top of it for the private tree, where build/ still holds them.
# It is pulled in by the public Makefile's `-include gates.mk`, so
# `make check` works here and is simply absent on a public checkout.
#
# Requires the gates under build/ - the parity checks, the cross-build
# engines (ARM/MIPS/x87), the instruction censuses, the CI entry point -
# which are likewise not shipped. MODEL is a command-line override
# (`make lz4-model-gate MODEL=...`), as before.

# Build == verify (D1). The DLL build doubles as the correctness gate: a
# clean link proves the .def export surface matches the engine, and
# -D__MMX__=1 is baked in so the hand-written assembly cannot silently
# vanish. DOS is left out of `check` because it needs the extender stub
# and is a separate delivery decision; run `make watcom-dos` explicitly.
# bare-endif-gate runs first and needs no build: a source-text invariant
# caught before either compiler spends time on it. dedupe-gate (task #31)
# and dead-ifndef-gate run next, also build-free. mm-alias-
# gate (task #30/#32) runs second, before the real link targets: it
# compiles its own throwaway ops_sse2.c objects (both toolchains, both
# LZ_SSE2_GROUP settings) rather than depending on watcom-cli having
# already produced one, so it does not need to come after it.
# kernel-matrix-gate is last because it is the slowest (it builds the gcc
# CLI, the Watcom CLI and, when the cross toolchain is there, the ARM one
# - the union of three binaries is the whole point, one binary cannot
# show it). LZ_KM_SKIP_ARM=1 drops the ARM arm for a faster loop; the
# gate then says which columns went unchecked.
#
# c89-floor-gate, comment-hygiene-gate and vc4-limits-gate were written,
# committed, and then invoked by nothing: not here, not in ci.sh. They
# passed every time anyone ran them by hand and could not have reddened
# a build. A gate no target depends on is a script.
#
# gcc-warning-gate had a caller - ci.sh - and was still missing HERE,
# which is a weaker version of the same hole: `check` is the target that
# gets run, and it asserted the Watcom warning count while letting the
# gcc one past. Found by a dropped comment terminator that left a `/*`
# nested inside another comment; gcc said so on every build and `check`
# reported 16 PASS over it. Having a caller somewhere is not the test -
# being reachable from what people actually run is.
check: root-hygiene-gate bare-endif-gate wscr-borrow-gate dead-ifndef-gate ppbrace-gate f32count-billing-gate fp-exception-gate bf16-native-gate json-offset-gate softfp-gate dedupe-gate mm-alias-gate kv-share-ced-boundary-gate ced-proj-arm-gate csa2-modes-config-gate csa2-cache-idx-alias-gate csa2-mode-dispatch-gate csa2-mode-export-load-gate csa2-last-writer-bin-gate csa2-reindex-causality-gate csa2-indexer-rope-slice-gate csa2-out-rope-gate csa2-z-subst-gate csa2-compress-m-layers-gate smf-lead-gate c89-floor-gate pri64-scan-gate comment-hygiene-gate ascii-source-gate kernel-reach-gate vc4-limits-gate vc4-compile-gate dup-body-gate twin-pair-gate gcc-watcom-pair-gate gate-wiring-gate gcc-warning-gate gcc-flag-parity-gate watcom-flag-parity-gate seed-gate gui-selftest-gate codesize-gate isa-floor-avx2-gate watcom-cli watcom-gui watcom-dll isa-floor-watcom-gate arch-allowlist-gate kernel-matrix-gate aux-parm-arity-gate aux-equiv-gate mathf-xcompiler-gate bf16-torch-gate lz4-model-gate gcc-watcom-output-gate fixture-exists-gate

# Zero gcc warnings, asserted rather than printed. The Watcom side has
# always been counted ("N lines, 0 warnings, 0 errors", per file); this
# is the same assertion for the toolchain whose warnings only ever went
# to the console. Forces a real compile - see the script for why an
# up-to-date build is the failure mode it has to avoid.
#
# MAKEFLAGS IS CLEARED because the gate proves a real compile happened by
# finding gcc command lines in the sub-make's output, and `make -s check`
# propagates -s to that sub-make - which suppresses exactly those lines.
# The gate then reports "ran no compile", a FAIL that says nothing about
# the tree. It is the gate's own blindness check firing on the harness
# instead of on the subject.
.PHONY: gcc-warning-gate
gcc-warning-gate:
	@MAKEFLAGS= MFLAGS= bash build/gcc_warning_gate.sh

# The GUI and the CLI compile the engine with the SAME flags - same
# optimization level, same ISA gating, same operator selection. A GUI
# built at different settings is a different engine wearing the same
# version string, and the difference would show up as numbers that
# disagree with the CLI's on identical input.
#
# In `check` rather than ci.sh-only because it is seconds and needs no
# model: it reads both flag sets and compares them. It had no target at
# all before - the Makefile named it once, in a comment.
.PHONY: gcc-flag-parity-gate
gcc-flag-parity-gate:
	@bash build/gcc_flag_parity_gate.sh

# Watcom passes #pragma aux arguments to the __parm registers LEFT TO
# RIGHT and ignores the rest, so a spare register at the head of the list
# shifts every argument and leaves the last one holding whatever the
# caller left there. In `check` rather than ci.sh-only: it is a text scan
# of src/, no build and no model.
.PHONY: aux-parm-arity-gate
aux-parm-arity-gate:
	@bash build/aux_parm_arity_gate.sh

# A reorder of a hand-written kernel must not change what it computes.
# SKIPs without pentiumlint, so it costs nothing where the linter is not
# checked out; where it is, it is a second or two of symbolic execution.
.PHONY: aux-equiv-gate
aux-equiv-gate:
	@$(call GATE_OK,build/aux_equiv_gate.sh)

# The same question of the pair that ships to the target. Its own file
# rather than a second half of the one above: that one is named for the
# toolchain it checks, and a gcc_ gate that also checked Watcom would be
# a name to have to explain.
.PHONY: watcom-flag-parity-gate
watcom-flag-parity-gate:
	@bash build/watcom_flag_parity_gate.sh

# Where a fixed random seed is needed it is the same one everywhere. A
# grep over build/, so it belongs in `check` rather than ci.sh: two
# harnesses seeded differently produce incomparable numbers that both
# look reproducible, which is the expensive version of this failure.
.PHONY: seed-gate
seed-gate:
	@bash build/seed_gate.sh

# The GUI's own 421 runtime checks, which nothing ran. They build the
# real windows and read the real controls back, and two of them had
# been red since the beep checkbox landed - a duplicated mnemonic and a
# gap in the Tab chain, both found by a user pressing the key. Depends
# on the binary so the gate can never be handed a stale one.
.PHONY: gui-selftest-gate
gui-selftest-gate: kunkun98-gui
	@bash build/gui_selftest_gate.sh

# Two size criteria for the shipping binaries: the GUI fits on a 1.44 MB
# floppy, and the CLI's .text stays under TEXT_CEIL - the early one, since
# the floppy criterion sits at 73% and would not redden until something had
# gone very wrong. It MEASURES; it builds nothing (in code position it
# invokes make/wcc386/wlink zero times - all four coarse-grep hits for
# those words are prose), it needs no model and no cross toolchain, and it
# runs in about a second. Depends on both artifacts for the reason
# gui-selftest-gate depends on one: a gate handed a stale binary reports a
# size for a configuration nobody shipped.
#
# IN `check` RATHER THAN ci.sh-ONLY, and that relocation is the point of
# this target: parked in ci.sh it was run by nobody, and .text crossed the
# ceiling without anything saying so - 228999a^ measured 328,752 against
# 327,680. The red it now reports in `check` is the debt becoming visible,
# not a new defect.
.PHONY: codesize-gate
codesize-gate: llama98 kunkun98-gui
	@$(call GATE_OK,build/codesize_gate.sh)

# The AVX2 tier's ISA floor, from its own object rather than its flags.
# The tier is auto-selected by kernel_detect on any machine whose CPUID
# passes lz_cpu_has_avx2, so an instruction outside that set is a #GP on a
# machine the selector called capable - and every numeric gate is blind to
# it, since all tiers are required to agree bit for bit. Depends on
# kunkun98-gui for build/gui/ops_avx2.o, the same dependency
# gui-selftest-gate has.
.PHONY: isa-floor-avx2-gate
isa-floor-avx2-gate: kunkun98-gui
	@bash build/isa_floor_avx2_gate.sh

# Every source byte is ASCII. VC++ 4.0 and Watcom read source in the
# host codepage, and on a DBCS host a lead byte swallows the one after
# it - which is sometimes a comment terminator. A byte test, not a
# character test: the character-level search for CJK returns a false
# zero when the locale disagrees with the file.
.PHONY: ascii-source-gate
ascii-source-gate:
	@bash build/ascii_source_gate.sh

# Every hand-written kernel can be reached by a probe. A #pragma aux
# body has no address, so a kernel with no thin non-static wrapper in
# its own file cannot be called, compared or timed from anywhere else -
# and nothing says so, because nothing links against it.
.PHONY: kernel-reach-gate
kernel-reach-gate:
	@bash build/kernel_reach_gate.sh

# Two shared helpers with the same body under different names, which is
# the duplication dedupe-gate cannot see: that one knows the names it is
# looking for, this one compares bodies.
.PHONY: dup-body-gate
dup-body-gate:
	@bash build/dup_body_gate.sh

# Every x86 kernel written twice - gcc intrinsics and a Watcom
# #pragma aux - or registered with a reason for being impossible.
# twin-pair-gate counts the pairs; gcc-watcom-pair-gate checks the two
# sides answer for the same set.
.PHONY: twin-pair-gate
twin-pair-gate:
	@bash build/twin_pair_gate.sh

.PHONY: gcc-watcom-pair-gate
gcc-watcom-pair-gate:
	@bash build/gcc_watcom_pair_gate.sh

# Every gate has a caller. Six did not, and two of those were the only
# checks behind standing requirements - see the script's header.
.PHONY: gate-wiring-gate
gate-wiring-gate:
	@bash build/gate_wiring_gate.sh

# The language floor, as a ratchet plus a real -std=c89 compile. See the
# script's header for what green does and does not mean.
.PHONY: c89-floor-gate
c89-floor-gate:
	@bash build/c89_floor_gate.sh

# Comments say what is true, not when it changed or which rule says so.
.PHONY: comment-hygiene-gate
comment-hygiene-gate:
	@bash build/comment_hygiene_gate.sh

# The translation limits a 1995 compiler may enforce - identifier
# truncation and string literal length. Not the same question as the
# floor gate, which asks whether the code is standard C89 at all.
.PHONY: vc4-limits-gate
vc4-limits-gate:
	@bash build/vc4_limits_gate.sh

# The real compiler, on the no-assembly set. Wired only once it was
# green: a gate that starts red is one people learn to skip.
.PHONY: vc4-compile-gate
vc4-compile-gate:
	@bash build/vc4_compile_gate.sh

# FIRST in `check`, and it needs no build: the repository root holds an
# enumerated set of entries, and anything else is a stray. Ad-hoc work
# in this directory leaves droppings - an .obj from a probe compile, a
# .pytest_cache from a run in the wrong tree - and a stray in the ROOT
# is the one that reads as a top-level project file rather than as
# rubbish. make_lz4.py shipped to the public remote twice that way.
.PHONY: root-hygiene-gate
root-hygiene-gate:
	@bash build/root_hygiene_gate.sh

# A bare "#endif" (no trailing comment) is a regression the split ops
# commits already made once - every #endif must say what it closes.
.PHONY: bare-endif-gate
bare-endif-gate:
	@bash build/bare_endif_gate.sh

# A borrow of s->wscr that stays live across a second borrow of it.
# lz_t_f32 hands the scratch back for every format but F32 and every
# shipped checkpoint is packed, so the second call REPLACES the first
# rather than joining it. Source text only, like the two above it.
.PHONY: wscr-borrow-gate
wscr-borrow-gate:
	@bash build/wscr_borrow_gate.sh

# A "#ifndef X" whose X is already defined upstream (a header, or a
# second #ifndef in the same file) is dead code that lies about the default.
.PHONY: dead-ifndef-gate
dead-ifndef-gate:
	@bash build/dead_ifndef_gate.sh

# A brace pair that opens under one #if and closes under another, or an
# assignment cut in half by one. Every configuration compiles, so the
# only symptom is text that stops meaning what its indentation says.
.PHONY: ppbrace-gate
ppbrace-gate:
	@bash build/ppbrace_gate.sh

# A `float` function reading an int32_t*/int64_t* accumulator (the
# dequant-epilogue family's own shape) that never calls LZ_FCX/LZ_FC is
# invisible to the f32 census - epi_q41's own history. Needs no build,
# runs in milliseconds, so it sits with the other static gates.
.PHONY: f32count-billing-gate
f32count-billing-gate:
	@bash build/f32count_billing_gate.sh

# Needs a model and a built binary, so it sits with the other
# inference-running gates rather than the source-shape ones.
.PHONY: fp-exception-gate
fp-exception-gate:
	@bash build/fp_exception_gate.sh

# Compiles and runs a probe, so it sits with the other build-and-run
# gates rather than the source-shape ones. Skips where __bf16 is absent.
.PHONY: bf16-native-gate
bf16-native-gate:
	@bash build/bf16_native_gate.sh

# LZJsonNode.num must be a double (strtod), not a float (strtof):
# safetensors data_offsets are byte offsets into real checkpoints,
# routinely past float32's 24-bit exact-integer mantissa, and
# src/safetensors.c reads them straight off ->num with no
# lz_json_get_int in the way. Mutation-tested (turns the fix back into
# the bug via sed on the current tree, not a git-pinned copy, so the
# control stays meaningful after this fix is committed), plus a
# config.json regression (the float->double widening must not change
# any of the small values lz_load_config reads) and a Watcom
# compile-only check. No model dir needed - builds its own fixture -
# so it sits with the other fast, build-and-run gates.
.PHONY: json-offset-gate
json-offset-gate:
	@bash build/json_offset_gate.sh

# Every operator src/lz_softfp.c exports, against libgcc's own, under
# qemu - plus an assertion that none escaped the comparison, which is
# the half that matters. Skips without the ARM cross toolchain, so it
# goes through GATE_OK like every other gate that can reach exit 2: a
# bare recipe here stopped `make check` at this point and the forty-odd
# gates after it never ran at all.
.PHONY: softfp-gate
softfp-gate:
	@$(call GATE_OK,build/arm/softfp_gate.sh)

# --bf16-store on/off must give identical logits AND must have changed
# SKIP (exit 2) MUST NOT ABORT `check`, and the five recipes below are
# the ones that can legitimately reach it - four want a model dir and one
# wants wcc386. make has no notion of a third outcome: an exit 2 stops
# the whole run at whichever gate got there first, and everything after
# it never runs. That is the same "a gate that did not run reads as a
# gate that passed" failure these were just wired in to end, so the
# recipes convert 2 to 0 and let 1 through unchanged. The gate still
# PRINTS its SKIP line, which is the part that has to stay visible - a
# silent pass is what would be wrong, not a reported skip.
#
# GATE_OK is one text for all five: five copies of a shell idiom is five
# chances for one of them to swallow a FAIL, and this tree has that
# failure mode on record for copied constants already.
GATE_OK = rc=0; bash $(1) || rc=$$?; if [ "$$rc" = 2 ]; then rc=0; fi; exit $$rc

# bf16-store-gate HAS BEEN REMOVED, along with build/bf16_store_gate.sh.
# `--bf16-store` applies only to safetensors inputs - lz_open's visit_read
# takes an LZStTensor (src/model.c:1450), while a bin carries its own
# LZ_FMT_* codes - and no loadable safetensors model on this tree holds a
# BF16 tensor, so the gate could only ever report SKIP. The BF16
# CONVERSION is still covered twice, by bf16-native-gate and
# bf16-torch-gate; what is uncovered is the loader switch.

# Compressed model == uncompressed model, and the decompressor round
# trips byte for byte. Needs a model dir with a model.bin; SKIPs without
# one. Verified working: 58,629,648 bytes round-tripped and the
# compressed and uncompressed logits identical over 917,504 bytes.
.PHONY: lz4-model-gate
lz4-model-gate:
	@$(call GATE_OK,build/lz4_model_gate.sh "$(MODEL)")

# --ced must not move the numbers, and the skip must actually happen.
# Needs a model dir and the native build; SKIPs without either.
#
# It is in `check` rather than ci.sh-only because the part that costs
# nothing - the two arms, the counter, and the comparison - needs no
# toolchain, and because the expensive half is what makes it worth
# having: it rebuilds the PREVIOUS source from git and compares the off
# arm against it, then builds two mutants and requires both to go red.
# A gate for a knob whose whole claim is "the output does not change" is
# the one kind that cannot be trusted on its own, since a knob wired to
# nothing also produces an unchanged output.
.PHONY: ced-gate
ced-gate:
	@$(call GATE_OK,build/ced_gate.sh "$(MODEL)")

# ced_split/ced_replay_w (E3, the KDA half of CED bounded replay).
# Unlike ced-gate above, this knob's claim is NOT "output unchanged" -
# see build/ced_replay_gate.sh's own header for the three-arm shape and
# why the mutation's discriminator has to be logits, not a layer count.
# Needs a model dir with a linear_attention layer at or above the
# 50/50 split; SKIPs without one.
.PHONY: ced-replay-gate
ced-replay-gate:
	@$(call GATE_OK,build/ced_replay_gate.sh "$(MODEL)")

# S4 MLA low-rank Q/O (mla_q_lora_rank/mla_o_lora_rank). Own script's
# header has the off/on/wrong three-arm shape and why the assertion is
# argmax+KL against `off`, not cmp -s (chaining two matmuls through an
# int8 intermediate is a real, expected precision difference, not a
# regression). Needs a use_subn=0 model dir with a full_attention layer
# and python3 safetensors/numpy; SKIPs without them.
.PHONY: mla-gate
mla-gate:
	@$(call GATE_OK,build/mla_gate.sh "$(MODEL)")

# S2 CSA2 engine-side scalar reference. Own script's header has the
# batch4-vs-batch1 (the strongest signal available without a trained
# checkpoint) and mutation-sensitivity arms. Needs a use_subn=0 model
# dir with a full_attention layer and python3 safetensors/numpy; SKIPs
# without them.
.PHONY: csa2-gate
csa2-gate:
	@$(call GATE_OK,build/csa2_gate.sh "$(MODEL)")

# S1 attn_kv_share, both halves: the loader's alias (model.c) and the
# forward's input swap (forward.c/forward_attn.c). Own script's header has
# the dup/share, adjacent-pair and collapse arms, the bin arms (the
# attn_kv_share word off/on, and the same stream behind a v4 header), and the
# one thing no arm there can see (a PARTIAL swap - only k_proj or only
# v_proj fed the source's input - because logits are the only observable
# this engine has). Needs a model dir this engine can load that holds at
# least two full_attention layers; SKIPs without one, and says why a
# checkpoint it cannot load (kunkun98/kunkun-ce, which do set the key) is
# not a candidate.
.PHONY: kv-share-gate
kv-share-gate:
	@$(call GATE_OK,build/kv_share_gate.sh "$(MODEL)")

# attn_kv_share vs CED's decoder region: lz_kv_share_target must not
# claim a layer at or past ced_split as a target when CED is on -
# report Sec2.2 Eq(1) is unconditional there. Own script's header has
# the three-row fixture and the mutation control (the fix's if-block
# deleted from a scratch model.c, which must redden row 2 only). No
# model dir needed - builds its own fixture - so it sits with the
# other fast, build-and-run gates.
.PHONY: kv-share-ced-boundary-gate
kv-share-ced-boundary-gate:
	@bash build/kv_share_ced_boundary_gate.sh

# CED's engine-side "proj" arm (report Sec2.2 Eq(1)): ced_kv/s->ced_hs.
# Own script's header has the mechanism, the LZ_TAP-based h_S capture
# check, and the mutation control (lz_ced_kv_target forced to
# `return 0`, which must redden exactly the checks that function feeds
# and nothing else). No model dir needed - builds its own tiny
# LZModel fixture - so it sits with the other fast, build-and-run
# gates; one part of the probe optionally checks a real config.json
# (kunkun98's own, which already carries ced_kv="proj") and SKIPS that
# part alone when it is not found.
.PHONY: ced-proj-arm-gate
ced-proj-arm-gate:
	@bash build/ced_proj_arm_gate.sh

# csa2_kv_mode/csa2_kv_source (model.h's own csa2_mode[]/csa2_source[]
# comment): the engine-side schema for report Sec2.3.1's Full/Reindex/
# Reuse cross-layer KV reuse axis. Own script's header has the 6-layer
# fixture and the mutation control (the reuse-source-cannot-itself-be-
# reuse cross-check deleted from a scratch model.c, which must redden
# case 6 only). No model dir needed - builds its own synthetic
# config.json files - so it sits with the other fast, build-and-run
# gates. forward_csa2.c does not read either array yet - a later,
# separate task's cache_idx aliasing and mode dispatch.
.PHONY: csa2-modes-config-gate
csa2-modes-config-gate:
	@bash build/csa2_modes_config_gate.sh

# cache_idx aliasing for CSA2 REINDEX/REUSE-mode layers (Part 5 Task
# 5.3 of the same plan): such a layer must land on its csa2_source's
# cache_idx instead of a fresh nf++ slot, so the compressed-block ring
# a later task's forward_csa2.c dispatch reads is actually shared.
# Own script's header has the 4-layer [FULL, LINEAR, FULL, FULL]
# fixture (a real 2-deep REINDEX/REUSE transitive chain) and the
# mutation control (the aliasing reverted to the old unconditional
# nf++, which must redden exactly the three aliasing checks). No model
# dir needed - builds its own LZModel fixture and calls the real
# lz_state_alloc - so it sits with the other fast, build-and-run
# gates. forward_csa2.c still does not read csa2_mode for dispatch - a
# later, separate task.
.PHONY: csa2-cache-idx-alias-gate
csa2-cache-idx-alias-gate:
	@bash build/csa2_cache_idx_alias_gate.sh

# Full/Reindex/Reuse mode dispatch itself (Part 5 Task 5.4, the last of
# the three): forward_csa2.c now actually reads L->csa2_mode and skips
# steps 2/3/4 per the report Sec2.3.1 table, and csa2_topk_idx/
# csa2_topk_n become per-(ci, token-in-chunk) publish buffers instead
# of a single transient scratch (forward.h's own csa2_topk_n_pub
# comment has the accurate account of csa2_topk_val, its pre-rename,
# never-written predecessor, removed along with the rename rather than
# carried forward unread - not "never existed" the way an earlier
# version of this comment put it). Own script's header has the full
# mutation-by-mutation
# argument (seven independent mutations, not two - the review rounds
# after this task's first landing each added their own). No model dir
# needed - builds its own hand-crafted LZModel/LZRunState fixtures with
# real synthetic weights and calls the real forward_csa2() directly -
# so it sits with the other fast, build-and-run gates. $(call GATE_OK,
# ...), not bare @bash: this script's own SKIP paths (no C compiler, no
# python3) exit 2 exactly like csa2-mode-export-load-gate's below, and
# `make -k check` should treat both the same way rather than one SKIP
# passing and the (currently unreachable in this dev environment, but
# not in every environment this Makefile runs in) other failing it.
.PHONY: csa2-mode-dispatch-gate
csa2-mode-dispatch-gate:
	@$(call GATE_OK,build/csa2_mode_dispatch_gate.sh)

# C1 (fifth review round on this same plan): lz_open_bin called
# lz_csa2_last_writer_valid before its own n_layers bound check, so a
# malformed header (n_layers=200, aliasing csa2_mode[135] onto
# csa2_source[7]) walked a 128-int stack array 108 ints past its end
# instead of being refused. Also carries I4 (same round): a second real-
# byte-stream case through the same real lz_open_bin path, for a per-
# layer csa2_mode corruption well within bounds - the review's own
# complaint that check_bin_validation (csa2-mode-dispatch-gate above)
# never exercises the real loader, only a hand-built LZModelConfig.
# Needs python3 and the training tree at TRAIN_ROOT (default
# E:/LLM/llama98-zh) to build a real exported fixture, and the shipped
# llama98(.exe) CLI - SKIPs without any of the three, same posture
# csa2-mode-export-load-gate already has.
.PHONY: csa2-last-writer-bin-gate
csa2-last-writer-bin-gate:
	@$(call GATE_OK,build/csa2_last_writer_bin_gate.sh)

# forward_csa2.c's causal block-visibility bound - the single
# `int n_vis = (pos + 1) / cm;` line saying which compressed blocks a
# query at position pos may score. The bound is the reference
# implementation's own compress_lens (.prof/dsv41_ref/model.py:564 for
# prefill, :567 for decode, in the TRAINING tree at E:\LLM\llama98-zh -
# "a block becomes visible once the query has passed its last token"),
# which is why it is a per-POSITION formula and never a count carried in
# shared state: a REINDEX layer's ci is aliased onto a source whose own
# call already finished, so a counter read from that state hands an
# early token of the chunk every block its source finalized later.
# Own script's header has the two mutations of that one line (the
# chunk-end count, and one block tight) and each one's expected fallout;
# own probe's header has the boundary and why it reads s->xb2 rather
# than s->attn_out (attn_out is structurally blind to anything but a
# chunk's last token). No model dir needed - builds its own hand-crafted
# LZModel/LZRunState fixture with real synthetic weights and calls the
# real forward_csa2() directly, same posture as csa2-mode-dispatch-gate
# above.
.PHONY: csa2-reindex-causality-gate
csa2-reindex-causality-gate:
	@$(call GATE_OK,build/csa2_reindex_causality_gate.sh)

# The indexer key's RoPE - WHICH dims it rotates and WHICH elements it
# pairs. The V4.1 reference does both its own way: the LAST rd dims of an
# indexer key (`k[..., -rd:]`, .prof/dsv41_ref/model.py in the TRAINING
# tree at E:\LLM\llama98-zh) and ADJACENT elements as one complex number
# (its apply_rotary_emb is `view_as_complex(x.unflatten(-1, (-1, 2)))`
# times a complex frequency), while src/ops_rope.c's lz_rope rotates the
# FIRST rotary_dim dims and pairs base[i] with base[i + rotary_dim/2] -
# which is what the main query and the main KV entries keep, so the
# indexer key and the main KV entry beside it are deliberately not in one
# convention. The two slices are the same dims exactly when
# rotary_dim == indexer_head_dim, which is the identity arm; the two
# pairings are the same at no width but rotary_dim == 2, which is
# degenerate and is not what the criterion rests on. Own script's header
# has the three mutations - the pairing reverted to the host's, the slice
# reverted to the leading dims, and both rows through the indexer's own
# rotation - and each one's expected fallout; own probe's header has why
# it reads the rows out of the csa2 rings rather than calling lz_rope
# itself, and why the pre-rotation row comes from the same model at
# position 0. No model dir needed - builds its own hand-crafted
# LZModel/LZRunState fixture with real synthetic weights and calls the
# real forward_csa2() directly, same posture as csa2-reindex-causality-
# gate above.
.PHONY: csa2-indexer-rope-slice-gate
csa2-indexer-rope-slice-gate:
	@$(call GATE_OK,build/csa2_indexer_rope_slice_gate.sh)

# A-3: the inverse RoPE on CSA2's attention OUTPUT. Step 5 scores and then
# accumulates the SAME ring pointer (Eq 19's shared K=V), and step 3
# rotated every ring entry, so the output carries a rotation the reference
# takes back off at the query's own position (.prof/dsv41_ref/model.py:781,
# `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`). This tree now does it
# by handing lz_rope the conjugate table, so every tier gets the inverse
# without a second kernel.
#
# READ THE GATE'S OWN HEADER BEFORE TRUSTING A GREEN RUN: it proves the
# table IS an inverse (round-trip against a non-conjugated control, an
# oracle that owes nothing to the reference) and that the step is WIRED IN
# (deleting the call moves the attention output). It does NOT prove the
# output-side rotation matches the reference - there is no independent
# oracle for that here, the report gives no formula for it, and the only
# specification available is the same reference file being compared
# against. No model dir needed: hand-built LZModel/LZRunState fixture, real
# forward_csa2(), same posture as csa2-indexer-rope-slice-gate above.
.PHONY: csa2-out-rope-gate
csa2-out-rope-gate:
	@$(call GATE_OK,build/csa2_out_rope_gate.sh)

# A-2: CED's substitution reaches BOTH of the compressor's inputs. Report
# Eq(1) projects `C` and `Z` from the same H_{L/2}; .prof/dsv41_ref/
# model.py's Compressor.forward does `kv, score = self.wkv(x),
# self.wgate(x)` off one `x`; the training side already did it. The engine
# fed `Z` this position's own hidden instead.
#
# The fixture runs at csa2_compress_m = 2 and that is the point of it: at
# m == 1 the softmax over a one-position block is 1 whatever the gate
# logits are, so Z drops out of the pooled entry and the defect is
# invisible - which is why every other fixture in this tree's gates (all
# at m == 1) ran straight past it. csa2-mode-dispatch-gate's check4a asks
# the neighbouring question "does this layer read kv_in at all" and cannot
# see this one either.
#
# Criterion is intrinsic: with kv_in non-NULL the finalized entry must not
# move when the position's own hidden moves, and must move when kv_in
# does. The second arm is the control. Mutation is that one line reverted
# (`kvh` -> `h`), and the gate asserts it reddens the a2 assertion while
# leaving the setup and control arms green.
.PHONY: csa2-z-subst-gate
csa2-z-subst-gate:
	@$(call GATE_OK,build/csa2_z_subst_gate.sh)

# A-4: the compression ratio is PER LAYER. config.json's
# csa2_compress_m_layers (an object keyed by layer index; a layer it does
# not name keeps the csa2_compress_m scalar) is what makes the axis
# expressible - DeepSeek-V4.1-Flash runs m=2 on its 18 encoder layers and
# m=1 on its 20 decoder ones, and this engine carried that assignment as
# one int. The criterion is "two layers, two ratios, both at their own" -
# NOT "either ratio is better", which needs training and has no trained
# CSA2 layer here to compare against (docs/model-selection-constraints.md
# has that boundary and section 7's A-4 entry the landing).
#
# The observable is s->csa2_topk_n_pub's per-(ci, token) k_eff =
# min(topk, n_vis), and n_vis = (pos + 1) / cm - an absolute readout of
# each layer's own ratio, at an offset no part of this change moves, so
# the same probe program runs under the failing and the passing arm.
# check1/check2 move the SCALAR and require that readout to follow: a red
# check3 beside a green check2 is the missing surface, and a red check2 is
# a broken fixture. check4 is the load-time refusals; the mutant (every
# layer consuming layer 0's ratio) must redden check3's two lines and
# nothing else.
#
# In `check` rather than ci.sh-only: gcc and python3 are its whole
# environment, it needs no model dir, and it runs in about half a minute
# (two engine bodies, fixed and mutant) - the ci.sh-only list above is for
# gates that need a model, a Watcom build, or minutes of wall clock.
.PHONY: csa2-compress-m-layers-gate
csa2-compress-m-layers-gate:
	@$(call GATE_OK,build/csa2_compress_m_layers_gate.sh)

# The 64-bit printf modifier. lz_int.h names the type (lz_i64) and its
# modifier (LZ_PRI64) once each. The TYPE half was swept across 213 sites
# and left no gate behind, so the 214th was hand-written and only the vc4
# gate caught it (2270867). The FORMAT half had ZERO call sites tree-wide
# until that same commit, and nothing else can see it: vc4-compile-gate
# compiles and does not run, and a format string is not type-checked, so
# err.c carries 30 of these past every build in this tree.
#
# The observable is the source text, scanned LEXICALLY. Comments are
# stripped before matching: err.c's tables put the match in DATA on a line
# that opens with a comment, err.h's enum puts it in a COMMENT on a line
# that opens with code - a line-keyed rule gets one of those wrong in each
# direction, and both wrong answers still run (one hides sites, the other
# invents them). The scan is a character-level state machine for that
# reason, and the gate's own header says so for whoever tries to simplify
# it.
#
# The baseline is per file, taken at 2270867 (cli_main 46, err 30,
# aboutdlg 1, model 1 = 78): it holds the ratchet without asking anyone to
# clean 78 sites first. It cannot see an exchange WITHIN one file; it does
# see one ACROSS files, because both counts move. That is the whole of the
# weakness.
#
# The control is a synthetic bare modifier that must be reported - a scan
# that stopped matching reports zero everywhere and reads as clean. Six
# mutations were run while writing it: +1 in a baselined file, +3 more,
# +1 in a file with no baseline, and the pair that is the whole point -  a
# modifier inside a comment staying green while the same file with it in a
# string reddens.
#
# In `check` rather than ci.sh-only: its whole environment is python3, it
# needs no model dir, no build and no compiler, and it runs in about 1.6
# seconds (three runs: 1.70 / 1.58 / 1.63).
.PHONY: pri64-scan-gate
pri64-scan-gate:
	@$(call GATE_OK,build/pri64_scan_gate.sh)

# F11 (code review, third round on the same plan): model_walk's
# CSA2_SPECS tensor walk was never gated on L->csa2_mode, so a
# REINDEX/REUSE-mode layer's model.bin - now that the training side's
# export_q8.py (a different repository) correctly writes only the
# tensors that mode owns - fails to load at all. Own script's header
# has the full argument, the mutation (CSA2_SPECS reverted to
# unconditional), and round 4's addition (a real forward pass through
# the shipped CLI, not just loading). Needs python3 and the training
# tree at TRAIN_ROOT (default E:/LLM/llama98-zh) to build its own real
# exported fixtures - SKIPs without either, same posture build/
# csa2_modes_config_gate.sh already has for python3 alone.
.PHONY: csa2-mode-export-load-gate
csa2-mode-export-load-gate:
	@$(call GATE_OK,build/csa2_mode_export_load_gate.sh)

# common/smf.c's lead-in trim: lz_smf_open's first-sounding-note scan has
# to take the MINIMUM tick over every track, not merely the first note
# any track happens to report - a part carried on a later track that
# sounds ahead of an earlier track's own first note used to lose that
# race, so the piece's whole lead-in trim locked onto the wrong (later)
# tick and everything before it played as a burst at tick 0 instead of
# an intro. No model dir needed - a synthetic three-track fixture built
# by hand in build/smf_lead_probe.c, plus an optional real-file
# regression against music/gadameiren_arr.mid (SKIPPED when that
# untracked file is not present).
.PHONY: smf-lead-gate
smf-lead-gate:
	@bash build/smf_lead_gate.sh

# The two-generation allowlist: Qwen3.5 (GDN) and KunMoe (KDA) load, any
# other checkpoint shape is refused by name. Own script's header has the
# four arm shapes (control, three differently-shaped refusals, the
# message table, and a mutant engine built from a copy of src/) and why
# the exit code is not the criterion. Needs four fixture dirs under
# $(MODELS_ROOT), default E:/LLM/models, and one mutant engine build;
# SKIPs without them.
.PHONY: arch-allowlist-gate
arch-allowlist-gate:
	@$(call GATE_OK,build/arch_allowlist_gate.sh "$(MODELS_ROOT)")

# A third opinion on lz_bf16.c, from PyTorch. Needs no model; SKIPs if
# torch is absent, which is why it is behind GATE_OK too.
.PHONY: bf16-torch-gate
bf16-torch-gate:
	@$(call GATE_OK,build/bf16_torch_gate.sh)

# gcc and Watcom must compute the same logits. Needs both builds and a
# model dir; SKIPs without them.
.PHONY: gcc-watcom-output-gate
gcc-watcom-output-gate:
	@$(call GATE_OK,build/gcc_watcom_output_gate.sh "$(MODEL)")

# A path naming a model is a claim about the disk, and it rots silently:
# the gate that wanted it SKIPs, and a skip reads as "not applicable
# here" rather than "this stopped happening". ci.sh's ARM_MODEL had
# rotted, so the x86-vs-ARM bit-identity contract was going unchecked
# with no failure reported.
.PHONY: fixture-exists-gate
fixture-exists-gate:
	@$(call GATE_OK,build/fixture_exists_gate.sh)

# The same question one layer down, and cheap enough for `check` because
# it needs no model and no engine build - two source files and both
# compilers. That is the whole point of it: the end-to-end gate saw this
# class of defect and then declined to run, so the cheap direct check is
# what keeps the expensive one from being the only asker. SKIPs where
# there is no wcc386.
.PHONY: mathf-xcompiler-gate
mathf-xcompiler-gate:
	@$(call GATE_OK,build/mathf_xcompiler_gate.sh)

.PHONY: dedupe-gate
dedupe-gate:
	@bash build/dedupe_gate.sh

.PHONY: mm-alias-gate
mm-alias-gate:
	@bash build/mm_alias_gate.sh

# Symmetric ISA-floor gate, Watcom side - reads watcom-cli's objects, so
# it must run after that target (it does: check lists it last).
.PHONY: isa-floor-watcom-gate
isa-floor-watcom-gate:
	@bash build/isa_floor_watcom_gate.sh

# Every operator carries a path on every tier, asserted rather than
# assumed: each (format x tier) cell is either implemented or registered
# with a reason, and the union across build classes covers everything
# the registry claims. This
# is the only check that can see a whole kernel path disappear - the
# numeric gates cannot, because falling back to the scalar reference is
# bit-identical.
.PHONY: kernel-matrix-gate
kernel-matrix-gate:
	@bash build/kernel_matrix_gate.sh

# The gcc native and serve links guard against stale ELF objects left in
# build/native by a previous WSL-gcc build; MinGW's ld links them into a
# malformed PE with no error of its own. The public Makefile wires this
# in as a prerequisite only when gates.mk is present.
.PHONY: native-gate
native-gate:
	@bash build/native_obj_gate.sh
