# llama98 - KunMoe LLM inference engine, standalone build.
#
# One Makefile drives both toolchains:
#   - gcc (dev machine)   -> native / gui / serve
#   - Open Watcom (target) -> watcom-cli / watcom-gui / watcom-dos / watcom-dll
#
# The source list lives ONCE, in the bare-name variables below. Every
# build - gcc or Watcom, CLI or GUI or DLL - derives its file list from
# them. The original tree kept three hand-written copies of the engine
# list plus a Makefile $(SRC); there is one here.

# The development gates (`check` and friends) live in gates.mk, which is
# local-private and NOT shipped. Included with a leading `-`, so a public
# checkout without the file is unaffected; `make check` simply has no
# rule there.
-include gates.mk

# native-gate (defined in gates.mk) guards the gcc native link against
# stale ELF objects a previous WSL-gcc build left in build/native;
# MinGW's ld links them into a malformed PE with no error of its own.
# Active only when gates.mk is present, so the shipped build never sees
# it.
NATIVE_GATE :=
ifneq ($(wildcard gates.mk),)
NATIVE_GATE := native-gate
endif

# ----------------------------------------------------------------------
# Source lists (single point of truth). Bare names, no .c, no directory.
# ----------------------------------------------------------------------

# The engine body. net/http/openai are the server and are deliberately
# NOT here: they are not in the CLI or DOS binary either, and there are
# no sockets on the target machines.
# lz_bf16 IS IN THE SHARED LIST, unlike lz_softfp, and it is worth being
# exact about why, because it is NOT yet earning its place the way the
# rest of this list does.
#
# NOTHING CALLS IT TODAY. grep finds no lz_bf16_* call anywhere in src/
# outside the file itself; safetensors.c's bf16 loader does its own
# inline <<16 rather than calling lz_bf16_to_f32. So this is 1,488 bytes
# of .text in every shipping binary that no code path reaches. That is
# deliberate but it is a cost, and stating it as "every target runs it"
# - which this comment did - was wrong.
#
# It stays in the list for one concrete reason: being compiled by gcc,
# Watcom, the ARM cross and the x87/MIPS scripts on every build is what
# keeps its portability honest continuously, rather than discovering at
# wiring-up time that it does not survive the C89 floor or Watcom. It
# has a portable C core with an ARM fast path bolted on, unlike
# lz_softfp.c, whose body is #if __arm__ && __GNUC__ and is built on
# ARM targets only.
#
# WHEN IT IS WIRED UP the design is that every target runs this software
# implementation, x86 included - the shipping build has no hardware bf16
# anywhere, so one arithmetic serves all targets and an ARM run stays
# byte-comparable with an x86 one. The compiler's own __bf16 appears
# only in a throwaway probe, as a witness and never a component.
ENG := err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda sampler generate lz_mathf lz_bf16 lz4d \
       tokenizer unicode chat gbk cpucheck
SRV := net http openai
GUI := main layout localized_strings worker modelload session compat40 \
       settingsdlg aboutdlg splash toolbar inifile caption captionwnd
COM := stream settings command savechat mru chatfile session beep
# The subset the COMMAND LINE needs - the gui links all of $(COM), the
# CLI and DOS builds only these. ONE list: each Watcom recipe names
# these files twice (a compile line and a link line) and the gcc recipe
# once, so spelling them out would put the same list in five places and
# adding one file would mean finding all five.
CLI_COM := session stream beep

# src/ops_mmx.c, src/ops_mmx_sse.c and src/ops_sse2.c are deliberately
# NOT in $(ENG) on EITHER toolchain, but for two different reasons as of
# task #30. gcc: they carry their own -mmmx/-msse/-msse2 flags, and gcc
# treats -msse (and -msse2) as license to assume CMOV is available for
# the WHOLE translation unit it is passed to, not just the SSE-touching
# functions in it - so the plain-MMX functions (ops_mmx.c), the
# genuinely-SSE1 ones (ops_mmx_sse.c) and the genuinely-SSE2 ones
# (ops_sse2.c) need different -march there (x86-64 has no such
# distinction, CMOV is baseline, so this
# Makefile compiles all three the same way regardless). Watcom: they
# carry #pragma aux bodies needing .586/.686 codegen, and -Nr/-Ns is a
# ceiling its own per-function directives cannot exceed (task #28's
# finding) - see HIGH_CPU_FLAGS below, which now applies to these three
# files too, on every Watcom target. The ARM cross-build must still
# never compile them: it has no MMX/SSE register file of any kind.
#
# Every gcc ENG file gets LZ_MMX_TU/LZ_SSE2_TU (but not -mmmx/-msse/
# -msse2 themselves) so its dispatch code knows these three files'
# symbols are still part of the link, even though its own object has
# none of the instructions. See src/ops_mmx.h and src/ops_sse2.h.
ENG_MMX := ops_mmx ops_mmx_sse ops_sse2
ENG_MMX_GCC_SRCS := $(addprefix src/,$(addsuffix .c,$(ENG_MMX)))

# gcc-side full paths.
ENG_SRCS := $(addprefix src/,$(addsuffix .c,$(ENG)))
SRV_SRCS := $(addprefix src/,$(addsuffix .c,$(SRV)))
GUI_SRCS := $(addprefix gui/,$(addsuffix .c,$(GUI)))
COM_SRCS := $(addprefix common/,$(addsuffix .c,$(COM)))
CLI_COM_SRCS := $(addprefix common/,$(addsuffix .c,$(CLI_COM)))

# ----------------------------------------------------------------------
# gcc (dev machine)
# ----------------------------------------------------------------------

# CC ?= gcc is wrong: GNU make's built-in default CC=cc counts as "already
# defined", so ?= is skipped. Use origin to tell "built-in default" from
# "user override" and still allow `make CC=clang`.
ifeq ($(origin CC),default)
CC := gcc
endif

# -ffp-contract=off is a hard requirement for cross-compiler
# bit-identity: gcc folds `acc += w*x` into a single-rounding FMA while
# Watcom's x87 does two-rounding multiply-then-add, and the two diverge on
# a measurable fraction of inputs. Cost is noise-level here (the hot path
# is hand-written intrinsics the contraction never touches).
FP_STRICT := -ffp-contract=off
# LZ_MMX_TU=1/LZ_SSE2_TU=1: every gcc TU needs to see these, not just
# ops.c - amax.h, dot.h and q8round.h's dispatch tables are #included
# into it and read the same macros. src/ops_mmx.c, src/ops_mmx_sse.c and
# src/ops_sse2.c are compiled with these flags PLUS -mmmx -msse -msse2,
# never with them removed - see the $(ENG_MMX) rules.
MMX_TU_DEFINE := -DLZ_MMX_TU=1
SSE2_TU_DEFINE := -DLZ_SSE2_TU=1
# Compile-switch hook for the A/B gates: `make EXTRA_DEFS=-DLZ_EPI_FIXED=0`.
# Appended to BOTH flag sets, so a gate that builds a control arm does
# not have to restate -O3 -march=x86-64 $(FP_STRICT) and the two TU
# defines to add one -D. A gate that restates them is a hardcoded
# sibling of this line, free to drift the moment either is edited, and
# the drift shows up as an A/B pair that silently compared two things
# that differed in more than the switch under test.
EXTRA_DEFS ?=
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Isrc $(FP_STRICT) $(MMX_TU_DEFINE) $(SSE2_TU_DEFINE) $(EXTRA_DEFS)
# Release builds are portable x86-64 (SSE2 baseline), NOT -march=native:
# a native build runs only on the machine that compiled it. The engine
# hot path is hand-written intrinsics anyway, so the portability cost is
# negligible.
#
# Per-TU optimisation tiers, not one level for the whole engine. gcov
# function-level call counts pick the hot TUs (see HOT_TU / HOT_COM
# below): they compile at -O2 for speed, everything else defaults to -Os
# for size. The -Os/-O2 split buys the size win without the -O3 cost:
# what -O3 adds over -O2 is loop unrolling and aggressive inlining applied
# to code that is already hand-written intrinsics, and it buys that with
# size - the .text section is the thing a target with an 8 KB I-cache has
# least of. Measured on this tree, llama98.exe -O3 against -O2: 401,616
# against 261,968 bytes of .text, 53.3% larger, and 731,538 against
# 590,881 bytes of file. Anything that wants -O3 can pass it - the flag
# is not forbidden, it is not the default.
# Cold TUs tune for the dev machine (5800X = znver3): at -Os the tune
# only reorders instructions (size is invariant - see the -O2/-O3 note
# above), so this is about scheduling the cold plain-C TUs for the CPU
# that actually runs them, not about .text.
X86_64_CFLAGS := -Os -march=x86-64 -mtune=znver3 -std=c99 -Wall -Wextra -Isrc $(FP_STRICT) $(MMX_TU_DEFINE) $(SSE2_TU_DEFINE) $(EXTRA_DEFS)
# Hot TUs keep -O2 AND swap the tune back to alderlake: gcc's
# long-pipeline branch-prediction optimisation is keyed on alderlake
# only - generic and znver3 do not carry it (in 16.1 or the newer
# releases), so a znver3 hot tune would silently drop that. Derived from
# X86_64_CFLAGS by substitution, so a command-line override still
# reaches every hot TU: an override carries no -Os and no
# -mtune=znver3, so both substs are no-ops and the override's own -O
# level, -mtune and -D flow through unchanged.
X86_64_HOT_CFLAGS := $(subst -Os,-O2,$(subst -mtune=znver3,-mtune=alderlake,$(X86_64_CFLAGS)))

HTTP_LIBS := $(if $(filter Windows_NT,$(OS)),-lwsock32,)

.PHONY: native gui serve
native: llama98
gui: kunkun98-gui
serve: kunkun98-serve

# One .o per $(ENG_MMX) file per gcc flavor below - not shared across
# flavors, the same way each Watcom target below keeps its own object
# dir, so a flag change to one flavor cannot link in an object built
# with another's.
#
# ALL THREE use X86_64_CFLAGS, and gcc_flag_parity_gate.sh holds them
# there. A gui recipe on its own flag variable links the SAME $(ENG_SRCS)
# as the cli, so any divergence ships a window running a differently
# compiled engine from the command line while both claim to be one build.
# Pattern rules, not one rule per file: `-c -o x.o a.c b.c` is invalid
# gcc (multiple sources cannot share one -c output), so each of
# $(ENG_MMX)'s files needs its own compile, and there are now two.
ENG_MMX_GCC_NATIVE_OBJS := $(addprefix build/native/,$(addsuffix .o,$(ENG_MMX)))
ENG_MMX_GCC_GUI_OBJS    := $(addprefix build/gui/,$(addsuffix .o,$(ENG_MMX)))

# Per-TU optimisation tier, gcc side (see X86_64_HOT_CFLAGS). HOT_TU are
# the engine TUs and HOT_COM the common/ TUs the gcov call counts mark
# hot; they compile separately at -O2. Everything else is cold and keeps
# the -Os default, compiled in the link target's one-shot compile exactly
# as before.
# Per-token execution path (gcov call counts, cov_run.sh on kmr20
# -n 128): the epilogue/quantize/matmul/recurrence TUs run 8M-85M
# calls; sched/norm/conv/moe/rope are the next tier; forward_* and
# sampler/generate are per-token dispatch whose call counts are low but
# whose bodies are the whole token loop. json is model-load one-shot
# (jparse_*), NOT hot, and is deliberately cold.
HOT_TU := ops ops_epi ops_matmul ops_quant ops_gdn ops_norm ops_sched \
          ops_conv1d ops_moe ops_rope forward forward_attn forward_kda \
          forward_moe forward_ssm sampler generate
HOT_COM := stream

HOT_ENG_SRCS := $(addprefix src/,$(addsuffix .c,$(HOT_TU)))
HOT_COM_SRCS := $(addprefix common/,$(addsuffix .c,$(HOT_COM)))

# Cold = full minus hot. The link targets compile these in one shot (as
# they always did) and take the hot TUs as prebuilt objects alongside the
# $(ENG_MMX) objects.
COLD_ENG_SRCS     := $(filter-out $(HOT_ENG_SRCS),$(ENG_SRCS))
COLD_CLI_COM_SRCS := $(filter-out $(HOT_COM_SRCS),$(CLI_COM_SRCS))
COLD_COM_SRCS     := $(filter-out $(HOT_COM_SRCS),$(COM_SRCS))

# Hot objects, split native vs gui exactly like $(ENG_MMX) above, so a
# flag change to one flavour cannot link an object built with another's.
HOT_GCC_NATIVE_ENG_OBJS := $(addprefix build/native/,$(addsuffix .o,$(HOT_TU)))
HOT_GCC_NATIVE_COM_OBJS := $(addprefix build/native/,$(addsuffix .o,$(HOT_COM)))
HOT_GCC_GUI_ENG_OBJS    := $(addprefix build/gui/,$(addsuffix .o,$(HOT_TU)))
HOT_GCC_GUI_COM_OBJS    := $(addprefix build/gui/,$(addsuffix .o,$(HOT_COM)))
HOT_GCC_NATIVE_OBJS     := $(HOT_GCC_NATIVE_ENG_OBJS) $(HOT_GCC_NATIVE_COM_OBJS)
HOT_GCC_GUI_OBJS        := $(HOT_GCC_GUI_ENG_OBJS) $(HOT_GCC_GUI_COM_OBJS)

# -MMD -MP ON EVERY OBJECT RULE, and it is not a tidiness change.
#
# Depending on the .c alone is not enough. A header can change the
# LAYOUT of a struct all of them read - LZTensor, LZModel - and make
# would rebuild nothing, because no .c changed. The link then put
# objects compiled against two different definitions of the same struct
# into one binary. That is not a compile error and not a link error; it
# is a program reading the wrong field offsets, which fails somewhere
# else entirely and looks nothing like a build problem.
#
# NOT HYPOTHETICAL - it cost most of a day. Adding rd/rd_ctx to
# LZModel for the compressed-model reader, and a dtype to the tensor
# format, left the engine loading a model and then dying in the forward
# pass with a checksum that looked like arithmetic gone wrong. Every
# diagnosis pointed at the source, because the same files compiled by a
# single gcc command worked - at -O0, at -O2 and at -Os - so it read
# like an optimiser or undefined-behaviour bug. It was stale objects,
# and make was green throughout, correctly believing it had nothing to
# do. `make clean` did not help either: it did not remove build/native.
#
# The -include is a wildcard rather than named files, so a tree with no
# .d yet builds and generates them as it goes. -MP emits a phony target
# per header, which keeps make from failing outright when one is
# renamed or deleted.
DEPFLAGS := -MMD -MP

# The generated dependencies. Wildcard, so this is a no-op on a tree
# that has not built yet and picks them up as they appear.
-include $(wildcard build/native/*.d) $(wildcard build/gui/*.d)


build/native/%.o: src/%.c
	@mkdir -p build/native
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -mmmx -msse -msse2 -c -o $@ $<

build/gui/%.o: src/%.c
	@mkdir -p build/gui
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -mmmx -msse -msse2 -c -o $@ $<

# Static-pattern rules (explicit, so they beat the MMX pattern rules
# above for these files) compile the hot TUs at -O2. They carry no
# -mmmx/-msse/-msse2: those are for $(ENG_MMX) only.
$(HOT_GCC_NATIVE_ENG_OBJS): build/native/%.o: src/%.c
	@mkdir -p build/native
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(HOT_GCC_NATIVE_COM_OBJS): build/native/%.o: common/%.c
	@mkdir -p build/native
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -Icommon -c -o $@ $<

$(HOT_GCC_GUI_ENG_OBJS): build/gui/%.o: src/%.c
	@mkdir -p build/gui
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(HOT_GCC_GUI_COM_OBJS): build/gui/%.o: common/%.c
	@mkdir -p build/gui
	$(CC) $(X86_64_HOT_CFLAGS) $(DEPFLAGS) -Icommon -c -o $@ $<

# cli_main.c's -i loop runs on the shared LZSession core (common/session.c),
# a front-end helper deliberately outside $(ENG) - so this target lists it
# explicitly with -Icommon, exactly as the Watcom CLI/DOS builds do.
#
# build/native/ holds whatever a previous toolchain left there; the
# objects are produced below and named by target, so nothing survives
# between toolchains in the same tree. Cross-compiling from WSL with
# MinGW against stale WSL-gcc objects would link a malformed PE; delete
# build/native before switching toolchains.
llama98: $(NATIVE_GATE) $(COLD_ENG_SRCS) $(COLD_CLI_COM_SRCS) src/cli_attr.c src/cli_main.c $(HOT_GCC_NATIVE_OBJS) $(ENG_MMX_GCC_NATIVE_OBJS)
	$(CC) $(X86_64_CFLAGS) -Icommon -o $@ $(COLD_ENG_SRCS) $(COLD_CLI_COM_SRCS) src/cli_attr.c src/cli_main.c $(HOT_GCC_NATIVE_OBJS) $(ENG_MMX_GCC_NATIVE_OBJS) -lm

kunkun98-serve: $(NATIVE_GATE) $(COLD_ENG_SRCS) $(SRV_SRCS) src/server_main.c $(HOT_GCC_NATIVE_ENG_OBJS) $(ENG_MMX_GCC_NATIVE_OBJS)
	$(CC) $(X86_64_CFLAGS) -o $@ $(COLD_ENG_SRCS) $(SRV_SRCS) src/server_main.c $(HOT_GCC_NATIVE_ENG_OBJS) $(ENG_MMX_GCC_NATIVE_OBJS) -lm $(HTTP_LIBS)

# Resources (icon / splash / about logo / lamps). windres for gcc; the
# Watcom build runs wrc instead (see watcom-gui). Both run from the repo
# root, so gui/kunkun98.rc's asset paths are repo-relative.
GUI_RES := build/gui/kunkun98.res.o
$(GUI_RES): gui/kunkun98.rc gui/resource.h gui/kunkun98.manifest \
            assets/kunkun98-photo-256.ico assets/splash-256.bmp \
            assets/kunkun98-photo-256-side.bmp assets/kunkun98-about-logo-256.bmp \
            assets/lamp-off.bmp assets/lamp-ready.bmp \
            assets/lamp-busy.bmp assets/lamp-error.bmp
	@mkdir -p build/gui
	windres -I. gui/kunkun98.rc -O coff -o $@

kunkun98-gui: $(GUI_SRCS) $(COLD_COM_SRCS) $(COLD_ENG_SRCS) gui/layout.h gui/localized_strings.h $(GUI_RES) $(HOT_GCC_GUI_OBJS) $(ENG_MMX_GCC_GUI_OBJS)
	$(CC) $(X86_64_CFLAGS) -Igui -Icommon -mwindows -o $@ $(GUI_SRCS) $(COLD_COM_SRCS) $(COLD_ENG_SRCS) $(GUI_RES) $(HOT_GCC_GUI_OBJS) $(ENG_MMX_GCC_GUI_OBJS) \
	    -lgdi32 -luser32 -lcomdlg32 -lm -Wl,--stack=524288

# ----------------------------------------------------------------------
# Open Watcom (target compiler)
# ----------------------------------------------------------------------

# Two hosts, two path shapes. Git Bash maps C: to /c and hosts the native
# 64-bit Windows tools in binnt64; WSL maps it to /mnt/c and hosts the
# Linux tools in binl64.
#
# The DEV machine's `make` is a native Windows program (chocolatey GNU
# Make): its wildcard() resolves paths with WINDOWS semantics, so the
# probe must be `C:/WATCOM/...`, NOT the MSYS `/c/...` form (which reads
# as empty and silently falls through to the WSL branch). But the recipe
# runs under Git Bash's sh, which only understands the `/c/...` form, so
# the tool paths handed to sh use that. The two forms must NOT be mixed.
#
# The response files wlink reads are NOT rewritten by MSYS (unlike
# command-line args), so the lib/stub paths inside them must be
# host-shaped: Windows (`C:\WATCOM\...`) on Git Bash, POSIX (`/mnt/c/...`)
# on WSL.
ifeq ($(wildcard C:/WATCOM/binnt64/wcc386.exe),)
  WATCOM_HOST   := wsl
  WATCOM_ROOT   := /mnt/c/WATCOM
  WCC           := /mnt/c/WATCOM/binl64/wcc386
  WLINK         := /mnt/c/WATCOM/binl64/wlink
  WRC           := /mnt/c/WATCOM/binl64/wrc
  WATCOM_NATIVE := /mnt/c/WATCOM
  LIBROOT       := /mnt/c/WATCOM/lib386
  BINW          := /mnt/c/WATCOM/binw
else
  WATCOM_HOST   := gitbash
  WATCOM_ROOT   := /c/WATCOM
  WCC           := /c/WATCOM/binnt64/wcc386.exe
  WLINK         := /c/WATCOM/binnt64/wlink.exe
  WRC           := /c/WATCOM/binnt64/wrc.exe
  WATCOM_NATIVE := C:/WATCOM
  LIBROOT       := C:\WATCOM\lib386
  BINW          := C:\WATCOM\binw
endif
WATCOM_BINDIR := $(dir $(WLINK))

# wlink resolves `system nt_win` (the GUI's windowed-subsystem directive)
# from wlink.lnk, and it finds that file via PATH - not %WATCOM%, which is
# what it looks like. The CLI/DLL/DOS builds spell out `format ...` and do
# not need it; GUI does. WATCOM env is set for the same reason the DLL/GUI
# scripts set it. The `$` before PATH must stay shell-side, hence `$$`.
WLINK_ENV := export WATCOM='$(WATCOM_NATIVE)'; export PATH='$(WATCOM_BINDIR)':$$PATH;

# The Windows API FLOOR, one line for every Watcom Windows build. -we is
# the half that makes it a gate: wcc386 reports a hidden (post-3.51) API
# as warning W131, not an error, so without -we it compiles the call and
# wlink resolves it from its NT4-era import library. WINVER alone would
# not stop it. WIN32_LEAN_AND_MEAN is what makes the 3.51 floor compile
# at all (windows.h -> commdlg.h -> prsht.h -> NMHDR is guarded by
# WINVER>=0x0400). Override LZ_WINVER/LZ_WINNT to raise the floor.
LZ_WINVER ?= 0x0351
LZ_WINNT  ?= 0x0351
WINVER_FLAGS := -we -DWIN32_LEAN_AND_MEAN -DWINVER=$(LZ_WINVER) -D_WIN32_WINNT=$(LZ_WINNT)

# -D__MMX__=1 is NOT optional and its absence is silent: wcc386 does not
# predefine it, every #pragma aux kernel and its dispatch-table entry sit
# inside `#if defined(__MMX__)`, so the whole block vanishes and matmul
# falls back to the scalar reference - while staying bit-identical, which
# is exactly why no consistency gate sees the loss.
#
# -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 (task #30): the gcc precedent's own two
# macros, now also needed by Watcom - src/ops.c's dispatch reaches
# src/ops_mmx.c's/ops_mmx_sse.c's/ops_sse2.c's row-granularity functions
# (lz_p2_rows_mmx et al) through src/ops_mmx.h/ops_sse2.h declarations
# gated on these, exactly the way gcc's LZ_P2_MMX_EXTERN-style
# declarations always were. Harmless on every other ENG file: nothing
# outside ops.c/ops_mmx.c/ops_mmx_sse.c/ops_sse2.c even #includes the two
# headers that test them.
WATCOM_BASE := -za99 -otexan -zp4 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 $(EXTRA_DEFS)

# CPU-GENERATION FLOOR (task #28). -Nr/-Ns select codegen AND calling
# convention together; both targets below use -Nr (register convention -
# args in eax/edx/ebx/ecx), never -Ns, because switching the letter
# would break every hand-written #pragma aux kernel's ABI assumptions,
# not just lower the CPU floor. Verified (not assumed): -3r and -4r
# produce IDENTICAL calling-convention codegen for ordinary C (checked
# via wdis on a two-function probe - only alignment padding differs
# between generations), and wcc386's own undocumented default with no
# -N flag at all was already exactly -4r, byte-for-byte - this project
# has always shipped at the i486 floor for Windows-NT targets, just
# without ever writing that down.
#
# NT_CPU_FLAGS=-4r: Watcom-Windows floor is i486. DOS_CPU_FLAGS=-3r:
# MS-DOS/DOS32A floor is i386, one generation lower - DOS32A's own
# extender has no 486-specific requirement, and #27's EFLAGS.ID guard
# is what makes probing for anything above that floor at runtime safe
# on hardware that does not have it.
#
# ONE FILE STILL CANNOT USE THAT FLOOR: src/cpucheck.c.
# Measured (not assumed): -Nr/-Ns is a HARD CEILING, not merely a
# codegen preference - Watcom's own per-function ".586"/".686"
# assembler directives (used throughout mmintrin.h and this project's
# #pragma aux kernels, e.g. cpucheck.c's own lz_cpuid_ext_edx, task #27's
# guarded CPUID call) cannot exceed it. Explicit -4r fails to compile
# cpucheck.c's EXISTING, already-guarded cpuid call with "Error! E1156:
# Invalid instruction with current CPU setting", even though that call
# already carries its own ".586" directive right there in the source -
# the directive cannot override the file-level ceiling wcc386 was given.
# cpucheck.c needs -5r for its CPUID probe (a .586-class instruction),
# and takes -6r here for one shared tier rather than pinning it exactly.
# ops.c moved to the ordinary floor in task #30: its kernels (amax/dot/
# norm/q8round/p2/gdn1/wsum) now live in ops_mmx.c/ops_mmx_sse.c/ops_sse2.c,
# and its remaining emms sites emit the opcode as raw bytes (mmx_compat.h)
# so they need no .586 ceiling either. The gap this paragraph used to
# describe - Watcom's kernels living inline in ops.c via #include instead
# of their own TU, left over from tasks #18/#22 splitting only gcc - is
# closed.
#
# Checked before relying on this split being SAFE, not just necessary:
# does wcc386, like gcc before #18/#22, autonomously reach for an
# instruction above the floor in ORDINARY C once the ceiling merely
# PERMITS it? No - measured zero cmov/fcmov/fcomi anywhere in ops.c at
# -6r outside the hand-written #pragma aux bodies that explicitly ask
# for them. The ceiling gates what wcc386 is ALLOWED to emit; it never
# changes what wcc386 CHOOSES to emit for code that does not ask for it.
#
# EVERY LOWERED TARGET ASSUMES A 387 IS PRESENT (386+387 or 486DX) -
# this is not incidental, it is the floor: no target here is a no-387
# build. No-387 support waits until the ARM branch is finished; until
# then it is a RESERVATION, not a shipped configuration - tracked as
# isa_floor.py's `x87` class, which stays a reservation rather than an
# enforced floor for exactly that reason. Do not infer soft-float
# coverage from the fixed-point tiers `--fixed all` selects - those are
# tested for cross-compiler bit-identity, never against a target with
# no FPU at all, and presenting them as such would be a claim this
# project has not measured.
NT_CPU_FLAGS := -4r
DOS_CPU_FLAGS := -3r
# cpucheck.c only among the ENG files, both targets - plus the ENG_MMX
# units, which take this set outright. See the comment block above.
# __MMX__ lives here (not WATCOM_BASE) because only the units that may
# emit MMX should define it: mmx_compat.h's Watcom branch gates the full
# <mmintrin.h> intrinsics on it, and a non-kernel TU that defined __MMX__
# would pull that header and fail below the -4r/-3r floor. The ENG_MMX
# units below reuse this same flag set, so they carry __MMX__ too.
HIGH_CPU_FLAGS := -6r -D__MMX__=1

# cpucheck.c needs HIGH_CPU_FLAGS; every other ENG file uses the
# target's ordinary floor.
#
# NOT ops.c, which this said for a long time while the case arm below
# named cpucheck alone. The code was the correct half: src/ops.c's own
# include block says it "must never be compiled with" __MMX__, because
# HIGH_CPU_FLAGS carries -D__MMX__=1 and that makes mmx_compat.h pull
# the full <mmintrin.h> - which does not compile below the -6r floor
# that a non-kernel TU does not get. So the trap here was a comment
# inviting someone to add `ops)` to the arm and "fix the
# inconsistency", which would define __MMX__ for the one file that
# documents needing it undefined.
#
# A shell `case`, not a Make function: the
# loop variable ($$f below) is only known when the recipe's shell runs,
# not at Make's own macro-expansion time, so the per-file choice has to
# be made inside the loop. One case arm, reused by all four `for f in
# $(ENG)` loops below, so the exception lives in exactly one place per
# invocation site instead of drifting across CLI/GUI/DLL/DOS.
ENG_CPU_CASE = case "$$f" in cpucheck) hi=1 ;; *) hi=0 ;; esac

CLI_FLAGS := $(WATCOM_BASE) $(NT_CPU_FLAGS) -bt=nt $(WINVER_FLAGS)
CLI_FLAGS_HIGH := $(WATCOM_BASE) $(HIGH_CPU_FLAGS) -bt=nt $(WINVER_FLAGS)
DLL_FLAGS := $(CLI_FLAGS)
DLL_FLAGS_HIGH := $(CLI_FLAGS_HIGH)
# -bm: multithreaded runtime. gui/worker.c calls _beginthreadex.
GUI_FLAGS := $(WATCOM_BASE) $(NT_CPU_FLAGS) -bt=nt -bm $(WINVER_FLAGS)
GUI_FLAGS_HIGH := $(WATCOM_BASE) $(HIGH_CPU_FLAGS) -bt=nt -bm $(WINVER_FLAGS)
# DOS drops the Windows-API floor: there is no Windows API to floor, and
# -we would turn every W131 into the first error and hide the real one.
# The CPU floor still applies - see DOS_CPU_FLAGS above.
DOS_FLAGS := $(WATCOM_BASE) $(DOS_CPU_FLAGS) -bt=dos
DOS_FLAGS_HIGH := $(WATCOM_BASE) $(HIGH_CPU_FLAGS) -bt=dos

INC_ENG := -I$(WATCOM_ROOT)/h -I$(WATCOM_ROOT)/h/nt -Isrc
INC_CLI := $(INC_ENG) -Icommon
INC_GUI := $(INC_ENG) -Igui -Icommon
INC_DOS := -I$(WATCOM_ROOT)/h -Isrc -Icommon

# Object dirs, one per target, so differently-flagged builds cannot leak
# stale objects into one another.
CLI_OBJ := build/watcom/cli
GUI_OBJ := build/watcom/gui
DOS_OBJ := build/watcom/clidos
DLL_OBJ := build/watcom/dll

# gui/session.c and common/session.c both want session.obj in a flat dir.
# This is the rename: a common/ file becomes common_<name>.obj when the
# bare name is already taken by a gui/ file. Used by BOTH the compile
# loops and the link list so the two cannot drift.
common_obj = $(if $(filter $1,$(GUI)),common_$1,$1)

CLI_EXE := llama98-w.exe
GUI_EXE := kunkun98-w.exe
DOS_EXE := llama98-dos.exe
DLL_EXE := llama98.dll

.PHONY: watcom-cli watcom-gui watcom-dos watcom-dll

watcom-cli:
	@mkdir -p $(CLI_OBJ)
	@set -e; for f in $(ENG); do \
	    $(ENG_CPU_CASE); \
	    if [ "$$hi" = 1 ]; then flags='$(CLI_FLAGS_HIGH)'; else flags='$(CLI_FLAGS)'; fi; \
	    "$(WCC)" $$flags $(INC_CLI) -fo="$(CLI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(ENG_MMX); do \
	    "$(WCC)" $(CLI_FLAGS_HIGH) $(INC_CLI) -fo="$(CLI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(CLI_COM); do \
	    "$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/common_$$f.obj" "common/$$f.c"; \
	done
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_main.obj" src/cli_main.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_argv.obj" src/cli_argv.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_attr.obj" src/cli_attr.c
	@{ \
	    echo "format windows nt"; \
	    echo "runtime console"; \
	    echo "option quiet"; \
	    echo "name $(CLI_EXE)"; \
	    echo "libpath $(LIBROOT)\\nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(ENG),echo "file $(CLI_OBJ)/$f.obj";) \
	    $(foreach f,$(ENG_MMX),echo "file $(CLI_OBJ)/$f.obj";) \
	    $(foreach f,$(CLI_COM),echo "file $(CLI_OBJ)/common_$f.obj";) \
	    echo "file $(CLI_OBJ)/_main.obj"; \
	    echo "file $(CLI_OBJ)/_argv.obj"; \
	    echo "file $(CLI_OBJ)/_attr.obj"; \
	    echo "library $(LIBROOT)\\nt\\clib3r.lib, math387r, kernel32, user32, advapi32"; \
	} > "$(CLI_OBJ)/cli.lnk"
	"$(WLINK)" @"$(CLI_OBJ)/cli.lnk"
	@echo "== done: $(CLI_EXE)"

# WHICH DOS EXTENDER - a contested choice, so it is a knob.
#   dos4gw  wstub.exe   loads DOS4GW.EXE at run time - the period default,
#                       but DOS/4GW OOMs loading a real model, measured,
#                       so dos32a is the default.
#   dos32a  stub32a.exe loads DOS32A.EXE at run time - faster, no DOS4GW
#                       licensing history, and loads the model fine. A
#                       drop-in otherwise; this is the default now.
# Both are LINK-TIME choices (`option stub=`) over the same 32-bit LE
# image. Neither is self-contained; SB.EXE (16-bit, DOSBox-X only) is what
# binds the extender IN - see watcom-dos-single.
LZ_DOS_EXTENDER ?= dos32a
ifeq ($(LZ_DOS_EXTENDER),dos32a)
  DOS_STUB := stub32a.exe
else
  DOS_STUB := wstub.exe
endif

watcom-dos:
	@mkdir -p $(DOS_OBJ)
	@set -e; for f in $(ENG); do \
	    $(ENG_CPU_CASE); \
	    if [ "$$hi" = 1 ]; then flags='$(DOS_FLAGS_HIGH)'; else flags='$(DOS_FLAGS)'; fi; \
	    "$(WCC)" $$flags $(INC_DOS) -fo="$(DOS_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(ENG_MMX); do \
	    "$(WCC)" $(DOS_FLAGS_HIGH) $(INC_DOS) -fo="$(DOS_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(CLI_COM); do \
	    "$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/common_$$f.obj" "common/$$f.c"; \
	done
	@"$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/_main.obj" src/cli_main.c
	@"$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/_argv.obj" src/cli_argv.c
	@"$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/_attr.obj" src/cli_attr.c
	@{ \
	    echo "format os2 le"; \
	    echo "option stub=$(BINW)\\$(DOS_STUB)"; \
	    echo "option quiet"; \
	    echo "name $(DOS_EXE)"; \
	    echo "libpath $(LIBROOT)\\dos"; \
	    echo "libpath $(LIBROOT)"; \
	    echo "option stack=262144"; \
	    $(foreach f,$(ENG),echo "file $(DOS_OBJ)/$f.obj";) \
	    $(foreach f,$(ENG_MMX),echo "file $(DOS_OBJ)/$f.obj";) \
	    $(foreach f,$(CLI_COM),echo "file $(DOS_OBJ)/common_$f.obj";) \
	    echo "file $(DOS_OBJ)/_main.obj"; \
	    echo "file $(DOS_OBJ)/_argv.obj"; \
	    echo "file $(DOS_OBJ)/_attr.obj"; \
	    echo "library clib3r, math387r"; \
	} > "$(DOS_OBJ)/clidos.lnk"
	"$(WLINK)" @"$(DOS_OBJ)/clidos.lnk"
	@echo "== done: $(DOS_EXE)"

# llama98.def is the single source of truth for the DLL export surface;
# gen_exports.py derives the wlink export section from it (function names
# get a trailing underscore, `; data` names a leading underscore).
$(DLL_OBJ)/exports.lnk: llama98.def build/watcom/gen_exports.py
	@mkdir -p $(DLL_OBJ)
	python3 build/watcom/gen_exports.py llama98.def $@

watcom-dll: $(DLL_OBJ)/exports.lnk
	@mkdir -p $(DLL_OBJ)
	@set -e; for f in $(ENG); do \
	    $(ENG_CPU_CASE); \
	    if [ "$$hi" = 1 ]; then flags='$(DLL_FLAGS_HIGH)'; else flags='$(DLL_FLAGS)'; fi; \
	    "$(WCC)" $$flags $(INC_ENG) -fo="$(DLL_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(ENG_MMX); do \
	    "$(WCC)" $(DLL_FLAGS_HIGH) $(INC_ENG) -fo="$(DLL_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@{ \
	    echo "format windows nt dll"; \
	    echo "option map=$(DLL_OBJ)/llama98.map"; \
	    echo "name $(DLL_EXE)"; \
	    echo "libpath $(LIBROOT)\\nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(ENG),echo "file $(DLL_OBJ)/$f.obj";) \
	    $(foreach f,$(ENG_MMX),echo "file $(DLL_OBJ)/$f.obj";) \
	    echo "library $(LIBROOT)\\nt\\clib3r.lib, math387r, kernel32, user32, advapi32"; \
	} > "$(DLL_OBJ)/dll.lnk"
	@cat "$(DLL_OBJ)/exports.lnk" >> "$(DLL_OBJ)/dll.lnk"
	$(WLINK_ENV) "$(WLINK)" @"$(DLL_OBJ)/dll.lnk"
	@echo "== done: $(DLL_EXE)"

watcom-gui:
	@mkdir -p $(GUI_OBJ)
	@set -e; for f in $(GUI); do \
	    "$(WCC)" $(GUI_FLAGS) $(INC_GUI) -fo="$(GUI_OBJ)/$$f.obj" "gui/$$f.c"; \
	done
	# session is the one common/ name that collides with gui/ - mirror
	# `common_obj` above (the link list uses it); keep the two in sync.
	@set -e; for f in $(COM); do \
	    case "$$f" in session) o=common_session ;; *) o="$$f" ;; esac; \
	    "$(WCC)" $(GUI_FLAGS) $(INC_GUI) -fo="$(GUI_OBJ)/$$o.obj" "common/$$f.c"; \
	done
	@set -e; for f in $(ENG); do \
	    $(ENG_CPU_CASE); \
	    if [ "$$hi" = 1 ]; then flags='$(GUI_FLAGS_HIGH)'; else flags='$(GUI_FLAGS)'; fi; \
	    "$(WCC)" $$flags $(INC_GUI) -fo="$(GUI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@set -e; for f in $(ENG_MMX); do \
	    "$(WCC)" $(GUI_FLAGS_HIGH) $(INC_GUI) -fo="$(GUI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@{ \
	    echo "system nt_win"; \
	    echo "option quiet"; \
	    echo "option stack=0x80000"; \
	    echo "libpath $(LIBROOT)\\nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(GUI),echo "file $(GUI_OBJ)/$f.obj";) \
	    $(foreach f,$(COM),echo "file $(GUI_OBJ)/$(call common_obj,$f).obj";) \
	    $(foreach f,$(ENG),echo "file $(GUI_OBJ)/$f.obj";) \
	    $(foreach f,$(ENG_MMX),echo "file $(GUI_OBJ)/$f.obj";) \
	    echo "library $(LIBROOT)\\nt\\clib3r.lib, math387r, kernel32, user32, gdi32, comdlg32"; \
	    echo "name $(GUI_EXE)"; \
	} > "$(GUI_OBJ)/gui.lnk"
	$(WLINK_ENV) "$(WLINK)" @"$(GUI_OBJ)/gui.lnk"
	"$(WRC)" -q -bt=nt -i="$(WATCOM_ROOT)/h" -i="$(WATCOM_ROOT)/h/nt" -i=. gui/kunkun98.rc "$(GUI_EXE)"
	@echo "== done: $(GUI_EXE)"

.PHONY: watcom-dos-single
watcom-dos-single: watcom-dos
	@echo "Run inside DOSBox-X (mount this dir as C:):"
	@echo "  SB /B llama98-dos.exe"
	@echo "SB.EXE binds DOS32A.EXE into the exe (404KB -> 431KB, measured)."

# ----------------------------------------------------------------------
# Ask the Makefile for a variable: `make -s print-ENG`.
#
# The source lists live here and the out-of-tree builds - build/x87 and
# build/arm, which are shell scripts rather than make targets - read
# the source list here via `make -s print-ENG` instead of restating it.
#
# A gate comparing the copies would have caught that. Deleting the
# copies is better: `make -s print-ENG` cannot be out of step with the
# thing it prints.
# ----------------------------------------------------------------------
print-%:
	@echo "$($*)"

# ----------------------------------------------------------------------
# clean
# ----------------------------------------------------------------------

.PHONY: clean
clean:
	rm -f llama98 llama98.exe kunkun98-gui kunkun98-gui.exe \
	      kunkun98-serve kunkun98-serve.exe \
	      $(CLI_EXE) $(GUI_EXE) $(DOS_EXE) $(DLL_EXE)
	# build/native WAS MISSING HERE, and that is why the stale-object
	# failure described at DEPFLAGS survived a `make clean` and read
	# as a source bug rather than a build one.
	rm -rf build/native build/gui build/watcom/cli build/watcom/gui \
	       build/watcom/clidos build/watcom/dll
