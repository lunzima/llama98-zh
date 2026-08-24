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

# ----------------------------------------------------------------------
# Source lists (single point of truth). Bare names, no .c, no directory.
# ----------------------------------------------------------------------

# The engine body. net/http/openai are the server and are deliberately
# NOT here: they are not in the CLI or DOS binary either, and there are
# no sockets on the target machines.
ENG := err compat lfn json safetensors model ops ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn fwht forward sampler generate \
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
# (ops_sse2.c) need different -march there (see
# build/x87/build_engine.sh, not this Makefile, for where that actually
# bites - x86-64 has no such distinction, CMOV is baseline, so this
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
# -O2, NOT -O3. What -O3 adds over -O2 here is loop unrolling and
# aggressive inlining applied to code that is already hand-written
# intrinsics, and it buys that with size: the .text section is the thing
# a target with an 8 KB I-cache has least of. Measured on this tree,
# llama98.exe -O3 against -O2: 401,616 against 261,968 bytes of .text,
# 53.3% larger, and 731,538 against 590,881 bytes of file. Anything that
# wants -O3 can pass it - the flag is not forbidden, it is not the
# default.
X86_64_CFLAGS := -O2 -march=x86-64 -std=c99 -Wall -Wextra -Isrc $(FP_STRICT) $(MMX_TU_DEFINE) $(SSE2_TU_DEFINE) $(EXTRA_DEFS)

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

build/native/%.o: src/%.c
	@mkdir -p build/native
	$(CC) $(X86_64_CFLAGS) -mmmx -msse -msse2 -c -o $@ $<

build/gui/%.o: src/%.c
	@mkdir -p build/gui
	$(CC) $(X86_64_CFLAGS) -mmmx -msse -msse2 -c -o $@ $<

# cli_main.c's -i loop runs on the shared LZSession core (common/session.c),
# a front-end helper deliberately outside $(ENG) - so this target lists it
# explicitly with -Icommon, exactly as the Watcom CLI/DOS builds do.
#
# native_obj_gate.sh runs before the link: a Linux/WSL gcc invocation
# that writes into build/native/ leaves ELF .o files there, and MinGW's
# ld links them into a malformed PE with no error of its own. The gate
# checks the file type of whatever is in build/native/*.o right now, so
# a stale ELF object is a build failure here instead of a bad binary
# discovered only when someone tries to run it.
llama98: $(ENG_SRCS) $(CLI_COM_SRCS) src/cli_attr.c src/cli_main.c $(ENG_MMX_GCC_NATIVE_OBJS)
	@bash build/native_obj_gate.sh
	$(CC) $(X86_64_CFLAGS) -Icommon -o $@ $(ENG_SRCS) $(CLI_COM_SRCS) src/cli_attr.c src/cli_main.c $(ENG_MMX_GCC_NATIVE_OBJS) -lm

kunkun98-serve: $(ENG_SRCS) $(SRV_SRCS) src/server_main.c $(ENG_MMX_GCC_NATIVE_OBJS)
	@bash build/native_obj_gate.sh
	$(CC) $(X86_64_CFLAGS) -o $@ $(ENG_SRCS) $(SRV_SRCS) src/server_main.c $(ENG_MMX_GCC_NATIVE_OBJS) -lm $(HTTP_LIBS)

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

kunkun98-gui: $(GUI_SRCS) $(COM_SRCS) $(ENG_SRCS) gui/layout.h gui/localized_strings.h $(GUI_RES) $(ENG_MMX_GCC_GUI_OBJS)
	$(CC) $(X86_64_CFLAGS) -Igui -Icommon -mwindows -o $@ $(GUI_SRCS) $(COM_SRCS) $(ENG_SRCS) $(GUI_RES) $(ENG_MMX_GCC_GUI_OBJS) \
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
# ops.c/cpucheck.c only, both targets - see the comment block above.
# __MMX__ lives here (not WATCOM_BASE) because only the units that may
# emit MMX should define it: mmx_compat.h's Watcom branch gates the full
# <mmintrin.h> intrinsics on it, and a non-kernel TU that defined __MMX__
# would pull that header and fail below the -4r/-3r floor. The ENG_MMX
# units below reuse this same flag set, so they carry __MMX__ too.
HIGH_CPU_FLAGS := -6r -D__MMX__=1

# ops.c and cpucheck.c need HIGH_CPU_FLAGS; every other ENG file uses
# the target's ordinary floor. A shell `case`, not a Make function: the
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

.PHONY: watcom-cli watcom-gui watcom-dos watcom-dll check

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
#                       so it is no longer the default.
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
check: bare-endif-gate dead-ifndef-gate ppbrace-gate f32count-billing-gate dedupe-gate mm-alias-gate c89-floor-gate comment-hygiene-gate vc4-limits-gate vc4-compile-gate dup-body-gate twin-pair-gate gcc-watcom-pair-gate gate-wiring-gate gcc-warning-gate gcc-flag-parity-gate seed-gate watcom-cli watcom-gui watcom-dll isa-floor-watcom-gate kernel-matrix-gate

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

# Where a fixed random seed is needed it is the same one everywhere. A
# grep over build/, so it belongs in `check` rather than ci.sh: two
# harnesses seeded differently produce incomparable numbers that both
# look reproducible, which is the expensive version of this failure.
.PHONY: seed-gate
seed-gate:
	@bash build/seed_gate.sh

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

# A bare "#endif" (no trailing comment) is a regression the split ops
# commits already made once - every #endif must say what it closes.
.PHONY: bare-endif-gate
bare-endif-gate:
	@bash build/bare_endif_gate.sh

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

# Optional single-file DOS packaging (D9). SB.EXE is 16-bit and cannot run
# on 64-bit Windows; the bind step must run inside DOSBox-X.
.PHONY: watcom-dos-single
watcom-dos-single: watcom-dos
	@echo "Run inside DOSBox-X (mount this dir as C:):"
	@echo "  SB /B llama98-dos.exe"
	@echo "SB.EXE binds DOS32A.EXE into the exe (404KB -> 431KB, measured)."

# ----------------------------------------------------------------------
# Ask the Makefile for a variable: `make -s print-ENG`.
#
# The source lists live here and the out-of-tree builds - build/x87 and
# build/arm, which are shell scripts rather than make targets - used to
# restate them, each carrying a comment saying the copy had to stay in
# step. It did not: both scripts said "Same list as the Makefile's ENG"
# and only the ARM one was right, because only the ARM one runs in CI.
# The x87 copy was nine files short and its link had been broken for as
# long as its CI stage had been skipping.
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
	rm -rf build/gui build/watcom/cli build/watcom/gui \
	       build/watcom/clidos build/watcom/dll
