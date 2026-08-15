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
ENG := err compat lfn json safetensors model ops forward sampler generate \
       tokenizer unicode chat gbk cpucheck
SRV := net http openai
GUI := main layout localized_strings worker modelload session compat40 \
       settingsdlg aboutdlg splash toolbar inifile caption captionwnd
COM := stream settings command savechat mru chatfile session

# gcc-side full paths.
ENG_SRCS := $(addprefix src/,$(addsuffix .c,$(ENG)))
SRV_SRCS := $(addprefix src/,$(addsuffix .c,$(SRV)))
GUI_SRCS := $(addprefix gui/,$(addsuffix .c,$(GUI)))
COM_SRCS := $(addprefix common/,$(addsuffix .c,$(COM)))

# ----------------------------------------------------------------------
# gcc (dev machine)
# ----------------------------------------------------------------------

# CC ?= gcc is wrong: GNU make's built-in default CC=cc counts as "already
# defined", so ?= is skipped. Use origin to tell "built-in default" from
# "user override" and still allow `make CC=clang`.
ifeq ($(origin CC),default)
CC := gcc
endif

# -ffp-contract=off is a hard requirement for cross-compiler bit-identity
# (iron law two): gcc folds `acc += w*x` into a single-rounding FMA while
# Watcom's x87 does two-rounding multiply-then-add, and the two diverge on
# a measurable fraction of inputs. Cost is noise-level here (the hot path
# is hand-written intrinsics the contraction never touches).
FP_STRICT := -ffp-contract=off
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Isrc $(FP_STRICT)
# Release builds are portable x86-64 (SSE2 baseline), NOT -march=native:
# a native build runs only on the machine that compiled it. The engine
# hot path is hand-written intrinsics anyway, so the portability cost is
# negligible.
X86_64_CFLAGS := -O3 -march=x86-64 -std=c99 -Wall -Wextra -Isrc $(FP_STRICT)

HTTP_LIBS := $(if $(filter Windows_NT,$(OS)),-lwsock32,)

.PHONY: native gui serve
native: llama98
gui: kunkun98-gui
serve: kunkun98-serve

# cli_main.c's -i loop runs on the shared LZSession core (common/session.c),
# a front-end helper deliberately outside $(ENG) - so this target lists it
# explicitly with -Icommon, exactly as the Watcom CLI/DOS builds do.
llama98: $(ENG_SRCS) common/session.c common/stream.c src/cli_attr.c src/cli_main.c
	$(CC) $(X86_64_CFLAGS) -Icommon -o $@ $(ENG_SRCS) common/session.c common/stream.c src/cli_attr.c src/cli_main.c -lm

kunkun98-serve: $(ENG_SRCS) $(SRV_SRCS) src/server_main.c
	$(CC) $(X86_64_CFLAGS) -o $@ $^ -lm $(HTTP_LIBS)

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

kunkun98-gui: $(GUI_SRCS) $(COM_SRCS) $(ENG_SRCS) gui/layout.h gui/localized_strings.h $(GUI_RES)
	$(CC) $(CFLAGS) -Igui -Icommon -mwindows -o $@ $(GUI_SRCS) $(COM_SRCS) $(ENG_SRCS) $(GUI_RES) \
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
WATCOM_BASE := -za99 -otexan -zp4 -D__MMX__=1
CLI_FLAGS := $(WATCOM_BASE) -bt=nt $(WINVER_FLAGS)
DLL_FLAGS := $(CLI_FLAGS)
# -bm: multithreaded runtime. gui/worker.c calls _beginthreadex.
GUI_FLAGS := $(WATCOM_BASE) -bt=nt -bm $(WINVER_FLAGS)
# DOS drops the floor: there is no Windows API to floor, and -we would
# turn every W131 into the first error and hide the real one.
DOS_FLAGS := $(WATCOM_BASE) -bt=dos

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
	    "$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/common_session.obj" common/session.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/common_stream.obj" common/stream.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_main.obj" src/cli_main.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_argv.obj" src/cli_argv.c
	@"$(WCC)" $(CLI_FLAGS) $(INC_CLI) -fo="$(CLI_OBJ)/_attr.obj" src/cli_attr.c
	@{ \
	    echo "format windows nt"; \
	    echo "runtime console"; \
	    echo "option quiet"; \
	    echo "name $(CLI_EXE)"; \
	    echo "libpath $(LIBROOT)/nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(ENG),echo "file $(CLI_OBJ)/$f.obj";) \
	    echo "file $(CLI_OBJ)/common_session.obj"; \
	    echo "file $(CLI_OBJ)/common_stream.obj"; \
	    echo "file $(CLI_OBJ)/_main.obj"; \
	    echo "file $(CLI_OBJ)/_argv.obj"; \
	    echo "file $(CLI_OBJ)/_attr.obj"; \
	    echo "library clib3r, math387r, kernel32, user32, advapi32"; \
	} > "$(CLI_OBJ)/cli.lnk"
	"$(WLINK)" @"$(CLI_OBJ)/cli.lnk"
	@echo "== done: $(CLI_EXE)"

# WHICH DOS EXTENDER (iron law nine: a contested choice is a knob).
#   dos4gw  wstub.exe   loads DOS4GW.EXE at run time - the period default.
#   dos32a  stub32a.exe loads DOS32A.EXE at run time - faster, no DOS4GW
#                       licensing history. A drop-in otherwise.
# Both are LINK-TIME choices (`option stub=`) over the same 32-bit LE
# image. Neither is self-contained; SB.EXE (16-bit, DOSBox-X only) is what
# binds the extender IN - see watcom-dos-single.
LZ_DOS_EXTENDER ?= dos4gw
ifeq ($(LZ_DOS_EXTENDER),dos32a)
  DOS_STUB := stub32a.exe
else
  DOS_STUB := wstub.exe
endif

watcom-dos:
	@mkdir -p $(DOS_OBJ)
	@set -e; for f in $(ENG); do \
	    "$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@"$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/common_session.obj" common/session.c
	@"$(WCC)" $(DOS_FLAGS) $(INC_DOS) -fo="$(DOS_OBJ)/common_stream.obj" common/stream.c
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
	    echo "file $(DOS_OBJ)/common_session.obj"; \
	    echo "file $(DOS_OBJ)/common_stream.obj"; \
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
	    "$(WCC)" $(DLL_FLAGS) $(INC_ENG) -fo="$(DLL_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@{ \
	    echo "format windows nt dll"; \
	    echo "option map=$(DLL_OBJ)/llama98.map"; \
	    echo "name $(DLL_EXE)"; \
	    echo "libpath $(LIBROOT)/nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(ENG),echo "file $(DLL_OBJ)/$f.obj";) \
	    echo "library clib3r, math387r, kernel32, user32, advapi32"; \
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
	    "$(WCC)" $(GUI_FLAGS) $(INC_GUI) -fo="$(GUI_OBJ)/$$f.obj" "src/$$f.c"; \
	done
	@{ \
	    echo "system nt_win"; \
	    echo "option quiet"; \
	    echo "option stack=0x80000"; \
	    echo "libpath $(LIBROOT)/nt"; \
	    echo "libpath $(LIBROOT)"; \
	    $(foreach f,$(GUI),echo "file $(GUI_OBJ)/$f.obj";) \
	    $(foreach f,$(COM),echo "file $(GUI_OBJ)/$(call common_obj,$f).obj";) \
	    $(foreach f,$(ENG),echo "file $(GUI_OBJ)/$f.obj";) \
	    echo "library clib3r, math387r, kernel32, user32, gdi32, comdlg32"; \
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
check: watcom-cli watcom-gui watcom-dll

# Optional single-file DOS packaging (D9). SB.EXE is 16-bit and cannot run
# on 64-bit Windows; the bind step must run inside DOSBox-X.
.PHONY: watcom-dos-single
watcom-dos-single: watcom-dos
	@echo "Run inside DOSBox-X (mount this dir as C:):"
	@echo "  SB /B llama98-dos.exe"
	@echo "SB.EXE binds DOS32A.EXE into the exe (404KB -> 431KB, measured)."

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
