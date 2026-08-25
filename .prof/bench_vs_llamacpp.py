"""llama98 vs llama.cpp on the same Qwen3.5-0.8B, CPU only.

Registered before measuring:

  model         Qwen3.5-0.8B. llama98 reads model.bin (Q8, gs=128,
                798,497,099 B / 772,845,888 params = 1.0332 B/param);
                llama.cpp reads unsloth's Q8_0 GGUF (763.78 MiB /
                752.39 M = 1.0645 B/param). SAME BIT CLASS, and the
                20.46 M difference in the parameter column is the MTP
                block, which the GGUF does not carry and which llama98
                does not touch at --spec 0.
  statistic     decode seconds/token from the SLOPE of wall time against
                -n, not from any single run's average. The slope drops
                model load, tokenizer load and prompt prefill, which are
                constants of the run and would otherwise inflate a short
                measurement and shrink a long one.
  repeats       R runs per point, MINIMUM taken (multiple
                rounds, take the min, and enough rounds).
  control       llama98 prints its own s/token. The slope and that
                number must agree; if they do not, one of the two is
                measuring something other than decode.
  caliber       amd64 dev host, NOT a target machine. llama.cpp loads
                ggml-cpu-haswell.dll (AVX2/FMA); llama98's dev build is
                SSE2. Single thread is the comparable arm - llama98 has
                no threading at all.

Usage: python .prof/bench_vs_llamacpp.py [REPS]
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "llama98.exe"
MODEL = "E:/LLM/models/Qwen3.5-0.8B-run"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

PROMPT = "The capital of France is"
N_LO, N_HI = 32, 160
P_LO, P_HI = 128, 640


def run(args):
    t0 = time.time()
    r = subprocess.run([str(EXE), MODEL] + args, capture_output=True,
                       timeout=3600)
    el = time.time() - t0
    if r.returncode != 0:
        print("FAILED rc=%s\n%s" % (r.returncode,
                                    r.stdout.decode("utf-8", "replace")[-500:]))
        sys.exit(1)
    return el, r.stdout.decode("utf-8", "replace")


def best(args, reps=REPS):
    ts, out = [], ""
    for _ in range(reps):
        el, out = run(args)
        ts.append(el)
    return min(ts), out


print("llama98 decode: slope of wall time vs -n (%d reps, min)" % REPS)
lo, out_lo = best(["--prompt", PROMPT, "-n", str(N_LO), "--temp", "0",
                   "--ctx", "1024"])
hi, out_hi = best(["--prompt", PROMPT, "-n", str(N_HI), "--temp", "0",
                   "--ctx", "1024"])
slope = (hi - lo) / (N_HI - N_LO)
print("  n=%-4d %7.3f s      n=%-4d %7.3f s" % (N_LO, lo, N_HI, hi))
print("  decode  %.4f s/token  = %.2f tok/s" % (slope, 1.0 / slope))

m = re.search(r"\[(\d+) tokens, ([\d.]+) s, ([\d.]+) s/token\]", out_hi)
if m:
    self_rep = float(m.group(3))
    print("  control: llama98's own s/token %.4f vs slope %.4f (%+.1f%%) -> %s"
          % (self_rep, slope, 100 * (self_rep - slope) / slope,
             "agree" if abs(self_rep - slope) / slope < 0.10
             else "DISAGREE - one of them is not decode"))

print()
print("llama98 prefill: slope of wall time vs PROMPT length, generation path")
# Deliberately NOT --tokens/--tokens-file. That is the validation path
# and it walks the prompt one token at a time; generate.c calls
# lz_forward_batch for the prompt (generate.c:1434/1447/1483). Measuring
# the validation path would report an operating point no user reaches -
# and did, on the first run of this script: 1.02x decode, "never enters
# the batched path", which is true of --tokens and false of the engine.
from transformers import AutoTokenizer
_tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
_src = open(r"E:/LLM/corpus_clean/cci3-hq.jsonl", encoding="utf-8")
import json as _json
_txt = []
for _ln in _src:
    try:
        _o = _json.loads(_ln)
    except Exception:
        continue
    _t = _o.get("text") or _o.get("content") or ""
    if _t:
        _txt.append(_t)
    if sum(len(x) for x in _txt) > 8000:
        break
_ids = _tk("\n\n".join(_txt)).input_ids


def prompt_of(n):
    """Text that re-tokenizes to exactly n ids, verified after decode."""
    s = _tk.decode(_ids[:n])
    while len(_tk(s).input_ids) > n:
        s = _tk.decode(_tk(s).input_ids[:n - 1])
    return s, len(_tk(s).input_ids)


pl, nl = prompt_of(P_LO)
ph, nh = prompt_of(P_HI)
plo, _ = best(["--prompt", pl, "-n", "1", "--temp", "0", "--ctx", "1024"])
phi, _ = best(["--prompt", ph, "-n", "1", "--temp", "0", "--ctx", "1024"])
pslope = (phi - plo) / (nh - nl)
print("  p=%-4d %7.3f s      p=%-4d %7.3f s" % (nl, plo, nh, phi))
print("  prefill %.4f s/token  = %.2f tok/s" % (pslope, 1.0 / pslope))
print("  speedup over decode   %.2fx  (LZ_BATCH_MAX=4 is the ceiling)"
      % (slope / pslope))
