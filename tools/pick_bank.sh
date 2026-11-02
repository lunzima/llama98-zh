#!/bin/sh
# Rebuild banks/picked.ksb from the eleven source DLS banks.
#
# WHY THIS FILE EXISTS. The bank on disk was built by a hand-typed
# command whose only record was the session transcript: the tool, the
# four maps and the eleven DLS files were all still here, and not one
# script in the tree named ksb_pick. That is not "it cannot be run
# again" - it is "nobody can say which parameters produced what is
# shipped", and it makes every question about re-running it a guess.
# This is that command, verbatim, from the run that wrote
# build/tmp/picked_pk3.ksb (2026-10-27).
#
# THE MAPS ARE INPUTS, NOT OUTPUTS OF THIS SCRIPT:
#   tune_map_17.json     tune_pick --render-map   (tuning, seam, gm/FF8 sim)
#   tone_map_full.json   the rendered brightness of gm and FF8 per program
#   air_map2.json        air_census --air-map
#   dirty_map2.json      the rendered dirt census
# They take far longer to produce than the selection itself, so a run
# that needs new ones must say so rather than quietly reuse stale ones.
#
# --retune-frac 0.5 writes half of each region's measured detune back as
# fine-tune. With it absent NO pitch correction is applied at all.
#
# --calibrate settles each program's loudness against gm.dls. --keep 0.3
# (its default) is how much of a difference inside the KEEP_BAND_DB band
# survives as character.
#
# IT TAKES ABOUT THIRTY MINUTES and it does NOT finish the job. The run
# verified on 2026-10-30 took 29m43s and its own --calibrate ended at a
# level deviation of median 1.3 dB, p90 4.1 dB. The shipped bank reaches
# an internal spread of median 2.9 dB, p90 3.8, worst 4.7 because five
# further passes run on top of it - fill_key_gaps, then split_from_audit
# / balance_from_audit / collapse_velocity_bands against a corpus audit,
# then apply_level_trim. This script reproduces the SELECTION, which is
# the part that was missing; the calibration above it lives in the
# session's own tooling.
#
# USAGE
#   sh tools/pick_bank.sh [OUTPUT.ksb]        # default banks/picked.ksb
set -e
cd "$(dirname "$0")/.."
OUT=${1:-banks/picked.ksb}
T=build/tmp

python3 tools/ksb_pick.py --out "$OUT" \
  --kac "$PWD/$T/kac.exe" \
  --render-map "$T/tune_map_17.json" \
  --tone-map "$T/tone_map_full.json" \
  --air-map "$T/air_map2.json" \
  --dirt-map "$T/dirty_map2.json" \
  --retune-frac 0.5 --calibrate \
  build/dls/Chorium.dls \
  build/dls/GeneralUser_GS_144.dls \
  build/dls/Fury.dls \
  build/dls/FluidR3_GM.dls \
  build/dls/SGM-V2.01.dls \
  build/dls/gm.dls \
  build/dls/FF8.dls \
  build/dls/SC-55.dls \
  build/dls/Creative_28MB.dls \
  build/dls/Airfont_340.dls \
  build/dls/Arachno.dls
