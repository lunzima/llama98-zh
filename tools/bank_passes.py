"""Run the five passes that sit above selection, in the order the record
gives them.

THE RECORD. pick_bank.sh says its run "reproduces the SELECTION, which is
the part that was missing; the calibration above it lives in the session's
own tooling", and names the five: fill_key_gaps, then split_from_audit /
balance_from_audit / collapse_velocity_bands against a corpus audit, then
apply_level_trim. That "session's own tooling" is what this file is: the
order and the inputs were not written down anywhere else, and re-running
the selection with a different pass order produced 3468 regions where the
shipped bank has 4621 - a gap of exactly the extra split records.

Each pass lives in tools/ksb_pick.py and is called, not re-derived. This
is only the order and the reporting.

WHAT THE AUDIT HAS TO BE. `split_from_audit`, `balance_from_audit` and
`collapse_velocity_bands` read `key_devs` and `level_db` from the bank
UNDER REPAIR - they are per-bank measurements. Handing them another
bank's audit is not a smaller version of the same run: it moved 18 regions
instead of 1148. `fills` is the same for any bank, and the audit has to
come from corpus_report.py pointed at THIS bank.

    python tools/bank_passes.py IN.ksb ROWS.json OUT.ksb

ROWS is corpus_report's corpus_rows.json, produced from IN.ksb.
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ksb_pick  # noqa: E402


def size(path):
    return os.path.getsize(path)


def main(argv):
    src, rows_p, dst = argv[1], argv[2], argv[3]
    rows = json.load(open(rows_p, encoding='utf-8'))
    shutil.copyfile(src, dst)
    print('%s -> %s' % (os.path.basename(src), os.path.basename(dst)))
    print('  in : %d bytes' % size(dst))

    # fill_key_gaps widens an existing zone's key range rather than adding
    # one, so neither the size nor the region count moves. Its return is
    # the number of key slots it filled.
    print('  fill_key_gaps           -> %s' % ksb_pick.fill_key_gaps(dst))
    print('  split_from_audit        -> %s' % ksb_pick.split_from_audit(dst, rows))
    print('  balance_from_audit      -> %s' % ksb_pick.balance_from_audit(dst, rows))
    print('  collapse_velocity_bands -> %s' % ksb_pick.collapse_velocity_bands(dst, rows))
    print('  apply_level_trim        -> %s' % ksb_pick.apply_level_trim(dst, ksb_pick.LEVEL_TRIM))
    print('  out: %d bytes' % size(dst))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
