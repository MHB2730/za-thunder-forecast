#!/usr/bin/env python3
"""Pin the GFS precipitation accumulation arithmetic.

    python scripts/test_gfs_accumulation.py

GFS resets its `tp` accumulator every 6 hours and each file carries both a
since-reset and a since-run window. Observed live from NOMADS on 2026-08-31:

    f003 -> ['0-3',   '0-3']       f012 -> ['6-12',  '0-12']
    f006 -> ['0-6',   '0-6']       f015 -> ['12-15', '0-15']
    f009 -> ['6-9',   '0-9']       f024 -> ['18-24', '0-24']

`0-3` and `0-6` overlap, so summing the reported window at every 3-hourly step
double-counts the first three hours of every block. That error INVENTS rainfall,
and rainfall drives both the `p > 12` veto and the wet-cold hypothermia trap —
it would manufacture hazards rather than hide them, which is the failure
direction that erodes trust in a safety product fastest.

These cases are arithmetic, not network: they encode the observed windows above
as a synthetic accumulation series and assert the decomposition sums correctly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_gfs import _incremental, block_start  # noqa: E402

failures = 0


def check(name: str, got, want) -> None:
    global failures
    if got == want:
        return
    failures += 1
    print(f"FAIL  {name}\n      got {got!r}, want {want!r}")


# ── block_start: which 6-hourly reset a step belongs to ─────────────────────
# A step landing exactly ON a boundary belongs to the block ENDING there —
# f006 carries '0-6', not '6-6'. This is the off-by-one the whole file exists
# to get right.
for step, want in [
    (3, 0), (6, 0),          # f006 is '0-6'
    (9, 6), (12, 6),         # f012 is '6-12'
    (15, 12), (18, 12),
    (21, 18), (24, 18),      # f024 is '18-24'
    (27, 24),
]:
    check(f"block_start({step})", block_start(step), want)


# ── The decomposition ───────────────────────────────────────────────────────
# A synthetic run where exactly 1 mm falls in every 3-hour period. The reported
# value is the since-block-start accumulation, so it ramps 1,2 then resets.
acc = {3: 1.0, 6: 2.0, 9: 1.0, 12: 2.0, 15: 1.0, 18: 2.0, 21: 1.0, 24: 2.0}
for step in sorted(acc):
    check(f"incremental at f{step:03d}", _incremental(step, acc), 1.0)

total = sum(_incremental(s, acc) for s in sorted(acc))
check("24 h total (1 mm per 3 h)", total, 8.0)

# The bug this guards: taking the reported window at face value.
naive = sum(acc.values())
check("naive summing really is wrong (sanity on the test itself)",
      naive == total, False)

# ── Real-world shape: rain only in the second half of a block ───────────────
# Nothing in 6-9, then 4 mm in 9-12. The block accumulation is 0 then 4.
acc2 = {9: 0.0, 12: 4.0}
check("dry first half of block", _incremental(9, acc2), 0.0)
check("wet second half of block", _incremental(12, acc2), 4.0)

# ── A missing earlier step must yield None, not a wrong number ──────────────
# Returning the raw accumulation here would attribute the whole block to one
# 3-hour slot. None propagates to "incomplete", which is the honest answer.
check("missing previous step -> None", _incremental(12, {12: 4.0}), None)

# ── A decode glitch must not produce negative rain ──────────────────────────
# Negative would SUBTRACT from the day's total and could mask a wet day.
check("non-monotonic accumulation clamps at zero",
      _incremental(12, {9: 5.0, 12: 3.0}), 0.0)

# ── First step of the run ───────────────────────────────────────────────────
check("f003 needs no subtraction", _incremental(3, {3: 2.5}), 2.5)

print(("FAILED  " if failures else "OK  ")
      + f"{failures} failure(s)" if failures else "OK  all accumulation cases pass")
sys.exit(1 if failures else 0)
