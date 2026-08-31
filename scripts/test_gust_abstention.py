#!/usr/bin/env python3
"""Pin the gust-abstention path in veto.py.

    python scripts/test_gust_abstention.py

Separate from test_veto_parity.py on purpose: that test compares against the
real metno.js, and metno.js has NO equivalent of `gust_assessed` — met.no
always publishes wind, so the concept does not exist there. These cases are
therefore Python-only by nature, and mixing them into a parity suite would
blur what that suite proves.

WHY THIS PATH EXISTS
--------------------
ECMWF stops publishing `10fg` beyond +90 h (verified against its live index
2026-08-31: +093h carries `2t` and `tp` and no `10fg`). Before this, the whole
step was dropped and ECMWF fell out of the ensemble from about day 4, even
though it could still see rain and cold.

THE BUG THIS GUARDS AGAINST
---------------------------
`evaluate_hazards` starts with `g = wind_gust_max or 0.0`. Passing a missing
gust through that turns "nobody looked" into a confident 0 km/h — a fabricated
all-clear on the veto that caps a day at 1.0, which is the most severe rung in
the ladder. The same `or 0.0` also silently exempts the wet-cold hypothermia
trap, whose whole purpose is the freezing wet windy day.

So: with gust_assessed=False, no wind veto may fire at ANY speed, the wet-cold
trap must not fire either, and both must be reported as unassessed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from veto import evaluate_hazards  # noqa: E402

failures = 0


def check(name, cond, got=None):
    global failures
    if cond:
        return
    failures += 1
    print(f"FAIL  {name}" + ("" if got is None else f"\n      got: {got!r}"))


def ev(**kw):
    base = dict(precip_sum=0.0, wind_gust_max=0.0, weather_code=1,
                temp_min=12.0, thunder_assessed=True)
    base.update(kw)
    return evaluate_hazards(**base)


# ── A missing gust must never fire a wind veto, at any speed ────────────────
for g in (0.0, 61.9, 62.0, 75.0, 120.0, 250.0):
    v = ev(wind_gust_max=g, gust_assessed=False)
    check(f"no wind veto at {g} km/h when unassessed",
          v.cap == 10.0 and not v.condemns, {"cap": v.cap})
    check(f"{g} km/h unassessed is reported", "gusts" in v.unassessed, v.unassessed)

# The same speeds MUST still veto when the model did look — otherwise this
# change would have disabled the veto rather than made it honest.
check("75 km/h still vetoes when assessed", ev(wind_gust_max=75.0).cap == 1)
check("62 km/h still vetoes when assessed", ev(wind_gust_max=62.0).cap == 3)

# ── The wet-cold trap needs wind, so it must abstain, not silently pass ─────
wet_cold = dict(precip_sum=5.0, temp_min=2.0, wind_gust_max=40.0)
check("wet-cold fires when the gust is known", ev(**wet_cold).cap == 3)
v = ev(**wet_cold, gust_assessed=False)
check("wet-cold does NOT fire on an unknown gust",
      not any(f.label == "Wet, cold and windy" for f in v.factors),
      [f.label for f in v.factors])
check("wet-cold is reported unassessed", "wet-cold" in v.unassessed, v.unassessed)

# ── Everything the model CAN still see must keep working ───────────────────
v = ev(precip_sum=40.0, gust_assessed=False)
check("heavy rain still vetoes without a gust", v.cap == 3, v.cap)
v = ev(temp_min=-12.0, gust_assessed=False)
check("severe frost still vetoes without a gust", v.cap == 3, v.cap)
v = ev(weather_code=75, temp_min=-1.0, gust_assessed=False)
check("heavy snow still vetoes without a gust", v.cap == 3, v.cap)
v = ev(weather_code=97, gust_assessed=False)
check("thunder still vetoes without a gust", v.cap == 2, v.cap)

# ── Abstentions compose, and are not invented ───────────────────────────────
v = ev(precip_sum=2.0, thunder_assessed=False, gust_assessed=False)
check("both abstentions reported together",
      "gusts" in v.unassessed and "thunder" in v.unassessed, v.unassessed)
v = ev()
check("nothing unassessed when everything was looked at", v.unassessed == [], v.unassessed)

# ── Default stays True, so no existing caller changes behaviour ─────────────
check("gust_assessed defaults to True", ev(wind_gust_max=80.0).cap == 1)

print(("FAILED  " if failures else "OK  ") +
      (f"{failures} failure(s)" if failures else "all gust-abstention cases pass"))
sys.exit(1 if failures else 0)
