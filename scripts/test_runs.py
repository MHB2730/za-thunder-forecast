#!/usr/bin/env python3
"""Pin run selection and whole-day completeness — runs.py and its callers.

    python scripts/test_runs.py

THE TWO BUGS THIS GUARDS AGAINST (found together, 2026-10-16)
-------------------------------------------------------------
1. GFS and ECMWF always used yesterday's 00Z run. The 15 Oct 20:32Z build
   shipped 14 Oct 00Z beside ICON 15 Oct 12Z, although both newer runs were
   fully published; its 19 Oct Berg gusts read 6-8 km/h against 73-103 km/h in
   the current runs.
2. That run ended at 02:00 SAST on 19 Oct, so 19 Oct held ONE night-time step,
   and `steps == expected` (1 == 1) marked it complete. It voted.

Offline: the network is replaced, but the adapters' own fetch_daily and the
builder's own main() run, so a fix that lives only in runs.py and is never
called fails here.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_ensemble  # noqa: E402
import model_ecmwf  # noqa: E402
import model_gfs  # noqa: E402
from runs import (  # noqa: E402
    MAX_RUN_AGE_H, STEPS_PER_DAY, candidate_runs, day_complete, pick_run,
)

failures = 0
UTC = dt.timezone.utc


def check(name: str, got, want) -> None:
    global failures
    if got == want:
        return
    failures += 1
    print(f"FAIL  {name}\n      got {got!r}, want {want!r}")


def quiet(*_a, **_k) -> None:
    pass


def t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s).replace(tzinfo=UTC)


# ── candidate_runs ──────────────────────────────────────────────────────────
build = t("2026-10-15T20:32")
check("GFS candidates at the 15 Oct evening build",
      candidate_runs(build, model_gfs.CYCLES),
      [t("2026-10-15T18:00"), t("2026-10-15T12:00"),
       t("2026-10-15T06:00"), t("2026-10-15T00:00")])
check("ECMWF candidates cross midnight backwards",
      candidate_runs(build, model_ecmwf.CYCLES),
      [t("2026-10-15T12:00"), t("2026-10-15T00:00"),
       t("2026-10-14T12:00"), t("2026-10-14T00:00")])
check("a cycle exactly at now is a candidate",
      candidate_runs(t("2026-10-16T12:00"), (0, 12), back=1), [t("2026-10-16T12:00")])

# ── pick_run ────────────────────────────────────────────────────────────────
# What the servers actually held at the 15 Oct 20:32Z build: GFS 12Z complete
# (16:04Z), 18Z not; ECMWF 12Z complete (19:34Z).
gfs_out = {t("2026-10-15T12:00"), t("2026-10-15T06:00"), t("2026-10-15T00:00")}
check("GFS picks the newest complete run, not yesterday's 00Z",
      pick_run(build, model_gfs.CYCLES, lambda r: r in gfs_out, log=quiet),
      t("2026-10-15T12:00"))
check("ECMWF picks 15 Oct 12Z",
      pick_run(build, model_ecmwf.CYCLES, lambda r: r <= t("2026-10-15T12:00"), log=quiet),
      t("2026-10-15T12:00"))


def flaky(r: dt.datetime) -> bool:
    if r == t("2026-10-15T12:00"):
        raise OSError("connection reset")
    return True


check("a probe that errors falls back one cycle instead of failing the model",
      pick_run(build, model_ecmwf.CYCLES, flaky, log=quiet), t("2026-10-15T00:00"))
try:
    pick_run(build, model_gfs.CYCLES, lambda r: False, log=quiet)
    check("nothing published raises", "returned", "raised")
except RuntimeError:
    pass

# ── day_complete ────────────────────────────────────────────────────────────
check("steps per day", STEPS_PER_DAY, 8)
check("a whole day", day_complete(8, 8), True)
check("a whole day with a missing step", day_complete(7, 8), False)
check("THE BUG: a one-step day the run only touches", day_complete(1, 1), False)
check("a run's first, partial day", day_complete(7, 7), False)

# ── the adapters, executed ──────────────────────────────────────────────────
POINTS = [(-29.5833, 29.2833)]  # Sani Pass
LAT, LON = np.array([p[0] for p in POINTS]), np.array([p[1] for p in POINTS])


def grid(v: float):
    return (np.array([v]), LAT, LON)


def gfs_step(url: str) -> int:
    return int(url.split(".pgrb2.0p25.f")[1][:3])


model_gfs._fetch = lambda url: url.encode()
model_gfs._sample = lambda raw, points: {
    f"tp|{model_gfs.block_start(gfs_step(raw.decode()))}-{gfs_step(raw.decode())}": grid(0.0),
    "gust": grid(5.0),
    "2t": grid(285.0),
}

r = model_gfs.fetch_daily(POINTS, run_day="20261014", cycle=0, log=quiet)
days = r["days"]
check("GFS 14 Oct 00Z: dates spanned", sorted(days),
      ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17",
       "2026-10-18", "2026-10-19"])
check("GFS 14 Oct 00Z: 19 Oct holds the one 02:00 SAST step", days["2026-10-19"]["steps"], 1)
check("GFS 14 Oct 00Z: 19 Oct is NOT complete", days["2026-10-19"]["complete"], False)
check("GFS 14 Oct 00Z: 14 Oct starts at 05:00 SAST, NOT complete",
      days["2026-10-14"]["complete"], False)
check("GFS 14 Oct 00Z: the four middle days are complete",
      [days[k]["complete"] for k in ("2026-10-15", "2026-10-16", "2026-10-17", "2026-10-18")],
      [True] * 4)

# Run picked by the adapter itself, against the real clock: anything at least
# six hours old counts as published.
now = dt.datetime.now(UTC)
model_gfs._published = lambda run, step: run <= now - dt.timedelta(hours=6)
r = model_gfs.fetch_daily(POINTS, log=quiet)
want = max(c for c in candidate_runs(now, model_gfs.CYCLES) if c <= now - dt.timedelta(hours=6))
check("GFS fetch_daily uses the picker", r["run"], want.isoformat())

seen: list[int] = []


def ecmwf_step(run_day, cycle, s, points):
    seen.append(s)
    f = {"tp": grid(0.0), "2t": grid(285.0)}
    if s <= model_ecmwf.GUST_HORIZON_H:
        f["10fg"] = grid(5.0)
    return f


model_ecmwf.fetch_step = ecmwf_step
r = model_ecmwf.fetch_daily(POINTS, run_day="20261015", cycle=12, log=quiet)
days = r["days"]
check("ECMWF 15 Oct 12Z: 15 Oct (from 17:00 SAST) is NOT complete",
      days["2026-10-15"]["complete"], False)
check("ECMWF 15 Oct 12Z: 20 Oct (to 14:00 SAST) is NOT complete",
      days["2026-10-20"]["complete"], False)
check("ECMWF 15 Oct 12Z: 16-19 Oct complete",
      [days[k]["complete"] for k in ("2026-10-16", "2026-10-17", "2026-10-18", "2026-10-19")],
      [True] * 4)

model_ecmwf._published = lambda run, step: run <= now - dt.timedelta(hours=9)
r = model_ecmwf.fetch_daily(POINTS, log=quiet)
want = max(c for c in candidate_runs(now, model_ecmwf.CYCLES) if c <= now - dt.timedelta(hours=9))
check("ECMWF fetch_daily uses the picker", r["run"], want.isoformat())

# ── the builder, executed ───────────────────────────────────────────────────
SAST = model_gfs.SAST
today = dt.datetime.now(SAST).date()


def fake(run_age_h: float):
    """An adapter whose run is `run_age_h` old, with yesterday and three days."""
    def fetch_daily(points, **_k):
        n = len(points)
        day = {"precip_sum": [0.0] * n, "wind_gust_max": [10.0] * n,
               "temp_min": [10.0] * n, "steps": 8, "complete": True}
        return {
            "model": "fake", "licence": "-", "source": "-",
            "run": (dt.datetime.now(UTC) - dt.timedelta(hours=run_age_h)).isoformat(),
            "steps_decoded": 40, "steps_requested": 40,
            "thunder_assessed": False, "snow_assessed": False,
            "days": {str(today + dt.timedelta(days=k)): dict(day) for k in range(-1, 3)},
        }
    return type("Fake", (), {"fetch_daily": staticmethod(fetch_daily)})


def run_builder(adapters: dict) -> tuple[int, dict | None]:
    build_ensemble.ADAPTERS = adapters
    build_ensemble.log = quiet
    out = os.path.join(tempfile.mkdtemp(), "wx.json.gz")
    sys.argv = ["build_ensemble.py", "--out", out, "--models", ",".join(adapters)]
    code = build_ensemble.main()
    doc = json.load(gzip.open(out)) if os.path.exists(out) else None
    return code, doc


code, doc = run_builder({"icon": fake(8), "gfs": fake(10), "ecmwf": fake(MAX_RUN_AGE_H + 8)})
check("builder: a stale model is dropped, the fresh two publish", code, 0)
check("builder: models in the file", [m["id"] for m in doc["models"]], ["icon", "gfs"])
check("builder: no day before today", doc["days"][0], str(today))

code, doc = run_builder({"icon": fake(8), "gfs": fake(44), "ecmwf": fake(44)})
check("builder: two stale models leave one, so nothing is published", (code, doc), (1, None))

if failures:
    print(f"{failures} failure(s)")
    sys.exit(1)
print("run selection and whole-day completeness: all cases pass")
