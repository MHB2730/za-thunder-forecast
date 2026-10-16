#!/usr/bin/env python3
"""Which model run to use, and which days of it are whole.

Shared by the ensemble adapters so that every model answers both questions the
same way. Pure functions with the network passed in, so the tests run offline.

WHY THIS EXISTS — two bugs found together on 2026-10-16
-------------------------------------------------------
1. GFS and ECMWF always took YESTERDAY's 00Z run ("certainly published"). At
   the evening build that run was ~44 h old and three cycles behind what was
   already on the servers: the 15 Oct 20:32Z build used 14 Oct 00Z although
   GFS 15 Oct 12Z had been complete since 16:04Z and ECMWF since 19:34Z. The
   19 Oct Berg gusts it reported were 6-8 km/h; current runs said 73-103.

2. A 120 h run that starts at 00Z ends at 02:00 SAST, so its last "day" held
   ONE 2 am step. `complete` compared the steps decoded with the steps the run
   happened to have for that day (1 == 1), so a calm-night snapshot was voting
   as a whole day — a fabricated all-clear on the wind veto. The same was true
   of the run's first day and of ICON's last (covered only to 14:00 SAST).

The fix for (1) is `pick_run`: the newest cycle whose LAST requested step is
already published. The fix for (2) is `day_complete`: a day is whole only when
the run covers all of it.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

STEP_H = 3
STEPS_PER_DAY = 24 // STEP_H

# How far back pick_run looks before giving up. Four cycles is a day for GFS
# (6-hourly) and two days for ECMWF (only 00Z/12Z reach +120 h). Giving up is
# the right failure: the builder drops the model and the min-models gate
# decides, which beats quietly falling back to a run nobody would choose.
MAX_CYCLES_BACK = 4

# The builder refuses a model whose run is older than this at build time. A
# backstop, not the mechanism: pick_run keeps runs well inside it, so tripping
# it means a provider has stopped publishing and the model should not vote.
MAX_RUN_AGE_H = 36


def candidate_runs(now: dt.datetime, cycles: tuple[int, ...],
                   back: int = MAX_CYCLES_BACK) -> list[dt.datetime]:
    """The `back` most recent nominal run times at or before `now`, newest first."""
    out: list[dt.datetime] = []
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    while len(out) < back:
        for c in sorted(cycles, reverse=True):
            t = day.replace(hour=c)
            if t <= now:
                out.append(t)
                if len(out) == back:
                    break
        day -= dt.timedelta(days=1)
    return out


def pick_run(now: dt.datetime, cycles: tuple[int, ...],
             published: Callable[[dt.datetime], bool],
             *, back: int = MAX_CYCLES_BACK, log=print) -> dt.datetime:
    """Newest run time for which `published(run)` is True.

    `published` should check the run's LAST requested step: providers publish
    steps in order, so the last one existing means the whole ladder does. A
    probe that errors counts as not published — the next older run is tried.
    """
    for run in candidate_runs(now, cycles, back):
        try:
            ok = published(run)
        except Exception as e:  # noqa: BLE001 - a failed probe is a "no"
            log(f"  {run:%Y-%m-%d %HZ}: probe failed ({type(e).__name__})")
            ok = False
        if ok:
            return run
        log(f"  {run:%Y-%m-%d %HZ}: not fully published yet")
    raise RuntimeError(f"no run among the last {back} cycles is fully published")


def day_complete(decoded: int, expected: int) -> bool:
    """A day is whole only when every step of it was decoded AND the run spans it.

    `expected` is how many of the run's steps fall on that SAST day. Fewer than
    STEPS_PER_DAY means the run starts or ends part-way through the day, and a
    daily maximum or sum over part of a day UNDER-states it — the direction
    that hides a hazard. Such a day must read as unknown, not as a vote.
    """
    return expected == STEPS_PER_DAY and decoded == expected
