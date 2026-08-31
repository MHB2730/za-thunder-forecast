#!/usr/bin/env python3
"""One spatial reduction, expressed in kilometres, shared by every model.

WHY THIS EXISTS
---------------
The ensemble's claim is "how many independent models think this day is
dangerous". That is only true if every model is asked the same question about
the same piece of ground. Nearest-neighbour sampling breaks it, because the
models have different grids:

    ICON global   ~13 km
    GFS 0.25°     ~25 km
    ECMWF 0.25°   ~25 km

Measured on ICON run 2026083100 +6 h, VMAX_10M around Sani Pass: the nearest
cell was 8.18 km away and was THE WINDIEST CELL IN SOUTH AFRICA, while its
eight neighbours ran from 3.4 to 28.0 m/s. ICON reported 102.7 km/h where GFS
said 30.0 and ECMWF 38.1 — enough alone to fire the cap-1 "violent gale" veto.
Cathedral Peak agreed across all three, so it was not a units bug. It is a
1,000 m escarpment inside a 13 km grid box, where exposure changes completely
between adjacent cells.

Rendering that as "ICON says violent gale" would overstate a SAFETY signal, and
the disagreement would be partly resolution rather than meteorology.

WHY A RADIUS IN KILOMETRES, AND NOT k NEAREST CELLS
---------------------------------------------------
`k` nearest cells covers a DIFFERENT AREA on each model — 4 ICON cells is
~676 km², 4 GFS cells is ~2,500 km² — so the ensemble would be measuring the
sampling. A radius in kilometres covers πR² on every model by construction. The
number of samples differs (a finer model resolves the same patch better), the
ground does not.

⚠️ THE COARSEST MODEL SETS THE FLOOR. At 0.25° the spacing is ~25 km, so a
radius much below that returns a single GFS cell and the equal-ground property
is lost exactly where it matters. Hence DEFAULT_RADIUS_KM = 25.

⚠️ AND THAT MAKES THIS A REGIONAL JUDGEMENT, NOT A POINT FORECAST. A 25 km
radius is ~1,960 km² of ground. For "is Saturday a day to be on the escarpment"
that is the right scale and honestly better than a point. It is NOT a summit
forecast, and the UI must not imply it is.

⚠️ THE PRICE, MEASURED — say this out loud rather than let a reader assume
otherwise. Around Sani Pass (2,876 m) a 25 km circle takes in roughly 2,000 m
of altitude range, so the median is representative of the AREA and NOT of the
pass. On run 2026083100 the daily minimum temperature moved from
GFS 3.7 / ECMWF 0.6 / ICON 10.6 °C under nearest-neighbour to
GFS 8.5 / ECMWF 7.2 / ICON 8.5 °C under this reduction: far more consistent
between models, and several degrees WARMER than the pass itself will be.

That is the trade the owner chose on 2026-08-31, and it is the right one for
the ensemble's claim — the models now agree because they are finally being
asked the same question. But it means these numbers must never be presented as
"the forecast at Sani Pass". They are "the forecast around Sani Pass", and a
cold-sensitive reader needs the point forecast, which is met.no's job.

WHAT THE REDUCTION DOES NOT FIX, and should not
-----------------------------------------------
After it, ICON still reports 65.3 km/h of gust at Sani where GFS says 14.4 and
ECMWF 25.9 (down from 102.7 under nearest-neighbour). That residual is NOT an
artefact: ICON at ~13 km resolves terrain-enhanced wind over the escarpment
that a 0.25° model physically cannot represent, and resolution is a property of
the model. Same ground, different ability to see it — which is legitimate
ensemble information, and the reason `describe()` reports the sample count per
model so a reader can weigh a 3-cell vote against an 11-cell one.

THE AGGREGATIONS, and why they differ by field
----------------------------------------------
Identical across models — that is the rule — but not identical across fields,
because the fields answer different questions:

  * precip, gust, temp -> MEDIAN. A representative value for the area. The
    median is what makes this robust to the Sani spike; a spatial max would
    reproduce the very problem this module exists to fix.
  * weather_code -> MAX. Thunder 15 km away is a real hazard to someone on a
    ridge, so "the worst code anywhere nearby" is the safety-correct reduction.
    This is a deliberate asymmetry, not an oversight: it fails toward warning.

The TEMPORAL reduction is separate and lives in the adapters: sum for precip,
max for gust, min for temp, max for code.
"""

from __future__ import annotations

import numpy as np

# Great-circle degrees to km at the equator; good to ~0.3% and the error is far
# below the grid spacings involved.
KM_PER_DEG = 111.32

DEFAULT_RADIUS_KM = 25.0


def cells_within_km(
    points: list[tuple[float, float]],
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> list[np.ndarray]:
    """Indices of every cell whose centre lies within `radius_km` of each point.

    Falls back to the single nearest cell when nothing is in range, so a coarse
    grid or an offshore point still yields a value rather than a gap. A gap
    would demote the day to unknown, which is a heavier answer than this
    situation deserves.
    """
    out: list[np.ndarray] = []
    for la, lo in points:
        dy = (lat - la) * KM_PER_DEG
        dx = (lon - lo) * KM_PER_DEG * np.cos(np.radians(la))
        d2 = dy * dy + dx * dx
        sel = np.nonzero(d2 <= radius_km * radius_km)[0]
        if sel.size == 0:
            sel = np.array([int(np.argmin(d2))])
        out.append(sel)
    return out


def reduce_median(values: np.ndarray, sel: np.ndarray) -> float:
    """Representative value over the patch. Robust to a single extreme cell."""
    return float(np.median(values[sel]))


def reduce_max(values: np.ndarray, sel: np.ndarray) -> float:
    """Worst value over the patch. For hazard CODES only — see the module note."""
    return float(np.max(values[sel]))


def describe(sel_per_point: list[np.ndarray], radius_km: float) -> dict:
    """Sample counts, so the artifact can state how well each model resolved it.

    A model contributing one cell and a model contributing twenty are not
    equally resolved, and a reader comparing them deserves to know which is
    which — particularly when they disagree.
    """
    counts = [int(s.size) for s in sel_per_point]
    return {
        "radius_km": radius_km,
        "cells_min": min(counts) if counts else 0,
        "cells_max": max(counts) if counts else 0,
        "cells_median": float(np.median(counts)) if counts else 0.0,
    }
