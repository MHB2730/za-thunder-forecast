#!/usr/bin/env python3
"""Build the model-agreement forecast — wx-ensemble.json.gz.

    python scripts/build_ensemble.py --out out/wx-ensemble.json.gz
    python scripts/build_ensemble.py --hours 24 --models gfs,ecmwf   # dev

WHAT IT ANSWERS
---------------
Not "what will the weather be" — met.no already answers that, and its verdict
is computed in the browser by `metno.js evaluateDay`, the authority itself.
This answers "how many independent models think this day is dangerous, and do
they agree", by running TrailTether's own hazard-veto ladder over each model.

⚠️ ABSTENTION IS NOT AGREEMENT, AND THE UI MUST NOT CONFLATE THEM.
GFS and ECMWF publish no WMO weather code, so they cannot see thunder or snow —
`veto.py` documents why deriving one would mean inventing a risk index. They
therefore ABSTAIN on those two vetoes rather than voting "clear". A panel that
renders "2 of 3 models say fine" when the third simply could not look is the
same fabricated all-clear that `fetch_dwd_thunder.py` was built to remove. Each
model's `abstains` list travels in the artifact so the UI cannot get this wrong
by omission.

⚠️ THESE ARE AREA FORECASTS, NOT POINT FORECASTS. Every model is reduced over a
25 km radius so the comparison is fair (see spatial.py). Around Sani Pass that
circle spans ~2,000 m of altitude, so the numbers describe the area and read
several degrees warmer than the pass. Say "around", never "at".

FAIL-SAFE CONTRACT — same as the thunder feed
---------------------------------------------
  * A day is only reported for a model if that model decoded every step
    covering it. Otherwise the model is `complete: false` for that day and the
    UI must show it as unknown, not as a clear vote.
  * If fewer than `--min-models` models produce anything, the script EXITS
    NON-ZERO and writes nothing. A stale-but-honest artifact beats a fresh
    file that under-counts the models condemning a day.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_ecmwf  # noqa: E402
import model_gfs  # noqa: E402
import model_icon  # noqa: E402
from spatial import DEFAULT_RADIUS_KM  # noqa: E402
from veto import evaluate_hazards  # noqa: E402

# The app's own defaults, copied from
# trailtether_app/lib/providers/weather_provider.dart `_defaultLocations`.
# ⚠️ Kept identical on purpose: the page and the app must name the same places
# with the same coordinates, or a reader comparing them sees two products.
LOCATIONS = [
    {"name": "Royal Natal", "lat": -28.6833, "lon": 28.9333},
    {"name": "Cathedral Peak", "lat": -28.9167, "lon": 29.1833},
    {"name": "Monk's Cowl", "lat": -29.0333, "lon": 29.4000},
    {"name": "Giant's Castle", "lat": -29.2833, "lon": 29.5167},
    {"name": "Sani Pass", "lat": -29.5833, "lon": 29.2833},
]

ADAPTERS = {
    "icon": model_icon,
    "gfs": model_gfs,
    "ecmwf": model_ecmwf,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _verdict(day: dict, i: int, meta: dict) -> dict:
    """Run the ladder for one model at one location on one day."""
    # Models without a WMO code pass None, which the ladder reads as 0 and
    # which -- combined with thunder_assessed=False -- routes them to the
    # "Lightning not assessed" rung instead of a silent all-clear.
    codes = day.get("weather_code")
    v = evaluate_hazards(
        precip_sum=day["precip_sum"][i],
        wind_gust_max=day["wind_gust_max"][i],
        weather_code=codes[i] if codes else None,
        temp_min=day["temp_min"][i],
        thunder_assessed=bool(meta["thunder_assessed"]),
        # PER-DAY, not per-model: ECMWF stops publishing wind gust beyond
        # +90 h, so the same model votes on wind early in the week and
        # abstains later. Defaults True so models that always carry a gust
        # are unaffected.
        gust_assessed=bool(day.get("gust_assessed", True)),
    )
    b = v.binding
    return {
        "cap": round(v.cap, 2),
        "condemns": v.condemns,
        "binding": b.label if b else None,
        "detail": b.detail if b else None,
        "severity": b.severity if b else None,
        # The raw fields, so the page can show WHY without re-deriving them and
        # so a disagreement can be inspected rather than merely counted.
        # Hazards this model could not look at on THIS day. The page must
        # render these as abstentions; an omitted key would read as a clear
        # vote, which is the fabricated all-clear the feed exists to remove.
        "unassessed": v.unassessed,
        "precip_mm": round(day["precip_sum"][i], 2),
        "gust_kmh": (round(day["wind_gust_max"][i], 1)
                     if day["wind_gust_max"][i] is not None else None),
        "temp_min_c": (round(day["temp_min"][i], 1)
                       if day["temp_min"][i] is not None else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/wx-ensemble.json.gz")
    ap.add_argument("--hours", type=int, default=120)
    ap.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    ap.add_argument("--models", default="icon,gfs,ecmwf")
    ap.add_argument("--min-models", type=int, default=2,
                    help="refuse to publish below this many models (fail-safe)")
    a = ap.parse_args()

    points = [(l["lat"], l["lon"]) for l in LOCATIONS]
    wanted = [m.strip() for m in a.models.split(",") if m.strip()]

    results: dict[str, dict] = {}
    for mid in wanted:
        mod = ADAPTERS.get(mid)
        if mod is None:
            log(f"unknown model {mid!r}")
            return 2
        log(f"— {mid} —")
        try:
            results[mid] = mod.fetch_daily(
                points, hours=a.hours, radius_km=a.radius_km, log=log
            )
        except Exception as e:  # noqa: BLE001
            # One provider being down must not take the whole feed with it; the
            # min-models gate below decides whether what survived is publishable.
            log(f"  {mid} FAILED entirely: {type(e).__name__}: {e}")

    if len(results) < a.min_models:
        log(f"FATAL: only {len(results)} model(s) succeeded (min {a.min_models}). "
            "Refusing to publish — the last good artifact stands.")
        return 1

    days = sorted({d for r in results.values() for d in r["days"]})
    verdicts: list[dict] = []
    for i, loc in enumerate(LOCATIONS):
        by_day: dict[str, dict] = {}
        for day in days:
            per_model: dict[str, dict] = {}
            for mid, r in results.items():
                d = r["days"].get(day)
                if d is None:
                    continue
                if not d.get("complete"):
                    # Present, but not trustworthy for this day. Explicitly
                    # marked rather than omitted: a missing key reads as "no
                    # opinion", and the UI must be able to tell those apart.
                    per_model[mid] = {"complete": False}
                    continue
                per_model[mid] = {"complete": True, **_verdict(d, i, r)}
            voting = [m for m in per_model.values()
                      if m.get("complete") and m.get("condemns") is not None]
            by_day[day] = {
                "models": per_model,
                # The headline number. Counts only models that actually voted.
                "condemning": sum(1 for m in voting if m["condemns"]),
                "voting": len(voting),
            }
        verdicts.append({"location": loc["name"], "days": by_day})

    doc = {
        "schema": 1,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timezone": "Africa/Johannesburg",
        "radius_km": a.radius_km,
        "reduction": ("median of all cells within radius_km; "
                      "max for WMO weather codes"),
        # ⚠️ AREA, not point — see the module docstring.
        "scope": "area",
        "models": [
            {
                "id": mid,
                "model": r["model"],
                "licence": r["licence"],
                "source": r["source"],
                "run": r["run"],
                "steps_decoded": r["steps_decoded"],
                "steps_requested": r["steps_requested"],
                "spatial": r.get("spatial"),
                # What this model can NEVER vote on, whatever the day. This is
                # the FLOOR: a model may additionally abstain on a given day —
                # ECMWF loses wind gust beyond +90 h — and those are carried
                # per-day in each verdict's `unassessed`. Reading only this
                # list would report ECMWF as a wind voter on day 6.
                "abstains": [k for k, ok in (("thunder", r["thunder_assessed"]),
                                             ("snow", r["snow_assessed"])) if not ok],
            }
            for mid, r in results.items()
        ],
        "locations": LOCATIONS,
        "days": days,
        "verdicts": verdicts,
    }

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(a.out, "wb", compresslevel=9) as f:
        f.write(json.dumps(doc, separators=(",", ":")).encode("utf-8"))
    log(f"wrote {a.out}: {os.path.getsize(a.out)/1024:.1f} KiB gz "
        f"({len(results)} models x {len(LOCATIONS)} locations x {len(days)} days)")

    for v in verdicts:
        line = " ".join(
            f"{d[5:]}:{v['days'][d]['condemning']}/{v['days'][d]['voting']}"
            for d in days
        )
        log(f"  {v['location']:16} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
