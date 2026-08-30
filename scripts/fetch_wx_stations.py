#!/usr/bin/env python3
"""Build the HillTrek Weather Station feed — one met.no forecast per station.

WHY THIS EXISTS
---------------
The weather station app (C:\\Dev\\GTP7500\\kiosk\\wx, github MHB2730/hilltrek-gtp7500)
calls api.met.no DIRECTLY FROM THE BROWSER. On one wall tablet that is fine. The
moment it is distributed to the public as a PC widget it is not, and the reason
is not a rate limit but met.no's Terms of Service:

    "All requests must (if possible) include an identifying User Agent-string
    (UA) in the request with the application/domain name."

and, on missing identification, "you risk being blocked without warning". A web
page CANNOT set that header — browsers forbid it. So every copy of a
browser-based widget is an anonymous caller, which is precisely what that rule
exists to stop.

This script is the fix, and it is the pattern met.no themselves recommend: their
ToS advises caching all API responses and says heavy-traffic users may copy data
to their own servers. ONE identified caller fetches, and every widget downloads
a static file. Consequences worth stating:

  * met.no sees one properly-identified client, far inside their 20 req/s cap.
  * The widget makes ZERO calls to any weather provider, so it costs nothing per
    user no matter how many install it.
  * Open-Meteo — whose free tier is NON-COMMERCIAL and which the app's own
    guardrails exclude from a distributed build — is not needed for this half at
    all.

WHY THE RAW met.no BODY IS STORED VERBATIM
-------------------------------------------
metno.js is bundled UNMODIFIED into the app, the website and the widget so all
three agree on what counts as a safe day; its own header says to keep it in
lockstep. parseMetNo(body, lat, lon) reads `body.properties.timeseries`, so
storing met.no's response untouched means the client needs NO change to that
file — only where it gets the bytes. Reshaping the payload here would force a
fork of the one file that must never be forked.

ELEVATION, AND WHY TWO NUMBERS
-------------------------------
met.no returns its own model topography height in geometry.coordinates[2]. For
Champagne Castle — a 3377 m summit — it answers about 1533 m, because global
models smooth terrain. That gap is real and it is the whole reason terrain.js
exists. The feed therefore carries BOTH `ele` (real ground height, null where no
trustworthy figure exists — never invented) and `model_ele` (what met.no
actually modelled), so a consumer can show the caveat rather than present a
valley forecast as a summit one.

FAIL-SAFE CONTRACT — READ BEFORE EDITING
-----------------------------------------
Inherited deliberately from fetch_dwd_thunder.py in this repo, for the same
reason: this feed backs a safety display, and a confident-looking artifact built
from missing data is worse than no artifact.

  * A station appears in the output ONLY if its forecast was fetched and parsed.
    A station that failed is OMITTED, and named in `failed`. It is never emitted
    with an empty or partial forecast that would render as calm weather.
  * If fewer than MIN_OK_FRACTION of stations succeed, the script EXITS NON-ZERO
    and writes NOTHING, so the previously published file stays in place. A
    stale-but-honest artifact beats a fresh lie.
  * Coordinates are re-validated against the Drakensberg box at runtime. The
    gazetteer this list was drawn from holds same-named places countrywide, and
    a naive lookup put "Kamberg" in the Eastern Cape. A station outside the box
    without an explicit `outside_box: true` is a hard error, not a warning.

LICENCE
-------
met.no data is CC BY 4.0 and the attribution is NOT optional — it must reach the
UI, not merely this file. The feed carries it so a consumer cannot ship without
having been handed it.

Usage:
    python scripts/fetch_wx_stations.py --out out/wx-stations.json.gz
    python scripts/fetch_wx_stations.py --stations stations/berg.json --dry-run
"""

import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.met.no/weatherapi/locationforecast/2.0/complete"

# met.no REQUIRES an identifying UA with a contact route. Do not make this
# generic and do not remove the contact address — that is the single thing their
# ToS asks for, and the cost of ignoring it is being blocked without warning.
USER_AGENT = "HilltrekWeatherStation/1.0 (https://hilltrek.co.za; info@hilltrek.co.za)"

LICENCE = "MET Norway (met.no) Locationforecast 2.0 — CC BY 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/documentation"

# Their cap is 20 req/s. This is ~1 req/s — three orders of magnitude of
# headroom, because there is no reason to go fast: the job runs on a schedule
# and nothing is waiting for it.
REQUEST_INTERVAL_S = 1.0

# Below this share of stations succeeding, publish nothing at all.
MIN_OK_FRACTION = 0.8


def log(msg: str) -> None:
    print(msg, flush=True)


FETCH_ERRORS = (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError)


def fetch_station(lat: float, lon: float, timeout: int = 60, attempts: int = 3):
    """Return met.no's parsed JSON body, or None. Never raises to the caller."""
    url = "%s?lat=%.4f&lon=%.4f" % (API, lat, lon)
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 203:
                    # met.no uses 203 to mean "deprecated product version".
                    log("      note: HTTP 203 — met.no flags this endpoint deprecated")
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 429 and 5xx are worth retrying; 4xx otherwise is our own bug.
            if e.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                log("      HTTP %s — giving up" % e.code)
                return None
            back = 5 * attempt
            log("      HTTP %s — backing off %ss" % (e.code, back))
            time.sleep(back)
        except FETCH_ERRORS as e:
            if attempt == attempts:
                log("      %s — giving up" % e)
                return None
            time.sleep(3 * attempt)
    return None


def validate(station: dict, box: dict) -> str:
    """Return an error string, or '' when the station is usable."""
    for key in ("id", "name", "lat", "lon"):
        if station.get(key) in (None, ""):
            return "missing '%s'" % key
    lat, lon = station["lat"], station["lon"]
    if station.get("outside_box"):
        return ""
    if not (box["lat_min"] <= lat <= box["lat_max"]):
        return "lat %.4f outside the box — same-name-different-province?" % lat
    if not (box["lon_min"] <= lon <= box["lon_max"]):
        return "lon %.4f outside the box — same-name-different-province?" % lon
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="stations/berg.json")
    ap.add_argument("--out", default="out/wx-stations.json.gz")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the station list and exit without calling met.no")
    args = ap.parse_args()

    with open(args.stations, encoding="utf-8") as fh:
        spec = json.load(fh)
    box = spec["box"]
    stations = spec["stations"]
    log("stations: %d from %s" % (len(stations), args.stations))

    # Validate the whole list BEFORE any network call. A bad coordinate should
    # fail in a second, not after twenty requests.
    seen_ids = set()
    problems = []
    for st in stations:
        err = validate(st, box)
        if err:
            problems.append("%s: %s" % (st.get("id", "?"), err))
        if st.get("id") in seen_ids:
            problems.append("%s: duplicate id" % st["id"])
        seen_ids.add(st.get("id"))
    if problems:
        for p in problems:
            log("  INVALID  %s" % p)
        log("refusing to run against an invalid station list")
        return 2
    log("  all %d stations validated (box + unique ids)" % len(stations))

    if args.dry_run:
        log("dry run — no requests made")
        return 0

    out_stations, failed = [], []
    for i, st in enumerate(stations, 1):
        log("  [%2d/%d] %s" % (i, len(stations), st["name"]))
        if i > 1:
            time.sleep(REQUEST_INTERVAL_S)
        body = fetch_station(st["lat"], st["lon"])
        if not body:
            failed.append(st["id"])
            continue
        ts = (body.get("properties") or {}).get("timeseries") or []
        if not ts:
            log("      empty timeseries — treating as failure")
            failed.append(st["id"])
            continue

        coords = (body.get("geometry") or {}).get("coordinates") or []
        model_ele = coords[2] if len(coords) > 2 else None

        out_stations.append({
            "id": st["id"],
            "name": st["name"],
            "short": st.get("short") or st["name"],
            "lat": st["lat"],
            "lon": st["lon"],
            "ele": st.get("ele"),            # real height, or null — never invented
            "model_ele": model_ele,          # what met.no actually modelled
            "kind": st.get("kind"),
            "updated_at": (body.get("properties") or {}).get("meta", {}).get("updated_at"),
            "steps": len(ts),
            # Verbatim, so metno.js parseMetNo() needs no change whatsoever.
            "forecast": body,
        })
        log("      ok — %d steps, model elevation %sm" % (len(ts), model_ele))

    ok_fraction = len(out_stations) / float(len(stations)) if stations else 0.0
    log("fetched %d/%d stations (%.0f%%)" % (len(out_stations), len(stations), ok_fraction * 100))

    if ok_fraction < MIN_OK_FRACTION:
        log("BELOW the %.0f%% floor — writing NOTHING so the last good file stands."
            % (MIN_OK_FRACTION * 100))
        log("failed: %s" % ", ".join(failed))
        return 1

    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
                          .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": "MET Norway Locationforecast 2.0",
        "source_url": SOURCE_URL,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        # Spelled out because CC BY makes it an obligation, not a courtesy, and
        # a consumer that never sees it will not display it.
        "attribution_required": True,
        "attribution_text": "Weather data from MET Norway (met.no), CC BY 4.0",
        "station_count": len(out_stations),
        "requested_count": len(stations),
        "failed": failed,
        "stations": out_stations,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with gzip.open(args.out, "wb", compresslevel=9) as fh:
        fh.write(payload)

    log("wrote %s — %d KB raw, %d KB gzipped"
        % (args.out, len(payload) // 1024, os.path.getsize(args.out) // 1024))
    if failed:
        log("NOTE: %d station(s) omitted and named in `failed`: %s"
            % (len(failed), ", ".join(failed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
