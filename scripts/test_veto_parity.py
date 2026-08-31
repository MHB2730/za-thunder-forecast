#!/usr/bin/env python3
"""Pin scripts/veto.py against metno.js evaluateDay — by RUNNING BOTH.

    python scripts/test_veto_parity.py [--metno PATH_TO_metno.js]

`ensemble.js` states the rule this enforces: if the ported ladder and
`metno.js` disagree, the port is wrong and `metno.js` is right. A test that
merely asserted the numbers I read out of the JS would pin my reading, not the
behaviour — so this drives the real `metno.js` in Node over a matrix of days
and compares what the two implementations actually say.

WHAT IS COMPARED, and why not the score
---------------------------------------
`evaluateDay` returns `min(comfort, cap)`, and the comfort half (UTCI, UV, fog)
is deliberately NOT ported — see the module docstring in veto.py. So the score
is not a like-for-like quantity and comparing it would fail for reasons that
are not drift.

What IS like-for-like is the reasoning: every hazard the port raises must be
raised by metno.js, with the same label — and the labels carry the numbers
("Gusts 80 km/h", "-6 °C at dawn"), so matching labels means matching rungs and
matching thresholds. Crucially the BINDING factor must match: that is the one
metno.js sorts to the front and the UI prints first, so a mismatch there is a
day headlined with the wrong hazard, which is the exact defect metno.js's own
comments record fixing.

The port is allowed to raise FEWER factors than metno.js — metno.js also emits
comfort notes (UV, mist, "Beyond detailed range", "Incomplete data") that have
no analogue here. It is never allowed to raise one metno.js does not.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from veto import evaluate_hazards  # noqa: E402

DEFAULT_METNO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "trailtether", "hilltrek-site", "assets", "js", "metno.js",
)

# Every rung of the ladder, its boundaries, and the combinations that made
# metno.js's own comments necessary. `t` is thunder_assessed.
#            precip  gust  code  tmin   t
CASES = [
    # -- nothing fires --------------------------------------------------
    (0.0,   5.0,   1,  12.0, True),
    (0.0,   0.0,   0,  20.0, True),
    # -- thunder, including the codes a (95,96,99) membership test misses --
    (0.0,  10.0,  95,  12.0, True),
    (0.0,  10.0,  96,  12.0, True),
    (0.0,  10.0,  97,  12.0, True),   # heavy thunderstorm with hail
    (0.0,  10.0,  98,  12.0, True),   # thunderstorm with duststorm
    (0.0,  10.0,  99,  12.0, True),
    (0.0,  10.0,  94,  12.0, True),   # just below — must NOT veto
    # -- lightning not assessed (the 7.0 rung) --------------------------
    (2.0,  10.0,   1,  12.0, False),
    (1.0,  10.0,   1,  12.0, False),  # p == 1 is NOT > 1 — boundary
    (0.5,  10.0,   1,  12.0, False),
    (2.0,  10.0,   1,  12.0, True),   # assessed: must NOT fire
    # -- wind boundaries -------------------------------------------------
    (0.0,  61.9,   1,  12.0, True),
    (0.0,  62.0,   1,  12.0, True),
    (0.0,  74.9,   1,  12.0, True),
    (0.0,  75.0,   1,  12.0, True),
    (0.0, 120.0,   1,  12.0, True),
    # -- rain boundaries -------------------------------------------------
    (12.0, 10.0,   1,  12.0, True),   # 12 is NOT > 12
    (12.1, 10.0,   1,  12.0, True),
    (40.0, 10.0,   1,  12.0, True),
    # -- snow -------------------------------------------------------------
    (0.0,  10.0,  71,  -1.0, True),
    (0.0,  10.0,  75,  -1.0, True),   # heavy snow
    (0.0,  10.0,  77,  -1.0, True),
    (0.0,  10.0,  85,  -1.0, True),
    (0.0,  10.0,  86,  -1.0, True),   # heavy snow shower
    (0.0,  10.0,  70,  -1.0, True),   # just outside the snow range
    # -- cold rungs and their boundaries ---------------------------------
    (0.0,  10.0,   1, -10.0, True),
    (0.0,  10.0,   1,  -9.9, True),
    (0.0,  10.0,   1,  -5.0, True),
    (0.0,  10.0,   1,  -4.9, True),
    (0.0,  10.0,   1,  -0.1, True),
    (0.0,  10.0,   1,   0.0, True),   # 0 is NOT < 0 — note only at <= 5
    (0.0,  10.0,   1,   5.0, True),   # "Cold start" note, not a veto
    (0.0,  10.0,   1,   5.1, True),   # silent
    # -- the wet-cold hypothermia trap, incl. the freezing case metno.js
    #    records as previously escaping it -------------------------------
    (2.0,  26.0,   1,  10.0, True),
    (2.0,  26.0,   1,   9.9, True),
    (2.0,  25.0,   1,   9.0, True),   # gust 25 is NOT > 25
    (2.0,  26.0,   1,  -3.0, True),   # freezing AND wet AND windy
    (1.0,  26.0,   1,   9.0, True),   # p == 1 is NOT > 1
    (2.0,  26.0,   1,  10.1, True),   # just above the temp bound
    # -- several vetoes at once: attribution is the point ----------------
    (20.0, 80.0,  95, -12.0, True),
    (20.0, 65.0,   1,  -6.0, True),
    (2.0,  70.0,  75,  -2.0, True),
    (13.0, 63.0,   1,   8.0, True),   # two vetoes both reaching cap 3
    # -- .5 exactly: JS Math.round goes toward +inf, Python round() goes to
    #    even. -9.5 -> -9 in JS, -10 in Python if not handled. Label-only, but
    #    labels are what the user reads. -----------------------------------
    (0.0,  10.0,   1,  -9.5, True),
    (0.0,  10.0,   1,  -4.5, True),
    (0.0,  10.0,   1,  -0.5, True),
    (0.0,  62.5,   1,  12.0, True),
    (12.5, 10.0,   1,  12.0, True),
    # -- missing temperature: must not fire the wet-cold trap ------------
    (5.0,  40.0,   1,  None, True),
    (5.0,  40.0,   1,  None, False),
]

HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
// argv[0]=node, argv[1]=this script, so user args start at [2].
const src = fs.readFileSync(process.argv[2], 'utf8');
const sandbox = {}; sandbox.window = sandbox;
sandbox.console = console; sandbox.fetch = () => Promise.reject(new Error('no net'));
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
const M = sandbox.window.MetNo;
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = cases.map(c => {
  const o = {
    // A comfortable UTCI so the comfort half never bottoms out and the
    // "Incomplete data" note never fires; it does not affect the vetoes.
    utci: 20,
    precipSum: c.precip, windGustMax: c.gust,
    weatherCode: c.code, tempMin: c.tmin,
    uvIndexMax: null, fogMax: null,
    // thunderProb stays null (met.no publishes it only in the Nordics).
    // dwdThunderCode carries the model's own assessment: a number means the
    // model looked, null means it cannot see thunder at all. That is exactly
    // the distinction thunder_assessed encodes on the Python side.
    thunderProb: null,
    dwdThunderCode: c.assessed ? (c.code >= 95 ? c.code : 0) : null,
    dwdThunderHour: null,
  };
  const f = M.scoreFactors(o);
  return {
    score: M.hikeScore(o),
    factors: f.map(x => ({ label: x.label, severity: x.severity, binding: !!x.binding })),
  };
});
process.stdout.write(JSON.stringify(out));
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metno", default=DEFAULT_METNO)
    a = ap.parse_args()

    metno = os.path.abspath(a.metno)
    if not os.path.exists(metno):
        print(f"SKIP: metno.js not found at {metno}")
        print("      (pass --metno /path/to/hilltrek-site/assets/js/metno.js)")
        return 0

    payload = [
        {"precip": p, "gust": g, "code": c, "tmin": t, "assessed": th}
        for (p, g, c, t, th) in CASES
    ]

    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "h.js")
        cp = os.path.join(td, "c.json")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(HARNESS)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        try:
            raw = subprocess.run(
                ["node", hp, metno, cp],
                capture_output=True, text=True, check=True,
                # ⚠️ utf-8 explicitly. Without it Windows decodes node's stdout
                # as cp1252 and every "°C at dawn" label mismatches -- which
                # looks exactly like a drifted threshold and is not.
                encoding="utf-8",
            ).stdout
        except FileNotFoundError:
            print("SKIP: node not available")
            return 0
        except subprocess.CalledProcessError as e:
            print("FAILED to run metno.js:\n" + (e.stderr or "")[:2000])
            return 1

    js = json.loads(raw)
    failures = 0
    for (p, g, code, tmin, assessed), j in zip(CASES, js):
        v = evaluate_hazards(
            precip_sum=p, wind_gust_max=g, weather_code=code,
            temp_min=tmin, thunder_assessed=assessed,
        )
        desc = (f"p={p} g={g} code={code} tmin={tmin} "
                f"assessed={assessed}")

        js_labels = {f["label"] for f in j["factors"]}
        py_labels = {f.label for f in v.factors}
        extra = py_labels - js_labels
        if extra:
            print(f"FAIL  {desc}\n      port raised factors metno.js did not: {sorted(extra)}")
            failures += 1
            continue

        js_binding = next((f["label"] for f in j["factors"] if f["binding"]), None)
        py_binding = v.binding.label if v.binding else None
        # metno.js only marks a veto binding when it actually bit (`cap <
        # comfort`). With a comfortable UTCI and no UV/fog the comfort term is
        # at or near 10, so any veto bites — except where precip has pulled
        # comfort below the cap, which is why a missing JS binding is tolerated
        # while a MISMATCHED one is not.
        if js_binding is not None and js_binding != py_binding:
            print(f"FAIL  {desc}\n      binding differs: metno.js={js_binding!r} port={py_binding!r}")
            failures += 1
            continue
        if js_binding is None and py_binding is not None and j["score"] < 10:
            # Both agree something is wrong with the day; only attribution
            # differs, and metno.js declined to attribute. Not a drift.
            pass

        # Where a veto bit, metno.js's score is capped by it, so the port's cap
        # must not be above the score metno.js published.
        if v.cap < 10 and j["score"] > v.cap + 1e-9:
            print(f"FAIL  {desc}\n      metno.js scored {j['score']} above the port's cap {v.cap}")
            failures += 1

    print(("FAILED  " if failures else "OK  ")
          + f"{len(CASES) - failures}/{len(CASES)} cases agree with metno.js")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
