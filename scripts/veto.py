#!/usr/bin/env python3
"""TrailTether's hazard-veto ladder, per model, for the ensemble.

WHAT THIS IS
------------
A verbatim port of the HAZARD half of `evaluateDay` in
`hilltrek-site/assets/js/metno.js` (the trailtether repo). It exists so the
same safety judgement the app makes on met.no can be run against every other
model we can legally use, producing the answer the ensemble reports:

    not "what will the weather be", but
    "how many independent models think this day is dangerous, and do they agree"

⚠️ `metno.js evaluateDay` IS THE AUTHORITY. `ensemble.js` states the rule and it
is repeated here because this file is the second place it can drift: if these
thresholds and THIS file disagree, this file is wrong. `test_veto_parity.py`
pins every rung; if you change a number here and the test still passes, the
test is inadequate, not the change.

WHY ONLY THE HAZARD HALF
------------------------
`evaluateDay` produces `min(comfort, cap)`. The comfort half is UTCI, which
needs hourly temperature, humidity, wind AND radiation per model — a different
order of request entirely, and for most models a different set of files. The
vetoes need only four daily fields, which every global model publishes:
`precipSum`, `windGustMax`, `weatherCode`, `tempMin`.

So met.no stays the single source for the NUMBER on the board. This layer only
says how much to trust it. Emitting a "score" here would invite exactly the
misreading the ensemble exists to prevent — that these are seven competing
forecasts rather than one forecast and six second opinions on its danger.

THE THUNDER SUBTLETY — the reason the ensemble is worth building at all
----------------------------------------------------------------------
`metno.js` carries this warning:

    ⚠️ In South Africa BOTH halves of this test are permanently false, so the
    storm veto never fires here.

met.no publishes `probability_of_thunder` only inside its Nordic model, and
derives its `*andthunder` symbols from that same field. So in South Africa the
veto for "the #1 mountain killer" is dead. That is why `dwd_thunder_code` was
added as a third source, and why an ensemble of models that DO encode thunder
globally is not merely a confidence signal — it restores hazard signal the
primary source structurally cannot provide in this region.

Hence `thunder_assessed`: a model that publishes WMO codes including 95/96/99
has assessed thunder, and a 0 from it means "no thunder forecast". A model that
does not publish thunder at all has NOT assessed it, and must fall through to
the "Lightning not assessed" rung rather than contributing a silent all-clear.
Passing `thunder_assessed=True` for a model that cannot see thunder would
manufacture exactly the fabricated all-clear this whole feed exists to remove.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Mirrors `var SEV_CAUTION = 0, SEV_SERIOUS = 1, SEV_SEVERE = 2` in metno.js.
SEV_CAUTION = 0
SEV_SERIOUS = 1
SEV_SEVERE = 2

THUNDER_CODES = (95, 96, 99)


def js_round(x: float) -> int:
    """JavaScript's `Math.round`, which is NOT Python's `round`.

    Python rounds a half to the nearest EVEN number; JavaScript rounds it
    toward +infinity. They disagree on every exact .5:

        value    Python round()   JS Math.round()
        -9.5     -10              -9
        62.5      62               63
        12.5      12               13

    Labels are user-visible text — "Gusts 62 km/h" against "Gusts 63 km/h" for
    the same forecast — and the ensemble puts the two implementations' output
    side by side, so a half-degree disagreement reads as the models disagreeing
    when it is only the languages. Caught by test_veto_parity.py, which is the
    entire reason that test drives the real metno.js instead of asserting what
    the thresholds ought to be.
    """
    return math.floor(x + 0.5)


def is_snow_code(c: int) -> bool:
    """metno.js: (c >= 71 && c <= 77) || c === 85 || c === 86."""
    return (71 <= c <= 77) or c == 85 or c == 86


@dataclass
class Factor:
    label: str
    detail: str
    severity: int
    binding: bool = False


@dataclass
class Verdict:
    """The hazard ceiling this model puts on the day.

    `cap` is 10 when the model sees no hazard at all — NOT a score. The day's
    score is met.no's, capped by met.no's own ladder; this is one model's
    opinion on the ceiling.
    """

    cap: float
    factors: list[Factor] = field(default_factory=list)
    # Hazards this model could not look at AT ALL for this day. Never empty
    # because nothing was found — empty means everything was assessed. The
    # difference is the whole point: "no gale forecast" and "nobody looked for
    # a gale" must not render the same way.
    unassessed: list[str] = field(default_factory=list)

    @property
    def condemns(self) -> bool:
        """A hazard veto actually fired. The countable unit of the ensemble."""
        return self.cap < 10

    @property
    def binding(self) -> Optional[Factor]:
        for f in self.factors:
            if f.binding:
                return f
        return None


def evaluate_hazards(
    *,
    precip_sum: Optional[float],
    wind_gust_max: Optional[float],
    weather_code: Optional[int],
    temp_min: Optional[float],
    thunder_assessed: bool,
    gust_assessed: bool = True,
) -> Verdict:
    """Run the veto ladder. Order matters — see `veto()` below.

    Every argument is keyword-only on purpose: four numbers of similar
    magnitude in a fixed order is a swap waiting to happen, and swapping
    `precip_sum` and `wind_gust_max` would silently produce a plausible wrong
    answer rather than an error.
    """
    factors: list[Factor] = []
    unassessed: list[str] = []
    cap = 10.0
    binding_at: Optional[int] = None

    def note(label: str, detail: str, severity: int) -> None:
        factors.append(Factor(label, detail, severity))

    def veto(c: float, label: str, detail: str, severity: int) -> None:
        # `c < cap`, strictly. metno.js uses the same comparison, so when two
        # vetoes reach the same ceiling the FIRST one registered stays binding.
        # Using <= would let a later, equally-severe veto steal the attribution
        # and change which hazard gets named as the reason.
        nonlocal cap, binding_at
        if c < cap:
            cap = c
            binding_at = len(factors)
        note(label, detail, severity)

    p = precip_sum or 0.0
    g = wind_gust_max or 0.0
    code = weather_code or 0

    # ── Thunder ────────────────────────────────────────────────────────────
    # For a single model the test collapses to its own WMO code: metno.js also
    # consults thunderProb and dwdThunderCode, but those are met.no-specific
    # and DWD-specific fields with no analogue in an arbitrary model.
    # ⚠️ `>= 95`, not `in THUNDER_CODES`. metno.js tests `o.weatherCode >= 95`
    # and WMO 95-99 are all thunderstorm — 97 (heavy, with hail) and 98
    # (with duststorm) included. THUNDER_CODES exists in fetch_dwd_thunder.py
    # because ICON's WW field only ever emits 95/96/99, which is a fact about
    # that model, not about the ladder. Narrowing to it here would silently
    # drop the veto on the two most violent codes in the range.
    if thunder_assessed and code >= 95:
        veto(2, "Thunderstorms",
             "Lightning on exposed ground — the top killer in the Berg",
             SEV_SEVERE)
    elif not thunder_assessed and p > 1:
        # metno.js: `else if (tp == null && dwd == null && p > 1)`. A wet day
        # where lightning was never assessed caps at 7.0 so it cannot read
        # "Good". CAUTION, not SEVERE — the condition is unknown, not known-bad.
        veto(7, "Lightning not assessed",
             "No thunder forecast is published for this region — on a wet day "
             "treat an afternoon storm as possible", SEV_CAUTION)
    if not thunder_assessed:
        unassessed.append("thunder")

    # ── Wind ───────────────────────────────────────────────────────────────
    # ⚠️ `gust_assessed` exists because `g = wind_gust_max or 0.0` above turns a
    # MISSING gust into a confident 0 km/h — a fabricated all-clear on the veto
    # that caps a day at 1.0. ECMWF stops publishing 10fg beyond +90 h, so this
    # is a live case, not a hypothetical.
    #
    # An unknown gust does NOT get a cap of its own. metno.js only caps for an
    # unassessed hazard when something else makes it dangerous (the "Lightning
    # not assessed" rung fires only on a wet day), and inventing a ceiling for
    # "we did not look" would manufacture a hazard rather than report one. It is
    # recorded as unassessed instead, and the UI must show that.
    if not gust_assessed:
        unassessed.append("gusts")
    elif g >= 75:
        veto(1, f"Gusts {js_round(g)} km/h",
             "Violent gale — you can be blown off your feet on a ridge",
             SEV_SEVERE)
    elif g >= 62:
        veto(3, f"Gusts {js_round(g)} km/h",
             "Gale-force — hard going on any exposed section", SEV_SERIOUS)

    # ── Rain ───────────────────────────────────────────────────────────────
    if p > 12:
        veto(3, f"{js_round(p)} mm rain",
             "Heavy rain — rivers rise fast and crossings turn serious",
             SEV_SERIOUS)

    # ── Snow ───────────────────────────────────────────────────────────────
    if code == 75 or code == 86:
        veto(3, "Heavy snow",
             "Paths vanish, cornices form — turn-back conditions", SEV_SEVERE)
    elif is_snow_code(code):
        veto(4.5, "Snow",
             "Cover hides the path and the footing — navigation risk",
             SEV_SERIOUS)

    # ── Cold ───────────────────────────────────────────────────────────────
    # The wet-cold trap sits INSIDE this guard, exactly as in metno.js. Hoisting
    # it would fire a hypothermia veto on a day with no temperature at all —
    # metno.js has a comment saying precisely this ("`null <= 10` is TRUE in
    # JavaScript ... Do not 'align' the two"). In Python `None <= 10` raises
    # instead of silently passing, but the placement stays faithful so the two
    # implementations can be read side by side.
    tmin = temp_min if isinstance(temp_min, (int, float)) else None
    if tmin is not None:
        if tmin <= -10:
            veto(3, f"{js_round(tmin)} °C at dawn",
                 "Severe frost — frostbite risk on any exposed skin", SEV_SEVERE)
        elif tmin <= -5:
            veto(4.5, f"{js_round(tmin)} °C at dawn",
                 "Hard freeze — ice on rock, real hypothermia exposure",
                 SEV_SERIOUS)
        elif tmin < 0:
            veto(7, f"{js_round(tmin)} °C at dawn",
                 "Sub-zero overnight — frost, and cold hands early on",
                 SEV_CAUTION)
        elif tmin <= 5:
            # A note, NOT a veto. A 2 °C dawn under a 14 °C sunny day is fine
            # hiking weather; capping it would cry wolf across a Berg winter.
            note(f"{js_round(tmin)} °C at dawn",
                 "Cold start — frost is possible before the sun gets on it",
                 SEV_CAUTION)

        # Needs all three of temperature, rain AND wind, so an unknown gust
        # makes it unevaluable. Reading `g` as 0 here would silently exempt
        # exactly the freezing wet day the trap was built for.
        if not gust_assessed:
            if "wet-cold" not in unassessed:
                unassessed.append("wet-cold")
        elif tmin <= 10 and p > 1 and g > 25:
            veto(3, "Wet, cold and windy",
                 "The classic hypothermia combination — worse than it looks",
                 SEV_SEVERE)

    # ── Attribution ────────────────────────────────────────────────────────
    # metno.js gates this on `cap < comfort`: a veto that fired but did not bite
    # did not set the score and must not claim to have. There is no comfort term
    # here, so the equivalent test is simply that a veto fired at all.
    if binding_at is not None:
        factors[binding_at].binding = True

    factors.sort(key=lambda f: (not f.binding, -f.severity))
    return Verdict(cap=max(0.0, min(10.0, cap)), factors=factors,
                   unassessed=unassessed)
