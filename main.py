"""GTA VI trophy earn-rate forecaster (make.fun bounty gta6-trophy-rarity-forecast-01).

Reads one case per file in /input (MAKEFUN_INPUT_DIR), writes one forecast per
case to /output (MAKEFUN_OUTPUT_DIR): for each of the game's trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's
target, in percent.

Method
------
An Exophase earn rate is earners / players-tracked. On launch night 638
players were tracked and every rate sat still; the first refresh (04:30 UTC,
19 Nov) took tracked to 3,868 and every rate but the 100% one fell, because
the new tracked players had earned far less. Two regimes follow from that:

  * A moving history: for each trophy, fit logit(rate) against ln(1 + hours
    since the release day began) over the last 24 hours of reads, with
    exponential recency weights (12 h half-life). Slopes are shrunk 15%
    toward the cross-trophy median slope, which steadies trophies whose own
    fit is noisy. The median at the target is the fitted line read there,
    with the logit move from the last read capped (up 3.5, down 2.0) so a
    steep recent trend cannot run away over a four-day horizon.
  * A flat history (every read identical, as on launch night): the page has
    not refreshed yet, so the next refresh will cut most rates as the tracked
    population grows. The median is moved down in logit space (about 1.5
    units for a trophy nobody has all of, nothing for one at 100%) and the
    spread is widened, instead of trusting the stale rate.

Spreads are honest: the fit's residual standard deviation, a random-walk
term (0.4 * sqrt(horizon in days)) and a base model-uncertainty term,
combined in quadrature and clamped to [0.35, 1.8] logit units.

Standard library only. Deterministic: same input bytes give the same output
bytes. No network.
"""

import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; the house curve baseline measures
# hours from the same origin.
RELEASE = datetime.fromisoformat("2026-11-19T00:00:00+00:00")

# Fit window and recency weighting.
WINDOW_HOURS = 24.0
HALF_LIFE_HOURS = 12.0
# Shrinkage of each trophy's slope toward the cross-trophy median slope.
SHRINK = 0.15
# Caps on the logit move from the last read (up / down).
MOVE_UP_CAP = 3.5
MOVE_DOWN_CAP = 2.0
# Spread: base model uncertainty, and the random-walk coefficient.
BASE_SD = 0.20
RW_SD = 0.40
SPREAD_MIN = 0.35
SPREAD_MAX = 1.8
# Flat-history refresh: logit drop for a trophy, scaled by (1 - rate/100).
FLAT_DROP = 1.5
FLAT_SD = 1.0


def clamp_rate(x):
    """A rate in percent, clamped to the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x*10000)/10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(ts):
    return (parse_ts(ts) - RELEASE).total_seconds() / 3600.0


def wls(points):
    """Weighted least squares y ~ a + b*x over (x, y, w) triples.

    Returns (a, b, resid_sd, n). Falls back to a flat line at the weighted
    mean when there are too few points or no spread in x.
    """
    n = len(points)
    if n == 0:
        return 0.0, 0.0, 0.3, 0
    if n < 3:
        ys = [y for _x, y, _w in points]
        return statistics.median(ys), 0.0, 0.3, n
    sw = sum(w for _x, _y, w in points)
    mx = sum(w * x for x, _y, w in points) / sw
    my = sum(w * y for _x, y, w in points) / sw
    vx = sum(w * (x - mx) ** 2 for x, _y, w in points)
    if vx <= 1e-12:
        return my, 0.0, 0.3, n
    b = sum(w * (x - mx) * (y - my) for x, y, w in points) / vx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y, _w in points]
    if n > 2:
        sd = math.sqrt(sum(w * r * r for r, (_x, _y, w) in zip(resid, points)) / sw)
        sd *= math.sqrt(n / max(1, n - 2))
    else:
        sd = 0.3
    return a, b, sd, n


def series_for(case, idx):
    """(t_hours, rate) for trophy idx, one row per read with a rate."""
    out = []
    for read in case.get("history", []) or []:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        try:
            t = hours_since_release(read["at"])
        except (KeyError, ValueError, TypeError):
            continue
        out.append((t, float(rates[idx])))
    return out


def recent_window(series, t_last):
    """The reads in the last WINDOW_HOURS, with recency weights.

    Falls back to the whole series when the window holds fewer than 3 reads.
    """
    window = [(t, r) for (t, r) in series if t >= t_last - WINDOW_HOURS]
    if len(window) < 3:
        window = list(series)
    out = []
    for (t, r) in window:
        w = 0.5 ** ((t_last - t) / HALF_LIFE_HOURS)
        out.append((t, r, w))
    return out


def fit_trophy(series, t_target):
    """Per-trophy fit at the target time.

    Returns (y_med, sd, slope): the median logit at the target, the spread
    in logit units, and the fitted slope (None for a flat history).
    """
    if not series:
        return logit(0.5), FLAT_SD, None
    t_last = series[-1][0]
    r_last = series[-1][1]
    y_last = logit(r_last)

    rates = [r for (_t, r) in series]
    if len(set(rates)) < 3:
        # Flat history: the page has not refreshed. The next refresh grows
        # the tracked population and cuts most rates; move the median down
        # and widen the spread instead of trusting the stale rate.
        drop = FLAT_DROP * (1.0 - r_last / 100.0)
        return y_last - drop, FLAT_SD, None

    window = recent_window(series, t_last)
    pts = [(math.log1p(t), logit(r), w) for (t, r, w) in window]
    a, b, sd_r, _n = wls(pts)
    x_target = math.log1p(t_target)
    y_med = a + b * x_target

    # Damp the move from the last read: a trophy's rate cannot run away in
    # a few days, and the log-time line is the least reliable far out.
    y_med = min(y_last + MOVE_UP_CAP, max(y_last - MOVE_DOWN_CAP, y_med))
    y_med = min(logit(99.99), max(logit(0.01), y_med))

    # Spread: fit residual, the random-walk term, and base uncertainty.
    horizon_days = max(0.0, (t_target - t_last) / 24.0)
    sd = math.sqrt(sd_r * sd_r + (RW_SD * math.sqrt(horizon_days)) ** 2 + BASE_SD ** 2)
    sd = min(SPREAD_MAX, max(SPREAD_MIN, sd))
    return y_med, sd, b


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{case.get('id')} asks for levels {case.get('levels')}, not {LEVELS}")
    if not case.get("history"):
        raise ValueError(f"{case.get('id')} has no history")
    t_target = hours_since_release(case["target"])
    trophies = case["trophies"]

    # First pass: per-trophy median logit, spread, slope, and last read.
    fits = []
    slopes = []
    for idx in range(len(trophies)):
        series = series_for(case, idx)
        y_med, sd, slope = fit_trophy(series, t_target)
        t_last = series[-1][0] if series else t_target
        y_last = logit(series[-1][1]) if series else logit(0.5)
        fits.append((y_med, sd, slope, y_last, t_last))
        if slope is not None:
            slopes.append(slope)

    b_med = statistics.median(slopes) if slopes else None

    rows = []
    for idx, trophy in enumerate(trophies):
        y_med, sd, slope, y_last, t_last = fits[idx]
        if slope is not None and b_med is not None:
            # Re-project with the slope shrunk toward the cross-trophy median,
            # anchored on the last observed rate.
            slope_shrunk = (1.0 - SHRINK) * slope + SHRINK * b_med
            y_new = y_last + slope_shrunk * (math.log1p(t_target) - math.log1p(t_last))
            y_new = min(y_last + MOVE_UP_CAP, max(y_last - MOVE_DOWN_CAP, y_new))
            y_new = min(logit(99.99), max(logit(0.01), y_new))
            y_med = y_new
        qs = [round4(clamp_rate(inv_logit(y_med + z * sd))) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
