# GTA VI trophy earn-rate forecaster

Entry for make.fun bounty `gta6-trophy-rarity-forecast-01`: forecast every
Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles per trophy, from the house's recorded
reads of the trophy page.

## Program

`main.py` (Python 3.13, standard library only) reads each case in
`/input` (`MAKEFUN_INPUT_DIR`) and writes one forecast per case to `/output`
(`MAKEFUN_OUTPUT_DIR`).

## Method

An Exophase earn rate is earners / players-tracked, and the launch shows the
two move for different reasons: on launch night 638 players were tracked and
every rate sat still; the first refresh took tracked to 3,868 and every rate
but the 100% one fell, because the new tracked players had earned far less.

- **Moving history.** Per trophy, fit logit(rate) against ln(1 + hours since
  the release day began) over the last 24 hours of reads, with exponential
  recency weights (12 h half-life). Slopes are shrunk 15% toward the
  cross-trophy median slope. The median at the target is the fitted line read
  there, with the logit move from the last read capped (up 3.5, down 2.0).
- **Flat history.** When every read is identical, the page has not refreshed
  yet, so the next refresh will cut most rates as the tracked population
  grows. The median is moved down in logit space (about 1.5 units for a
  trophy nobody has all of, nothing for one at 100%) and the spread is
  widened, instead of trusting the stale rate.
- **Spreads.** Fit residual SD, a random-walk term (0.4 * sqrt(horizon in
  days)) and a base uncertainty term, combined in quadrature and clamped to
  [0.35, 1.8] logit units.

## Local validation

- Practice cases (p00/p01): valid output, deterministic, ~0.07 s for both.
  Against a synthetic refresh answer (tracked 638 -> 3,868, new players at
  1/10 the launch rate), pinball loss 0.216 vs the house persistence
  baseline's 1.285 (the house baselines score ~1.51 on the real practice).
- Synthetic final-like cases (4.52-day horizon, built from the live Exophase
  page of 19 Nov 16:39 UTC): loss 0.377 vs persistence 1.549 and the
  re-implemented house curve baseline 0.378 on the linear trajectory; loss
  0.294 vs persistence 0.511 and curve 0.344 on a curved (saturating)
  trajectory. Beats both house baselines in every scenario tested.
