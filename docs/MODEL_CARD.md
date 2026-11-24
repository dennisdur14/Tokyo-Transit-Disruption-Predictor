# Model card: 30-minute Toei bus delay forecast

## Model details

- **Task:** forecast a bus route's mean delay (seconds behind schedule) in the 15-minute interval
  that starts 30 minutes from now.
- **Model:** XGBoost regressor (250 trees, depth 5, learning rate 0.05) predicting the *change* from
  the current delay. Six variants (A-F) differ only in their inputs; F uses everything.
- **Inputs:** time of day and route; recent bus delays and weather; the rail status header of linked
  lines; what the rule-based parser reads from their Japanese bulletins (disrupted, suspended,
  resumed, 振替輸送, cause, heuristic severity); TF-IDF embeddings of the bulletin text.
- **Version:** November 2026. Hyperparameters are fixed, not tuned.

## Intended use

- A research and portfolio project testing whether rail-disruption bulletins carry information about
  later bus delays.
- **Not** for operational dispatching or passenger information. It has not been validated on real data.

## Data

- **Current:** synthetic data from `src/simulate.py` (28 days, 8 routes, 6 rail lines). Bus spillover
  is planted from hidden incident attributes and never from the bulletin text or parser output.
- **Being collected:** real Toei bus positions with timetable-based delay estimates, ODPT rail
  bulletins and Open-Meteo weather, every 15 minutes from 18 November 2026.

## Evaluation

- **Rolling origin:** the newer half of the timeline is split into 4 successive test blocks. Each block
  is forecast by models trained only on earlier data, with a 30-minute gap so no training target
  falls inside the test block.
- **Baselines:** persistence (A), and model C, which has no rail information. Rail features have to
  beat C to count.
- **Placebo check:** the same evaluation on a simulated world with the rail effect switched off.
- **Metrics:** RMSE and MAE on all test rows and on rail-disruption windows (a linked line showed
  trouble in the past hour); F1 for "surge" intervals (≥3 minutes above the past hour's average, or
  ≥7 minutes late).

## Results (synthetic only)

| | Change in disruption-window RMSE vs. model C, mean of 5 seeds |
|---|---|
| D. + status header | planted −8.7%, placebo +0.5% |
| E. + parsed bulletin text | planted −18.4% (every seed improved, by 0.2% to 34.1%), placebo +0.4% |
| F. + TF-IDF embeddings | planted −17.9%, placebo +1.3% |

Negative means lower error. Seed-level results are in `artifacts/sanity_check_runs.csv` after
`python -m src.sanity_check`.

## Limitations and risks

- No real-data performance is known yet. The synthetic results depend on assumed incident rates and
  effect sizes.
- Rare, large disruptions dominate the error. With few incidents to learn from, the gain can vanish:
  in one of five synthetic seeds the text model gained almost nothing and the embedding model made
  forecasts slightly worse.
- ODPT's rail feed has no JR East or Tokyu bulletins, so disruptions on those lines are invisible.
- Bus delays are estimated from position snapshots (about ±1 minute) and the feed has gaps.
- The route-to-rail links are a hand-made hypothesis.
- The parser's severity score is a hand-set heuristic, not a calibrated probability.
