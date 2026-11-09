# AeroGuard-RUL: calibrated, decision-aware jet engine failure prediction

Predicts the remaining useful life (RUL) of NASA C-MAPSS turbofan engines from 21 sensors, and turns
the prediction into a **maintenance decision with a statistical guarantee**. Includes an interactive dashboard.

## Why this goes beyond a typical RUL project
Most portfolio versions output one number and report RMSE. Airlines cannot act on a single number,
because the cost of being late (an in-flight failure) is far higher than the cost of being early.
This project adds the missing layers:

| Layer | What it does | Why it matters |
|---|---|---|
| Regime-aware normalisation | KMeans on operating settings, per-regime z-scores (FD002/FD004) | Removes altitude/speed effects that hide degradation |
| Learned health index | 1-D damage coordinate fitted from all sensors | Interpretable "engine health" curve |
| Causal window features | level, mean, slope, volatility, HI dynamics | No look-ahead, works online |
| Quantile boosting | 7 quantiles (5% to 95%) of RUL | Full predictive distribution, not a point |
| Conformal calibration | CQR 90% interval and a one-sided **safe horizon** | Distribution-free coverage, checked on held-out engines |
| Cost-aware policy | Simulates failure vs planned vs wasted-life cost | Converts accuracy into money |
| Fleet triage | Critical / schedule / no action by safe horizon | The output a planner actually uses |
| Sensor attribution | Permutation importance grouped by sensor | Explains the model to engineers |

The safe horizon is the 10% quantile corrected by a conformal margin, so that the engine outlives it
with roughly 90% reliability. Replacing at that horizon, rather than at the median, is what drives the cost saving.

## Run it (VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run.py --open          # --fd FD002 for multi-regime data, --synthetic to force the demo
```
1. Download C-MAPSS and copy `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into `data/` (see `data/README.txt`).
2. Run the command above. It trains in roughly 15-60 seconds and writes `dashboard/index.html` and `results/metrics.json`.
3. Open `dashboard/index.html` (a static file, no server needed; it loads Chart.js and a font from a CDN, so be online).

Without the data files, the code runs on synthetic C-MAPSS-like data so you can see the whole flow immediately.
**The dashboard shipped in this zip was generated from that synthetic data. Re-run with the real files before quoting any numbers.**

## Project layout
```
run.py                         entry point
src/pipeline.py                data, features, models, conformal, cost simulation
src/dashboard_template.html    dashboard (data is injected at the end of the run)
dashboard/index.html           generated output
results/metrics.json           generated metrics
```

## Metrics reported
RMSE and the NASA asymmetric score (standard in the literature), coverage of the 90% interval, reliability of the
safe horizon, and fleet cost against three baselines (run to failure, replace immediately, replace at the median prediction).
RUL is capped at 125 cycles for training and for coverage checks, the usual convention. RMSE and score use the raw NASA test RUL.
Change cost assumptions in `COST` and the shop lead time in `LEAD` at the top of `src/pipeline.py`.

## Resume line
"Built an uncertainty-calibrated predictive maintenance system on NASA turbofan data: regime-aware health index, quantile
gradient boosting with conformal safe-horizon guarantees, a cost-optimised replacement policy, and an interactive fleet dashboard."

## Ideas to extend
GRU/Transformer encoder in place of window features, survival analysis (Weibull) heads, drift monitoring, a FastAPI scoring endpoint, and tests on all four FD sets.
