# per-strategy-winrate-eval

Computes per-strategy win rates over Polymarket fire history (BTC/XRP 5-minute up/down markets) and finds the model-probability threshold at which each `(strategy, side)` becomes statistically deployable.

Replays historical fires through the trained ensemble, slices results per strategy, and writes a per-strategy threshold map the bot can gate on.

## Scripts

- `per_strategy_wr.py`: entry point. Loads the trained ensemble and historical fires, rebuilds features, soft-votes on a chronological out-of-sample split, reports WR and Wilson lower-CI per `(strategy, side)` across thresholds 0.50 to 0.80, writes the deployable threshold map.
- `flip_features.py`: feature extractor vendored from the engine's `vps_feature_extraction.py`. Builds per-window (`entry`/`early`/`mid`/`late`/`very_late`) aggregates, cd-snapshot features, and UTC context features from raw tick rows. Edit upstream, not here.

## Pipeline

1. Load the trained ensemble (`models`, `scaler`, `features`) from the pickle.
2. Load tick history and recap history; keep only free fires (drop gate-held, dedup, price-capped, model/bn-vetoed, flip-blocked).
3. Rebuild each fire's feature vector via `flip_features.build_features`.
4. Sort chronologically, take the last 30% as out-of-sample test, soft-vote the ensemble.
5. Per `(strategy, side)` with n>=30: report WR and Wilson lower CI at each threshold. Mark deployable when Wilson-lower >= 70% with n>=20.

## Requirements

Python 3 with `numpy` and `scikit-learn` (pickled models are sklearn estimators; loading needs a matching version), plus stdlib. Read-only; no credentials required.

## Usage

```bash
python3 per_strategy_wr.py
```

Prints the per-strategy threshold table to stdout and writes `per_strategy_thresholds.json` (the deployable `(strategy, side) -> {threshold, n, wr, wilson_lo, wilson_hi}` map) into the data dir.

## Data

Not self-contained. Reads three artifacts from the private polymarket-data repo, expected in `$DATA_DIR`:

- `multi_indicator_models.pkl`: trained soft-vote ensemble (`models` / `scaler` / `features`).
- `market_history.jsonl`: per-market raw tick replays, keyed by slug.
- `market_recap_history.jsonl`: per-market fire records (strategy, side, entry, hypo PnL, gate flags).

None are committed; all `*.pkl` / `*.jsonl` / `*.json` are gitignored. Paths in `per_strategy_wr.py` are currently hardcoded to the engine checkout and `./data`; repoint them before running.

## Feedback

Suggestions, improvements, or something not working? Please
[open an issue](https://github.com/needmoneyforhoes/per-strategy-winrate-eval/issues/new). Pull requests are welcome too.
