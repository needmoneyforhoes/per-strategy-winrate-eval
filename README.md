# per-strategy-winrate-eval

Computes per-strategy win rates over Polymarket fire history (BTC/XRP 5-minute crypto up/down markets), and finds the model-probability threshold at which each `(strategy, side)` becomes statistically deployable.

## Why it exists

The live 5-minute bot fires dozens of named strategies per side. This tool answers one question offline: **at what ensemble-probability threshold does each `(strategy, side)` reach a 70%+ win rate that holds up out-of-sample?** It replays historical fires through the trained ensemble, slices results per strategy, and emits a per-strategy threshold map the bot can gate on.

## How it works

| Module | Role |
|--------|------|
| `per_strategy_wr.py` | Entry point. Loads the trained ensemble (`multi_indicator_models.pkl`) and historical fires, rebuilds features, scores them with a soft-vote ensemble on a chronological out-of-sample split, then reports WR / Wilson-lower-CI per `(strategy, side)` across thresholds 0.50–0.80. Writes the deployable threshold map. |
| `flip_features.py` | Vendored runtime feature extractor — mirrors the engine's `vps_feature_extraction.py` exactly so offline feature values match training/live. Builds per-window (`entry`/`early`/`mid`/`late`/`very_late`) aggregates, cd-snapshot features, and UTC context features from raw tick rows. **Vendored from the engine — edit upstream, not here.** |

Pipeline in `per_strategy_wr.py`:
1. Load the trained ensemble (`models`, `scaler`, `features`) from the pickle.
2. Load market tick history and recap history; keep only *free* fires (drop gate-held / dedup / price-capped / model- or bn-vetoed / flip-blocked).
3. Rebuild each fire's feature vector via `flip_features.build_features`.
4. Sort chronologically, take the last 30% as out-of-sample test, soft-vote the ensemble.
5. Per `(strategy, side)` with n≥30: report WR and Wilson lower CI at each threshold; mark **DEPLOYABLE** when Wilson-lower ≥ 70% with n≥20.

## Requirements

- Python 3 with `numpy`, `scikit-learn` (the pickled models are sklearn estimators; loading requires a matching version), plus stdlib (`json`, `pickle`, `math`, `statistics`, `datetime`).
- No wallet, private key, or network access — this is read-only analysis. It does **not** handle funds.
- Read access to the engine checkout and its data dir (paths are currently hardcoded to `/home/polybot/polymarket-bot` and `/home/polybot/polymarket-bot/data`).

## Usage

```bash
python3 per_strategy_wr.py
```

Prints the per-strategy threshold table to stdout and writes `per_strategy_thresholds.json` (the deployable `(strategy, side) → {threshold, n, wr, wilson_lo, wilson_hi}` map) into the data dir.

## Data

This is **not self-contained** — it reads three artifacts from the private `polymarket-data` repo (currently expected under `/home/polybot/polymarket-bot/data`):

- `multi_indicator_models.pkl` — the trained soft-vote ensemble (`models` / `scaler` / `features`).
- `market_history.jsonl` — per-market raw tick replays, keyed by slug.
- `market_recap_history.jsonl` — per-market fire records (strategy, side, entry, hypo PnL, gate flags).

Place those files in the data dir (or repoint the hardcoded paths) before running. None of them are committed; all `*.pkl` / `*.jsonl` / `*.json` are gitignored.

> Private research software. No warranty; trades/handles real funds at your own risk.
