#!/usr/bin/env python3
"""
PER-STRATEGY FIRE ACCURACY ANALYZER

Question: at what threshold does each (strategy, side) achieve 70-80% WR?

Loads multi_indicator_models.pkl (already trained ensemble) and slices its
predictions by (strategy, side). Reports per-strategy:
  - WR at thresholds 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
  - n_fires that pass each threshold
  - Wilson lower-CI to confirm WR is stat-significantly above target

Output: per_strategy_thresholds.json
  Per (strategy, side): the threshold that achieves >= 70% Wilson_lower WR
  with n >= 20

USAGE on VPS:
  python3 per_strategy_wr.py
"""
import json, math, sys, os, pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
os.chdir('./data')

# Load
print("Loading models...")
with open('multi_indicator_models.pkl', 'rb') as f:
    saved = pickle.load(f)

models = saved['models']
scaler = saved['scaler']
feat_names = saved['features']

print("Loading data...")
ticks_by_slug = {}
with open('market_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        if r.get('slug') and r.get('ticks'):
            ticks_by_slug[r['slug']] = r['ticks']

free_fires = []
with open('market_recap_history.jsonl') as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        slug = r.get('slug')
        ts = r.get('ts')
        if slug not in ticks_by_slug: continue
        for fire in r.get('fires', []):
            cd = fire.get('cd', 0)
            if cd is None or cd < 15: continue
            if (fire.get('pre_gate_held') or fire.get('opp_gate_held')
                or fire.get('dedup_excluded') or fire.get('pricecap_excluded')
                or fire.get('model_vetoed') or fire.get('flip_gate_blocked')
                or fire.get('bn_vetoed')):
                continue
            free_fires.append({
                'slug':slug,'ts':ts,'cd':cd,
                'strategy':fire.get('strategy'),'side':fire.get('side'),
                'entry':fire.get('entry_price'),
                'won':fire.get('hypo_pnl', 0) > 0,
                'hypo':fire.get('hypo_pnl', 0),
            })

# Compute features
print("Computing features...")
import flip_features
import datetime

data = []
for f in free_fires:
    visible = [t for t in ticks_by_slug[f['slug']] if t[0] >= f['cd']]
    if len(visible) < 10: continue
    try:
        feats = flip_features.build_features(
            visible,
            current_time=datetime.datetime.fromtimestamp(f['ts']) if f['ts'] else None
        )
        feats['fire_cd']      = float(f['cd'])
        feats['fire_entry']   = float(f['entry']) if f['entry'] else 0.0
        feats['fire_side_dn'] = 1.0 if f['side'] == 'DN' else 0.0
    except Exception:
        continue
    data.append({**f, 'feats':feats})

# Walk-forward chronological split
data.sort(key=lambda d: d['ts'] or 0)
split = int(len(data)*0.70)
test = data[split:]
print(f"Test set: {len(test)} fires (out-of-sample)")

# Build feature matrix and get ensemble probas
def featvec(d):
    row = []
    for fn in feat_names:
        v = d['feats'].get(fn, 0)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        row.append(float(v))
    return row

X_te = np.array([featvec(d) for d in test])
X_te_s = scaler.transform(X_te)

# Soft-vote ensemble
probas = np.mean([m.predict_proba(X_te_s)[:,1] for m in models.values()], axis=0)

# Wilson CI
def wilson_ci(wins, n, z=1.96):
    if n == 0: return (0,0)
    p = wins/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (max(0,center-spread), min(1,center+spread))

# Group by strategy|side
print("\n" + "="*100)
print("PER-STRATEGY FIRE ACCURACY (using soft-vote ensemble on out-of-sample test set)")
print("="*100)

by_strat = defaultdict(list)
for i, d in enumerate(test):
    key = (d['strategy'], d['side'])
    by_strat[key].append({'won':d['won'], 'proba':float(probas[i])})

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print(f"\n{'strategy|side':<35s}  {'base_WR%':>8s}  {'thr':>5s}  {'n':>4s}  {'WR%':>5s}  {'Wilson_lo':>9s}")
print('-'*90)

# Best threshold per strategy
deployable = {}
for (strat, side), recs in sorted(by_strat.items(), key=lambda x: -len(x[1])):
    n_total = len(recs)
    if n_total < 30: continue   # need decent sample
    
    base_wins = sum(1 for r in recs if r['won'])
    base_wr = base_wins/n_total
    
    print(f"\n{strat+'|'+side:<35s}  {base_wr*100:>7.0f}%  (lifetime n={n_total})")
    
    best_for_strat = None
    for thr in THRESHOLDS:
        passing = [r for r in recs if r['proba'] >= thr]
        n = len(passing)
        if n < 10:
            print(f"  {' ':<33s}  {'    ':>8s}  {thr:.2f}  {n:>3d}  {'':>5s}  {'<10 fires':>9s}")
            continue
        wins = sum(1 for r in passing if r['won'])
        wr = wins/n
        wr_lo, wr_hi = wilson_ci(wins, n)
        flag = '🎯' if wr_lo >= 0.70 else ('🟡' if wr_lo >= 0.55 else '  ')
        print(f"  {' ':<33s}  {'    ':>8s}  {thr:.2f}  {n:>3d}  {wr*100:>4.0f}%   {wr_lo*100:>6.1f}%  {flag}")
        
        if wr_lo >= 0.70 and n >= 20:
            if best_for_strat is None or wr > best_for_strat['wr']:
                best_for_strat = {
                    'threshold': thr, 'n': n, 'wr': wr,
                    'wilson_lo': wr_lo, 'wilson_hi': wr_hi
                }
    
    if best_for_strat:
        deployable[f"{strat}|{side}"] = best_for_strat
        print(f"  ✅ DEPLOYABLE at thr={best_for_strat['threshold']}: WR={best_for_strat['wr']*100:.0f}% Wilson_lo={best_for_strat['wilson_lo']*100:.0f}%")

# Save
with open('per_strategy_thresholds.json', 'w') as f:
    json.dump(deployable, f, indent=2)

print(f"\n{'='*100}")
print(f"SUMMARY")
print(f"{'='*100}")
print(f"Strategies analyzed:    {sum(1 for k,v in by_strat.items() if len(v) >= 30)}")
print(f"Strategies DEPLOYABLE:  {len(deployable)}")
print(f"  (deployable = at least one threshold gives Wilson_lower(WR) >= 70% with n >= 20)")
print()
if deployable:
    print(f"{'strategy|side':<35s} {'thr':>5s} {'n':>4s} {'WR%':>5s} {'Wilson_lo%':>11s}")
    for k, v in sorted(deployable.items(), key=lambda x: -x[1]['wilson_lo']):
        print(f"  ✅ {k:<32s} {v['threshold']:.2f}  {v['n']:>3d}  {v['wr']*100:>4.0f}%   {v['wilson_lo']*100:>9.1f}%")
else:
    print("❌ No strategy passes 70% Wilson_lower WR with n>=20")
    print()
    print("This means: the ensemble model can find fires with high WR globally,")
    print("but no single strategy achieves 70% WR consistently when filtered.")
    print()
    print("Two options:")
    print("  A) Lower target to 60% Wilson_lower (achievable per-strategy)")
    print("  B) Use global ensemble gate (90% WR at thr=0.70) without per-strategy")
    print("     constraint — fire any strategy as long as ensemble agrees")

print(f"\nSaved: per_strategy_thresholds.json")
