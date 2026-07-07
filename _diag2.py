import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '.')
from train_phase_scorer import (
    extract_features_for_swing, _get_benchmark_stats, _get_local_contribs_for_phase,
    PHASE_LABEL_MAP, FEATURE_FEEDBACK_MAP, MIN_LOCAL_CONTRIB, MIN_NORM_DEVIATION,
    FEEDBACK_SCORE_THRESHOLD, MODEL_OUT_PATH
)

with open(MODEL_OUT_PATH, 'rb') as f:
    bundle = pickle.load(f)

models       = bundle['models']
imputer      = bundle['imputer']
feature_cols = bundle['feature_cols']
benchmarks   = bundle.get('benchmarks', {})

# Only check phases that are below threshold
swings = [
    ('Minh_nn',     {}),             # all phases >= 85 — confirm no false feedback
    ('B2I_F9_I_nn', {}),             # all phases >= 85 — confirm no false feedback
    ('akh_nn',      {'Finish': 53.2}), # only Finish is broken
]

for swing_id, forced_phase_scores in swings:
    print(f'\n{"="*72}')
    print(f'SWING: {swing_id}')

    feats = extract_features_for_swing(swing_id)
    if feats is None:
        print('  [SKIP] missing data')
        continue

    row   = {col: feats.get(col, np.nan) for col in feature_cols}
    X_raw = np.array([[row[c] for c in feature_cols]])
    X     = imputer.transform(X_raw)

    # Determine which phases to examine
    # If forced_phase_scores is empty, compute actual scores for all phases
    phase_scores = {}
    for score_col, phase_name in PHASE_LABEL_MAP.items():
        if score_col not in models:
            continue
        if forced_phase_scores:
            if phase_name in forced_phase_scores:
                phase_scores[phase_name] = forced_phase_scores[phase_name]
        else:
            pred = float(np.clip(models[score_col].predict(X)[0], 0, 100))
            phase_scores[phase_name] = round(pred, 1)

    if not forced_phase_scores:
        print(f'  Predicted scores (all phases):')
        for pn, sc in phase_scores.items():
            tag = 'OK' if sc >= FEEDBACK_SCORE_THRESHOLD else 'WEAK'
            print(f'    {pn:<22} {sc:>6.1f}  [{tag}]')
        # Check if any are below threshold
        weak = {pn: sc for pn, sc in phase_scores.items() if sc < FEEDBACK_SCORE_THRESHOLD}
        if not weak:
            print('  All phases above threshold — no feedback triggered. Scoring is clean.')
            continue
        phase_scores = weak

    for score_col, phase_name in PHASE_LABEL_MAP.items():
        if phase_name not in phase_scores:
            continue
        score = phase_scores[phase_name]
        if score_col not in models:
            continue

        model          = models[score_col]
        bench          = benchmarks.get(score_col, {})
        local_contribs = _get_local_contribs_for_phase(model, X, feature_cols)
        prefix         = phase_name.replace('-', '_').replace(' ', '_').lower()

        print(f'\n  PHASE: {phase_name}  score={score}')
        print(f'  {"-"*72}')
        print(f'  {"METRIC (dir)":<30} {"NORM_DEV":>9} {"LOCAL":>9} {"WEIGHTED":>9}  FAILURE')

        rows = []
        for col in feature_cols:
            if not col.startswith(prefix + '_'):
                continue
            metric = col[len(prefix) + 1:]
            val    = feats.get(col, np.nan)
            if np.isnan(val):
                continue
            stats = _get_benchmark_stats(bench, col)
            if stats is None:
                continue
            med, iqr  = stats
            deviation = val - med
            norm_dev  = abs(deviation) / max(iqr, 1e-6)
            local     = float(local_contribs.get(col, 0.0))
            weighted  = local * norm_dev
            direction = 'high' if deviation > 0 else 'low'
            key       = (prefix, metric, direction)

            if local < MIN_LOCAL_CONTRIB and norm_dev < MIN_NORM_DEVIATION:
                status = 'FAIL: contrib+deviation both low'
            elif local < MIN_LOCAL_CONTRIB:
                status = 'FAIL: contrib too low'
            elif norm_dev < MIN_NORM_DEVIATION:
                status = 'FAIL: deviation too small'
            elif key not in FEATURE_FEEDBACK_MAP:
                status = 'FAIL: no map key'
            else:
                status = 'PASS mapped'

            label = f'{metric} ({direction})'
            rows.append((weighted, label, norm_dev, local, weighted, status))

        rows.sort(reverse=True)
        for _, label, nd, lc, w, st in rows[:14]:
            print(f'  {label:<30} {nd:>9.2f} {lc:>9.4f} {w:>9.4f}  {st}')

        # Summary counts
        fail_contrib  = sum(1 for r in rows if 'contrib too low' in r[5])
        fail_dev      = sum(1 for r in rows if 'deviation too small' in r[5])
        fail_nokey    = sum(1 for r in rows if 'no map key' in r[5])
        fail_both     = sum(1 for r in rows if 'both low' in r[5])
        passed        = sum(1 for r in rows if 'PASS' in r[5])
        all_zero_local = all(r[3] == 0.0 for r in rows)
        print(f'\n  SUMMARY: passed={passed} | no_map_key={fail_nokey} | contrib_low={fail_contrib} | dev_small={fail_dev} | both_low={fail_both}')
        if all_zero_local:
            print(f'  *** ALL LOCAL CONTRIBS ARE ZERO — model relies entirely on bias term for this phase ***')
