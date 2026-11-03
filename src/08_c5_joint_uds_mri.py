"""NACC Phase 1, Dataset C, task C.5: paired clinical/MRI descriptive summary.

One UDS clinical score, two MRI hemispheric measures, and their paired mean.
Only descriptive statistics and plots; no association tests or causal claims.
No participant-level data are exported. Run after C.4 and task 9.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'derived' / 'phase1' / 'c5_joint'
FIG = ROOT / 'figures'
FIG_NAME = 'phase1_c5_joint_uds_mri_scatter.png'
METRICS = ('CDRSUM', 'LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH', 'ENTORHINAL_MEAN_AVGTH')
GROUP_NAMES = {1: 'Normal', 2: 'Impaired (non-MCI)', 3: 'MCI', 4: 'Dementia'}


def descriptive(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    good = x[np.isfinite(x)]
    if len(good) == 0:
        raise ValueError(f'{series.name}: no usable measurements')
    q1, med, q3 = np.percentile(good, [25, 50, 75], method='linear')
    iqr = q3 - q1
    return {'n_valid': int(len(good)), 'n_missing': int(len(x) - len(good)),
            'mean': float(good.mean()), 'median': float(med),
            'sd_sample': float(good.std(ddof=1)) if len(good) > 1 else None,
            'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr),
            'min': float(good.min()), 'max': float(good.max()),
            'n_iqr_outliers': int(((good < q1 - 1.5 * iqr) |
                                   (good > q3 + 1.5 * iqr)).sum())}


def main() -> None:
    uds = pd.read_csv(RAW / 'udslong_imputed.csv', dtype={'NACCID': 'string'})
    mri = pd.read_csv(RAW / 'mriscanclean.csv', dtype={'NACCID': 'string'})
    for label, df in [('UDS', uds), ('MRI', mri)]:
        if df.NACCID.isna().any() or df.NACCID.duplicated().any():
            raise ValueError(f'{label} has missing/repeated participant identifiers')
    if set(uds.NACCID) != set(mri.NACCID):
        raise ValueError('Expected same participant set from C3; review inputs before pairing')
    joined = uds[['NACCID', 'NACCUDSD', 'CDRSUM']].merge(
        mri[['NACCID', 'NACCUDSD', METRICS[1], METRICS[2]]],
        on='NACCID', how='inner', validate='one_to_one', suffixes=('_uds', '_mri'))
    status_u = pd.to_numeric(joined['NACCUDSD_uds'], errors='raise')
    status_m = pd.to_numeric(joined['NACCUDSD_mri'], errors='raise')
    if (status_u != status_m).any() or (~status_u.isin(GROUP_NAMES)).any():
        raise ValueError('Diagnostic-status mismatch or invalid category')
    joined['group'] = status_u.astype(int)
    for metric in METRICS:
        if metric == 'ENTORHINAL_MEAN_AVGTH':
            continue
        joined[metric] = pd.to_numeric(joined[metric], errors='coerce')
    joined['ENTORHINAL_MEAN_AVGTH'] = (
        joined['LH_ENTORHINAL_AVGTH'] + joined['RH_ENTORHINAL_AVGTH']) / 2
    valid = joined[list(METRICS)].notna().all(axis=1)
    # CDRSUM is a half-step score from 0 to 18; regional thickness is positive.
    valid &= joined['CDRSUM'].between(0, 18)
    valid &= np.isclose(joined['CDRSUM'] * 2, np.round(joined['CDRSUM'] * 2))
    for v in METRICS[1:]:
        valid &= np.isfinite(joined[v]) & (joined[v] > 0)
    complete = joined.loc[valid].copy()
    n_excluded = int((~valid).sum())
    if complete.empty:
        raise ValueError('No jointly valid clinical/MRI pairs')

    overall = []
    for v in METRICS:
        overall.append({'variable': v, 'source': 'UDS' if v == 'CDRSUM' else 'MRI',
                        'unit': 'score' if v == 'CDRSUM' else 'mm (assumed)',
                        'n_joint_pairs': len(complete), **descriptive(complete[v])})
    by_group = []
    for code, title in GROUP_NAMES.items():
        sub = complete.loc[complete.group == code]
        for v in METRICS:
            if sub.empty:
                raise ValueError(f'Unexpected empty group: {code}')
            by_group.append({'group_code': code, 'group_name': title,
                             'n_group': len(sub), 'variable': v, **descriptive(sub[v])})

    # Cross-check C4 and common task 9: no silent changes in sample or variable values.
    c4 = json.loads((ROOT / 'data' / 'derived' / 'phase1' / 'c4_missingness' / 'c4_summary.json').read_text())
    if c4['n_matched'] != len(joined):
        raise ValueError('C5 joined population differs from C4 MRI availability')
    task9 = pd.read_csv(ROOT / 'data' / 'derived' / 'phase1' / 'visual_comparison' /
                        'task9_group_descriptive_summary.csv')
    if n_excluded == 0:
        for row in by_group:
            old = task9.loc[(task9['variable'] == row['variable']) &
                            (task9['category_code'] == row['group_code'])]
            if len(old) != 1 or int(old.iloc[0]['n_valid']) != row['n_valid']:
                raise ValueError(f'C5 group size differs from task 9: {row["variable"]}, {row["group_code"]}')
            for k in ('mean', 'median', 'sd_sample', 'iqr'):
                if not np.isclose(float(old.iloc[0][k]), row[k], atol=0.0011):
                    raise ValueError(f'C5 result differs from task 9: {row["variable"]}, {k}')

    # Plot the single derived primary response, not two separate response proxies.
    # Jitter affects plot x-coordinates only and never changes observations.
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    rng = np.random.default_rng(1405)
    jitter = rng.uniform(-0.09, 0.09, size=len(complete))
    for code, name in GROUP_NAMES.items():
        mask = (complete.group.to_numpy() == code)
        ax.scatter(complete.loc[mask, 'CDRSUM'].to_numpy() + jitter[mask],
                   complete.loc[mask, 'ENTORHINAL_MEAN_AVGTH'].to_numpy(),
                   alpha=0.36, s=14, edgecolors='none',
                   label=f'{name} (n={int(mask.sum())})')
    ax.set(title='Clinical impairment and bilateral entorhinal thickness (descriptive)',
           xlabel='CDR Sum of Boxes (score; visual jitter +/-0.09)',
           ylabel='Mean of left and right entorhinal thickness (mm; assumed)')
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / FIG_NAME, dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = {'n_matched_people': len(joined), 'n_joint_valid_people': len(complete),
               'n_excluded_from_joint_by_field_rules': n_excluded,
               'n_joint_valid_per_group': {str(k): int((complete.group == k).sum()) for k in GROUP_NAMES},
               'variables': list(METRICS), 'scatter_figure': FIG_NAME,
               'x_jitter_visual_only': True, 'jitter_amplitude_score': 0.09,
               'association_inference_performed': False,
               'known_limitations': [
                   'Same participant ID does not prove MRI and UDS measurements were obtained on the same date.',
                   'Imputation procedure for UDS not available in supplied course materials.',
                   'Small non-MCI impairment group (n=7); descriptive estimates unstable.',
                   'These data are observational and not representative of the general population.',
               ]}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall).to_csv(OUT / 'c5_joint_overall.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(by_group).to_csv(OUT / 'c5_joint_by_status.csv', index=False, encoding='utf-8-sig')
    (OUT / 'c5_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'C5: jointly valid UDS/MRI pairs={len(complete)}/{len(joined)}; excluded={n_excluded}')
    print(f'C5: grouped statistics and {FIG_NAME} created; no hypothesis test or causal inference.')


if __name__ == '__main__':
    main()
