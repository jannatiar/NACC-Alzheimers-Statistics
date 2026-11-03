"""Phase 1 (NACC Dataset C), common tasks 8 and 9 ONLY.

Read two original data files, match by participant NACCID, validate status,
create three task-8 figures (a fourth joint figure is produced by C5) and group tables.
No input is edited; no imputation, testing, confidence intervals, or causal claims.

Run from project root:
    python src/05_visualize_and_compare.py
Outputs:
    figures/phase1_*.png                         (tracked, anonymous aggregates)
    data/derived/phase1/visual_comparison/*.csv  (local intermediate aggregate tables)
    data/derived/phase1/visual_comparison/*.json (local summary)
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save files on all systems; no interactive window required.
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
FIG = ROOT / 'figures'
OUT = ROOT / 'data' / 'derived' / 'phase1' / 'visual_comparison'

GROUPS = {
    1: 'Normal',
    2: 'Impaired (non-MCI)',
    3: 'MCI',
    4: 'Dementia',
}
MEASURES = {
    'CDRSUM': ('UDS', 'CDR Sum of Boxes', 'score'),
    'LH_ENTORHINAL_AVGTH': ('MRI', 'Left entorhinal mean cortical thickness', 'mm'),
    'RH_ENTORHINAL_AVGTH': ('MRI', 'Right entorhinal mean cortical thickness', 'mm'),
    'ENTORHINAL_MEAN_AVGTH': ('MRI-derived', 'Bilateral mean entorhinal thickness', 'mm (assumed)'),
}
FIGURES = [
    ('phase1_08_01_cognitive_group_counts.png', 'Frequency of diagnostic-status groups'),
    ('phase1_08_02_cdrsum_discrete_distribution.png', 'Empirical distribution of CDR Sum of Boxes'),
    ('phase1_08_03_mri_thickness_distribution.png', 'Distribution of left/right entorhinal thickness'),
]


def read_unique(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'NACCID' not in reader.fieldnames:
            raise ValueError(f'Missing NACCID: {path.name}')
        rows = {}
        for line, row in enumerate(reader, 2):
            key = (row.get('NACCID') or '').strip()
            if not key or key in rows:
                raise ValueError(f'Missing/duplicate NACCID in {path.name}, line {line}')
            rows[key] = row
    return rows


def number(x: str | None, name: str) -> float:
    s = str(x or '').strip()
    if s.lower() in ('', 'nan', 'none', 'null', 'na', 'n/a'):
        return float('nan')
    try:
        z = float(s)
    except ValueError as e:
        raise ValueError(f'Unexpected numeric value {s!r} in {name}') from e
    return z if np.isfinite(z) else float('nan')


def load() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    uds = read_unique(RAW / 'udslong_imputed.csv')
    mri = read_unique(RAW / 'mriscanclean.csv')
    if set(uds) != set(mri):
        raise ValueError('UDS and MRI participant sets differ: cannot assume one-to-one merge')
    status = []
    measurements = {v: [] for v in MEASURES}
    for ident in sorted(uds):
        a, b = uds[ident], mri[ident]
        s_u = number(a.get('NACCUDSD'), 'UDS.NACCUDSD')
        s_m = number(b.get('NACCUDSD'), 'MRI.NACCUDSD')
        if s_u != s_m or s_u not in GROUPS:
            raise ValueError(f'Missing, invalid or discordant cognitive status in participant {ident}')
        status.append(int(s_u))
        for var, (source, _, _) in MEASURES.items():
            if var == "ENTORHINAL_MEAN_AVGTH":
                continue
            v = number((a if source == 'UDS' else b).get(var), var)
            if not np.isfinite(v):
                # Keep as NaN, reported per variable/group without coercion.
                measurements[var].append(np.nan)
            else:
                if var == 'CDRSUM' and (v < 0 or v > 18 or abs(v * 2 - round(v * 2)) > 1e-7):
                    raise ValueError(f'Invalid CDRSUM for participant {ident}: {v}')
                if var.endswith('AVGTH') and v < 0:
                    raise ValueError(f'Negative cortical thickness for participant {ident}: {v}')
                measurements[var].append(v)
    # Calculate from paired hemispheres only; never combine across persons.
    l = np.asarray(measurements['LH_ENTORHINAL_AVGTH'], dtype=float)
    r = np.asarray(measurements['RH_ENTORHINAL_AVGTH'], dtype=float)
    measurements['ENTORHINAL_MEAN_AVGTH'] = np.where(np.isfinite(l) & np.isfinite(r), (l + r) / 2, np.nan).tolist()
    return np.asarray(status, dtype=int), {k: np.asarray(v, dtype=float) for k, v in measurements.items()}


def summarize(values: np.ndarray) -> dict[str, float | int | None]:
    arr = values[np.isfinite(values)]
    n = len(arr)
    out: dict[str, float | int | None] = {'n_valid': n, 'n_missing': len(values) - n}
    if n == 0:
        return {**out, **{x: None for x in ('mean','median','sd_sample','q1','q3','iqr','min','max')}}
    q1, q3 = np.percentile(arr, [25, 75])
    return {**out, 'mean': float(np.mean(arr)), 'median': float(np.median(arr)),
            'sd_sample': float(np.std(arr, ddof=1)) if n > 1 else None,
            'q1': float(q1), 'q3': float(q3), 'iqr': float(q3 - q1),
            'min': float(np.min(arr)), 'max': float(np.max(arr))}


def csv_save(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('empty output table')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f'{v:.6f}' if isinstance(v, float) else ('' if v is None else v)
                             for k, v in row.items()})


def setup():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
                         'savefig.dpi': 185, 'figure.dpi': 120, 'axes.spines.top': False,
                         'axes.spines.right': False})


def save(fig: plt.Figure, name: str):
    fig.savefig(FIG / name, dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  figure -> figures/{name}', flush=True)


def make_charts(status: np.ndarray, measurements: dict[str, np.ndarray]):
    n = len(status)
    groups = [GROUPS[g] for g in GROUPS]
    counts = np.array([np.count_nonzero(status == g) for g in GROUPS])

    # 1. A categorical diagnostic code requires bars (not a histogram).
    fig, ax = plt.subplots(figsize=(9.1, 4.9), layout='constrained')
    bars = ax.bar(range(4), counts, width=.58)
    ax.set(xlabel='UDS cognitive-status category (NACCUDSD)', ylabel='Participants (count)',
           title=f'Cognitive status in the supplied NACC subset (N = {n:,})')
    ax.set_xticks(range(4), groups)
    ax.set_ylim(0, max(counts) * 1.17)
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(counts)*.023,
                f'{count:,} ({100*count/n:.1f}%)', ha='center', va='bottom', fontsize=9)
    save(fig, FIGURES[0][0])

    # 2. CDRSUM: half-point grid, hence display actual discrete mass counts.
    cdr = measurements['CDRSUM']
    valid = cdr[np.isfinite(cdr)]
    unique, freqs = np.unique(valid, return_counts=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.7), layout='constrained')
    ax.bar(unique, freqs, width=.36, align='center')
    ax.set(xlabel='CDR Sum of Boxes (score; increments of 0.5)',
           ylabel='Participants (count)', title=f'Empirical CDR Sum of Boxes frequencies (valid n = {len(valid):,})')
    ax.set_xlim(-.45, max(8.3, float(valid.max())+.4))
    ax.set_xticks(np.arange(0, max(9, np.ceil(valid.max())+1), 1))
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    save(fig, FIGURES[1][0])

    # 3. MRI: common-bin density histograms; count normalization avoids false differences.
    left = measurements['LH_ENTORHINAL_AVGTH']
    right = measurements['RH_ENTORHINAL_AVGTH']
    lv, rv = left[np.isfinite(left)], right[np.isfinite(right)]
    edges = np.histogram_bin_edges(np.concatenate([lv,rv]), bins=32)
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout='constrained')
    ax.hist(lv, bins=edges, histtype='step', linewidth=2, density=True, label=f'Left (n={len(lv)})')
    ax.hist(rv, bins=edges, histtype='step', linewidth=2, density=True, label=f'Right (n={len(rv)})')
    ax.set(xlabel='Entorhinal mean cortical thickness (mm)', ylabel='Density (per mm)',
           title='Marginal distribution of left and right entorhinal thickness')
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save(fig, FIGURES[2][0])



def main() -> None:
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Clean only the three obsolete task-8 charts; preserve other user files.
    for old_name in ('phase1_08_04_cdrsum_by_group.png',
                     'phase1_08_05_left_thickness_by_group.png',
                     'phase1_08_06_right_thickness_by_group.png'):
        (FIG / old_name).unlink(missing_ok=True)
    status, measurements = load()
    n = len(status)
    counts = Counter(status)
    if n != 1237:
        # This is an integrity check tied to the documented classroom subset.
        raise ValueError(f'Unexpected sample size {n} (expected 1237 in the supplied subset)')
    if dict(counts) != {1:847, 2:7, 3:252, 4:131}:
        raise ValueError('Cognitive group counts disagree with PROJECT_CARD and task 6')
    group_rows = []
    for category, label in GROUPS.items():
        for variable, (source, _, unit) in MEASURES.items():
            summary = summarize(measurements[variable][status == category])
            group_rows.append({'category_code': category, 'category_label': label,
                               'n_group': counts[category], 'variable': variable,
                               'source': source, 'unit': unit, **summary})
    overall_rows = []
    for variable, (source, _, unit) in MEASURES.items():
        overall_rows.append({'variable': variable, 'source': source, 'unit': unit,
                             **summarize(measurements[variable])})
    # Check against earlier task-7 summary; guards against silent refactor inconsistencies.
    old_path = ROOT / 'data/derived/phase1/center_spread/center_spread_summary.csv'
    if old_path.exists():
        with old_path.open(encoding='utf-8-sig', newline='') as stream:
            old = {row['variable']: row for row in csv.DictReader(stream)}
        for row in overall_rows:
            ref = old[row['variable']]
            for current_key, reference_key in [('n_valid','n_valid'),('mean','mean'),
                                               ('median','median'),('q1','q1'),('q3','q3')]:
                if abs(float(row[current_key]) - float(ref[reference_key])) > .00101:
                    raise ValueError(f'Inconsistent with task 7: {row["variable"]}/{current_key}')
    csv_save(OUT / 'task9_group_descriptive_summary.csv', group_rows)
    csv_save(OUT / 'task8_overall_distribution_check.csv', overall_rows)
    info = {'n_participants': n, 'cognitive_group_counts': {str(k):counts[k] for k in GROUPS},
            'figures': [file for file, _ in FIGURES],
            'n_figures': len(FIGURES), 'n_figures_in_final_report': len(FIGURES) + 1,
            'fourth_figure_generated_by': '08_c5_joint_uds_mri.py', 'tasks': ['8','9'],
            'analysis': 'descriptive-only; no hypothesis tests; NACCID one per participant',
            'notes': ['Imputation method of UDS is not documented',
                      'UDS/MRI dates are absent; contemporaneity cannot be established',
                      'Group 2 has only seven people; summaries unstable',
                      'All MRI thickness outliers are retained']}
    (OUT / 'task8_9_summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2),encoding='utf-8')
    make_charts(status, measurements)
    print('Tasks 8 and 9 completed: three charts here, one joint C5 chart; group comparison table.', flush=True)
    for r in group_rows:
        print(f"  {r['category_label']:<19} {r['variable']:<28} n={r['n_valid']:<4} "
              f"mean={r['mean']:.3f} median={r['median']:.3f} IQR={r['iqr']:.3f}")


if __name__ == '__main__':
    main()
