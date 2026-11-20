"""NACC Phase 2, Stage 1: events and empirical probabilities.

Covers common tasks 1-3 and NACC-specific tasks C.1 and C.3.
Run from anywhere: python src/10_phase2_events_probabilities.py
The original CSV files are read-only. Only aggregated results are exported.
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
OUT = ROOT / 'data' / 'derived' / 'phase2' / 'events'
FIG = ROOT / 'figures'
FIG_NAME = 'phase2_01_conditional_cognitive_status.png'
MRI_THRESHOLD = 3.0575  # Phase 1 median of bilateral entorhinal thickness.
STATUS_NAMES = {1: 'Normal', 2: 'Impaired (non-MCI)', 3: 'MCI', 4: 'Dementia'}


def main() -> None:
    # Read the same UDS and MRI files used in Phase 1; do not modify raw data.
    uds = pd.read_csv(RAW / 'udslong_imputed.csv', dtype={'NACCID': 'string'})
    mri = pd.read_csv(RAW / 'mriscanclean.csv', dtype={'NACCID': 'string'})

    required_uds = ['NACCID', 'NACCUDSD', 'CDRSUM']
    required_mri = ['NACCID', 'NACCUDSD',
                    'LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH']
    for column in required_uds:
        if column not in uds.columns:
            raise ValueError(f'UDS variable is missing: {column}')
    for column in required_mri:
        if column not in mri.columns:
            raise ValueError(f'MRI variable is missing: {column}')

    # One row per participant in each file, as established in Phase 1.
    for name, data in [('UDS', uds), ('MRI', mri)]:
        if (data['NACCID'].isna().any() or
                data['NACCID'].str.strip().eq('').any() or
                data['NACCID'].duplicated().any()):
            raise ValueError(f'{name}: missing, empty, or repeated NACCID')

    if set(uds['NACCID']) != set(mri['NACCID']):
        raise ValueError('UDS and MRI do not contain the same participant IDs')

    data = uds[required_uds].merge(
        mri[required_mri], on='NACCID', how='inner',
        validate='one_to_one', suffixes=('_uds', '_mri'))

    if len(data) != len(uds) or len(data) != len(mri):
        raise ValueError('The one-to-one UDS/MRI merge changed the participant count')

    status_uds = pd.to_numeric(data['NACCUDSD_uds'], errors='raise')
    status_mri = pd.to_numeric(data['NACCUDSD_mri'], errors='raise')
    if (status_uds != status_mri).any():
        raise ValueError('NACCUDSD differs between UDS and MRI')
    if not status_uds.isin(STATUS_NAMES).all():
        raise ValueError('Unexpected NACCUDSD category')

    data['NACCUDSD'] = status_uds.astype(int)
    data['CDRSUM'] = pd.to_numeric(data['CDRSUM'], errors='coerce')
    data['LH_ENTORHINAL_AVGTH'] = pd.to_numeric(
        data['LH_ENTORHINAL_AVGTH'], errors='coerce')
    data['RH_ENTORHINAL_AVGTH'] = pd.to_numeric(
        data['RH_ENTORHINAL_AVGTH'], errors='coerce')
    data['ENTORHINAL_MEAN_AVGTH'] = (
        data['LH_ENTORHINAL_AVGTH'] + data['RH_ENTORHINAL_AVGTH']) / 2

    # Use the same variable rules and complete paired records as Phase 1.
    valid = data[['CDRSUM', 'LH_ENTORHINAL_AVGTH',
                  'RH_ENTORHINAL_AVGTH', 'ENTORHINAL_MEAN_AVGTH']].notna().all(axis=1)
    valid &= np.isfinite(data['CDRSUM'])
    valid &= data['CDRSUM'].between(0, 18)
    valid &= np.isclose(data['CDRSUM'] * 2,
                        np.round(data['CDRSUM'] * 2))
    for variable in ['LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH',
                     'ENTORHINAL_MEAN_AVGTH']:
        valid &= np.isfinite(data[variable]) & (data[variable] > 0)

    complete = data.loc[valid].copy()
    if len(complete) != len(data):
        raise ValueError('Unexpected incomplete primary UDS/MRI records; review Phase 1 rules')
    if complete.empty:
        raise ValueError('No valid paired UDS/MRI records')

    # Confirm that the fixed threshold is still the Phase 1 empirical median.
    phase1_median = float(complete['ENTORHINAL_MEAN_AVGTH'].median())
    if not np.isclose(phase1_median, MRI_THRESHOLD, atol=1e-10):
        raise ValueError('The MRI median differs from the Phase 1 threshold')

    # Events and random variables used in this stage.
    event_a = complete['NACCUDSD'].eq(4)  # UDS cognitive status: dementia.
    event_b = complete['ENTORHINAL_MEAN_AVGTH'].lt(MRI_THRESHOLD)
    not_a = ~event_a
    not_b = ~event_b
    n = len(complete)

    # 2 x 2 table: rows are the UDS event; columns are the MRI event.
    n_a_b = int((event_a & event_b).sum())
    n_a_not_b = int((event_a & not_b).sum())
    n_not_a_b = int((not_a & event_b).sum())
    n_not_a_not_b = int((not_a & not_b).sum())
    n_a = int(event_a.sum())
    n_not_a = int(not_a.sum())
    n_b = int(event_b.sum())
    n_not_b = int(not_b.sum())

    if n_a_b + n_a_not_b + n_not_a_b + n_not_a_not_b != n:
        raise ValueError('The four cells of the joint table do not sum to N')

    joint_table = pd.DataFrame([
        {'UDS_event': 'A', 'n_B': n_a_b, 'n_not_B': n_a_not_b,
         'n_total': n_a, 'P(event_and_B)': n_a_b / n,
         'P(event_and_not_B)': n_a_not_b / n, 'P(event)': n_a / n},
        {'UDS_event': 'NOT A', 'n_B': n_not_a_b,
         'n_not_B': n_not_a_not_b, 'n_total': n_not_a,
         'P(event_and_B)': n_not_a_b / n,
         'P(event_and_not_B)': n_not_a_not_b / n,
         'P(event)': n_not_a / n},
        {'UDS_event': 'Total', 'n_B': n_b, 'n_not_B': n_not_b,
         'n_total': n, 'P(event_and_B)': n_b / n,
         'P(event_and_not_B)': n_not_b / n, 'P(event)': 1.0},
    ])

    # Exactly the two conditional probabilities required for the two MRI groups.
    conditional_rows = [
        {'probability': 'P(A | B)', 'event': 'A', 'condition': 'B',
         'numerator_n': n_a_b, 'denominator_n': n_b,
         'value': n_a_b / n_b, 'percent': 100 * n_a_b / n_b},
        {'probability': 'P(A | NOT B)', 'event': 'A', 'condition': 'NOT B',
         'numerator_n': n_a_not_b, 'denominator_n': n_not_b,
         'value': n_a_not_b / n_not_b,
         'percent': 100 * n_a_not_b / n_not_b},
    ]
    conditional_table = pd.DataFrame(conditional_rows)

    # Basic probability checks; all probabilities use the same N=1237 people.
    p_a = n_a / n
    p_b = n_b / n
    p_a_and_b = n_a_b / n
    p_a_given_b = n_a_b / n_b
    if not np.isclose(joint_table.loc[2, 'P(event_and_B)'] +
                      joint_table.loc[2, 'P(event_and_not_B)'], 1.0):
        raise ValueError('The marginal probabilities of B and not B do not sum to one')
    if not np.isclose(p_a_given_b * p_b, p_a_and_b):
        raise ValueError('P(A|B) * P(B) does not equal P(A and B)')
    if not np.isclose(joint_table['P(event_and_B)'].iloc[0] +
                      joint_table['P(event_and_B)'].iloc[1], p_b):
        raise ValueError('The joint table does not reproduce P(B)')

    # Save definitions and aggregate results only; no participant IDs are exported.
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    definitions = {
        'phase': 2,
        'stage': 1,
        'path': 'C (NACC UDS + MRI)',
        'source_files': ['data/raw/udslong_imputed.csv',
                         'data/raw/mriscanclean.csv'],
        'unit_of_analysis': 'one participant identified by NACCID',
        'n_uds': int(len(uds)),
        'n_mri': int(len(mri)),
        'n_matched_participants': int(n),
        'n_valid_paired_participants': int(len(complete)),
        'n_excluded': int(len(data) - len(complete)),
        'event_A': {
            'definition': 'NACCUDSD == 4',
            'meaning': 'UDS cognitive-status category: dementia',
            'count': n_a,
            'empirical_probability': p_a,
            'note': 'This category does not by itself identify Alzheimer etiology.'
        },
        'event_B': {
            'definition': 'ENTORHINAL_MEAN_AVGTH < 3.0575',
            'meaning': 'Bilateral entorhinal thickness below the Phase 1 median',
            'threshold': MRI_THRESHOLD,
            'threshold_source': 'Phase 1 empirical median in this course subset',
            'count': n_b,
            'empirical_probability': p_b,
            'n_equal_to_threshold': int(complete['ENTORHINAL_MEAN_AVGTH'].eq(MRI_THRESHOLD).sum()),
            'note': 'This is a descriptive split, not a clinical MRI cut-off.'
        },
        'complement_of_B': {
            'definition': 'ENTORHINAL_MEAN_AVGTH >= 3.0575',
            'count': n_not_b,
            'empirical_probability': n_not_b / n
        },
        'random_variables': {
            'X': {'variable': 'CDRSUM', 'source': 'UDS',
                  'type': 'quantitative discrete', 'scale': '0 to 18, half-point increments',
                  'observed_min': float(complete['CDRSUM'].min()),
                  'observed_max': float(complete['CDRSUM'].max())},
            'Y': {'variable': 'ENTORHINAL_MEAN_AVGTH', 'source': 'MRI',
                  'definition': '(LH_ENTORHINAL_AVGTH + RH_ENTORHINAL_AVGTH) / 2',
                  'type': 'quantitative continuous', 'unit': 'mm (assumed in Phase 1)',
                  'observed_min': float(complete['ENTORHINAL_MEAN_AVGTH'].min()),
                  'observed_max': float(complete['ENTORHINAL_MEAN_AVGTH'].max())}
        },
        'merge': {'key': 'NACCID', 'method': 'one-to-one inner join',
                  'n_duplicate_ids_uds': 0, 'n_duplicate_ids_mri': 0,
                  'n_unmatched_ids': 0, 'n_status_mismatches': 0},
        'probability_interpretation':
            'Empirical relative frequencies within the supplied 1,237-person course subset; '
            'not population prevalence or probabilities for the general population.',
        'raw_data_modified': False,
        'participant_level_data_exported': False
    }
    (OUT / 'definitions.json').write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2), encoding='utf-8')
    joint_table.to_csv(OUT / 'joint_counts_probabilities.csv',
                       index=False, encoding='utf-8-sig', float_format='%.8f')
    conditional_table.to_csv(OUT / 'conditional_probabilities.csv',
                             index=False, encoding='utf-8-sig', float_format='%.8f')

    # The two required conditional probabilities, with group sizes in labels.
    values = [p_a_given_b, n_a_not_b / n_not_b]
    labels = [f'MRI < {MRI_THRESHOLD}\n(n={n_b})',
              f'MRI >= {MRI_THRESHOLD}\n(n={n_not_b})']
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11})
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    bars = ax.bar(labels, values)
    ax.set(title='Empirical probability of dementia status by MRI group',
           ylabel='P(NACCUDSD = 4 | MRI group)',
           ylim=(0, max(values) * 1.28))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    for bar, numerator, denominator, value in zip(
            bars, [n_a_b, n_a_not_b], [n_b, n_not_b], values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{numerator}/{denominator} = {value:.1%}',
                ha='center', va='bottom', fontsize=10)
    fig.savefig(FIG / FIG_NAME, dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'Phase 2, Stage 1: {n} valid paired participants; excluded=0')
    print(f'A (NACCUDSD=4): {n_a}/{n} = {p_a:.4%}')
    print(f'B (MRI < {MRI_THRESHOLD}): {n_b}/{n} = {p_b:.4%}')
    print(f'P(A and B)={p_a_and_b:.4%}; P(A|B)={p_a_given_b:.4%}')
    print(f'P(A|NOT B)={n_a_not_b / n_not_b:.4%}')
    print(f'Created: {OUT} and figures/{FIG_NAME}')
    print('Only aggregate results were exported; Phase 1 files were not changed.')


if __name__ == '__main__':
    main()
