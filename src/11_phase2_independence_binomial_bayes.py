"""NACC Phase 2, Stage 2: independence, binomial model, moments, and Bayes.

Covers common tasks 4, 5, 7, 9 and NACC-specific task C.5.
Run after Stage 1: python src/11_phase2_independence_binomial_bayes.py
The original CSV files are read-only. Only aggregate results are exported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
EVENTS = ROOT / 'data' / 'derived' / 'phase2' / 'events'
OUT = ROOT / 'data' / 'derived' / 'phase2' / 'models'
FIG = ROOT / 'figures'
FIG_NAME = 'phase2_02_binomial_pmf.png'
MRI_THRESHOLD = 3.0575  # The same median threshold fixed in Stage 1.
N_TRIALS = 10  # A theoretical example of ten independent selections.


def main() -> None:
    # Stage 1 results are the fixed reference for this stage.
    definitions_path = EVENTS / 'definitions.json'
    joint_path = EVENTS / 'joint_counts_probabilities.csv'
    conditional_path = EVENTS / 'conditional_probabilities.csv'
    for path in [definitions_path, joint_path, conditional_path]:
        if not path.exists():
            raise FileNotFoundError(
                f'Missing Stage 1 output: {path.name}. Run Stage 1 first.')

    stage1 = json.loads(definitions_path.read_text(encoding='utf-8'))
    joint_stage1 = pd.read_csv(joint_path)
    conditional_stage1 = pd.read_csv(conditional_path)

    # Recheck one participant per supplied UDS/MRI record; do not infer visit history.
    uds = pd.read_csv(RAW / 'udslong_imputed.csv', dtype={'NACCID': 'string'})
    mri = pd.read_csv(RAW / 'mriscanclean.csv', dtype={'NACCID': 'string'})
    for name, data in [('UDS', uds), ('MRI', mri)]:
        if (data['NACCID'].isna().any() or
                data['NACCID'].str.strip().eq('').any() or
                data['NACCID'].duplicated().any()):
            raise ValueError(f'{name}: missing, empty, or repeated NACCID')

    if set(uds['NACCID']) != set(mri['NACCID']):
        raise ValueError('UDS and MRI do not contain the same participant IDs')

    data = uds[['NACCID', 'NACCUDSD']].merge(
        mri[['NACCID', 'NACCUDSD', 'LH_ENTORHINAL_AVGTH',
             'RH_ENTORHINAL_AVGTH']],
        on='NACCID', how='inner', validate='one_to_one',
        suffixes=('_uds', '_mri'))

    if (len(data) != len(uds) or len(data) != len(mri) or
            data['NACCID'].nunique() != len(data)):
        raise ValueError('The one-to-one merge changed the participant count')

    status_uds = pd.to_numeric(data['NACCUDSD_uds'], errors='raise')
    status_mri = pd.to_numeric(data['NACCUDSD_mri'], errors='raise')
    if (status_uds != status_mri).any():
        raise ValueError('NACCUDSD differs between UDS and MRI')

    left = pd.to_numeric(data['LH_ENTORHINAL_AVGTH'], errors='raise')
    right = pd.to_numeric(data['RH_ENTORHINAL_AVGTH'], errors='raise')
    y = (left + right) / 2
    if (not np.isfinite(y).all() or (y <= 0).any()):
        raise ValueError('Invalid bilateral entorhinal thickness values')

    # Recreate the two Stage 1 events and confirm that their counts did not change.
    event_a = status_uds.eq(4)
    event_b = y.lt(MRI_THRESHOLD)
    n = len(data)
    n_a = int(event_a.sum())
    n_b = int(event_b.sum())
    n_a_b = int((event_a & event_b).sum())
    n_a_not_b = int((event_a & ~event_b).sum())
    n_not_a_b = int((~event_a & event_b).sum())
    n_not_a_not_b = int((~event_a & ~event_b).sum())

    if n != stage1['n_valid_paired_participants']:
        raise ValueError('Stage 2 participant count differs from Stage 1')
    if n_a != stage1['event_A']['count'] or n_b != stage1['event_B']['count']:
        raise ValueError('Stage 2 event counts differ from Stage 1')
    if n_a_b + n_a_not_b + n_not_a_b + n_not_a_not_b != n:
        raise ValueError('The four joint event counts do not sum to N')

    # Confirm that the Stage 1 joint table is unchanged.
    joint_stage1 = joint_stage1.set_index('UDS_event')
    stage1_a = joint_stage1.loc['A']
    stage1_not_a = joint_stage1.loc['NOT A']
    if (int(stage1_a['n_B']) != n_a_b or
            int(stage1_a['n_not_B']) != n_a_not_b or
            int(stage1_not_a['n_B']) != n_not_a_b or
            int(stage1_not_a['n_not_B']) != n_not_a_not_b):
        raise ValueError('The Stage 1 joint table differs from the raw-data recheck')

    # Empirical probabilities used in the independence comparison.
    p_a = n_a / n
    p_b = n_b / n
    p_a_and_b = n_a_b / n
    p_a_times_p_b = p_a * p_b
    joint_difference = p_a_and_b - p_a_times_p_b
    p_a_given_b = n_a_b / n_b
    p_b_given_a = n_a_b / n_a
    p_a_given_not_b = n_a_not_b / (n - n_b)

    # Compare the recalculated conditional probabilities with Stage 1.
    stage1_direct = conditional_stage1.set_index('probability')
    if not np.isclose(p_a_given_b, stage1_direct.loc['P(A | B)', 'value']):
        raise ValueError('P(A|B) differs from the approved Stage 1 result')
    if not np.isclose(p_a_given_not_b, stage1_direct.loc['P(A | NOT B)', 'value']):
        raise ValueError('P(A|NOT B) differs from the approved Stage 1 result')
    # Bernoulli variable Z = 1(A), and a theoretical Binomial(10, p) model.
    p = p_a
    pmf = [math.comb(N_TRIALS, k) * p**k * (1 - p)**(N_TRIALS - k)
           for k in range(N_TRIALS + 1)]
    if not np.isclose(sum(pmf), 1.0):
        raise ValueError('The binomial probabilities do not sum to one')

    # Bayes theorem: P(A|B) = P(B|A) P(A) / P(B).
    p_a_given_b_bayes = p_b_given_a * p_a / p_b
    bayes_difference = p_a_given_b_bayes - p_a_given_b
    if not np.isclose(p_a_given_b_bayes, p_a_given_b):
        raise ValueError('Bayes result does not match the direct conditional probability')

    # The complete probability mass function for k = 0, 1, ..., 10.
    binomial_table = pd.DataFrame({
        'k': range(N_TRIALS + 1),
        'n_trials': N_TRIALS,
        'p_event': p,
        'probability': pmf
    })

    # Visit metadata were already audited in Phase 1; confirm field availability here.
    uds_visit_fields = ['NACCVNUM', 'NACCAVST', 'NACCNVST', 'VISITMO',
                        'VISITDAY', 'VISITYR', 'NACCFDYS', 'FORMVER']
    mri_visit_fields = ['MRIMO', 'MRIYR', 'MRIDY', 'STUDYDATE', 'SCANDT']
    uds_visit_fields_present = [c for c in uds_visit_fields if c in uds.columns]
    mri_visit_fields_present = [c for c in mri_visit_fields if c in mri.columns]

    # Export aggregate results only. No NACCID or participant-level rows are saved.
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    unused_comparison_file = OUT / 'independence_bayes.csv'
    if unused_comparison_file.exists():
        unused_comparison_file.unlink()
    binomial_table.to_csv(OUT / 'binomial_pmf.csv',
                          index=False, encoding='utf-8-sig', float_format='%.12f')

    summary = {
        'phase': 2,
        'stage': 2,
        'path': 'C (NACC UDS + MRI)',
        'n_participants': int(n),
        'events': {
            'A': {'definition': 'NACCUDSD == 4', 'count': n_a,
                  'probability': p_a},
            'B': {'definition': 'ENTORHINAL_MEAN_AVGTH < 3.0575',
                  'count': n_b, 'probability': p_b},
            'A_and_B': {'count': n_a_b, 'probability': p_a_and_b}
        },
        'independence': {
            'observed_joint_probability': p_a_and_b,
            'product_of_marginals': p_a_times_p_b,
            'difference_observed_minus_independence': joint_difference,
            'descriptive_interpretation':
                'The empirical frequencies in this supplied subset do not appear '
                'consistent with independence. This is descriptive only; no '
                'independence test or population-level conclusion was performed.'
        },
        'binomial_model': {
            'random_variable_Z': '1(A): 1 if the UDS status is dementia, otherwise 0',
            'random_variable_K': 'Number of A outcomes in 10 hypothetical independent selections',
            'distribution': 'Binomial(n=10, p=131/1237)',
            'n_trials': N_TRIALS,
            'p': p,
            'finite_subset_size': int(n),
            'hypothetical_sampling_fraction': N_TRIALS / n,
            'without_replacement_note': (
                'A sample of 10 drawn without replacement from the 1,237-person finite subset '
                'would follow a hypergeometric distribution exactly. Since 10/1237 is small, '
                'a binomial approximation may be reasonable under simple random sampling; '
                'the supplied data do not document such a sampling design.'
            ),
            'assumptions': [
                'The number of hypothetical selections is fixed at 10.',
                'Each selection has two outcomes: A or NOT A.',
                'The probability of A is the same for each selection and equals the empirical p.',
                'The hypothetical selections are independent.'
            ],
            'interpretation':
                'This is a theoretical teaching model based on the empirical probability '
                'in the supplied subset. It is not a new sample of ten real participants, '
                'not a without-replacement sample from the finite subset, and not a bootstrap.'
        },
        'bayes': {
            'formula': 'P(A|B) = P(B|A) * P(A) / P(B)',
            'P_B_given_A': p_b_given_a,
            'P_A': p_a,
            'P_B': p_b,
            'P_A_given_B_bayes': p_a_given_b_bayes,
            'P_A_given_B_direct': p_a_given_b,
            'absolute_difference': abs(bayes_difference),
            'checks_equal': bool(np.isclose(p_a_given_b_bayes, p_a_given_b)),
            'interpretation':
                'Probability of UDS dementia status within the below-median MRI group '
                'in this supplied subset; not MRI diagnostic accuracy or Alzheimer probability.'
        },
        'one_participant_one_record_audit': {
            'uds_rows': int(len(uds)),
            'uds_unique_NACCID': int(uds['NACCID'].nunique()),
            'uds_duplicate_NACCID': int(uds['NACCID'].duplicated().sum()),
            'mri_rows': int(len(mri)),
            'mri_unique_NACCID': int(mri['NACCID'].nunique()),
            'mri_duplicate_NACCID': int(mri['NACCID'].duplicated().sum()),
            'matched_unique_participants': int(data['NACCID'].nunique()),
            'joined_rows': int(len(data)),
            'NACCUDSD_mismatches': int((status_uds != status_mri).sum()),
            'uds_visit_metadata_fields_present': uds_visit_fields_present,
            'mri_visit_metadata_fields_present': mri_visit_fields_present,
            'rule': 'Use the one supplied UDS row and one supplied MRI row per NACCID as the analytical record.',
            'limitation':
                'One row per NACCID in the supplied files does not prove one real-world NACC visit. '
                'Visit and scan dates/visit numbers are unavailable; first, last, or nearest-to-MRI '
                'visit selection cannot be reconstructed.'
        },
        'scope_controls': {
            'formal_independence_test_performed': False,
            'p_value_calculated': False,
            'bootstrap_performed': False,
            'hypothesis_test_performed': False,
            'regression_performed': False,
            'raw_data_modified': False,
            'participant_level_data_exported': False
        }
    }
    (OUT / 'model_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # Simple PMF chart, following the visual style of the previous stage.
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11})
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    bars = ax.bar(range(N_TRIALS + 1), pmf)
    ax.set(title='Binomial distribution: number of dementia-status outcomes in 10 hypothetical selections',
           xlabel='Number of A outcomes (k)', ylabel='P(K = k)',
           xticks=range(N_TRIALS + 1), ylim=(0, max(pmf) * 1.22))
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    for bar, probability in zip(bars, pmf):
        if probability >= 0.0001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{probability:.3f}', ha='center', va='bottom', fontsize=8)
    fig.savefig(FIG / FIG_NAME, dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'Phase 2, Stage 2: {n} unique paired participants; duplicates=0')
    print(f'P(A and B)={p_a_and_b:.6f}; P(A)P(B)={p_a_times_p_b:.6f}')
    print(f'Difference={joint_difference:.6f}')
    print(f'Binomial(10, {p:.6f}): PMF values for k=0,...,10')
    print(f'Bayes P(A|B)={p_a_given_b_bayes:.8f}; direct={p_a_given_b:.8f}')
    print(f'Created: {OUT} and figures/{FIG_NAME}')
    print('Only aggregate results were exported; Phase 1 and Stage 1 files were not changed.')


if __name__ == '__main__':
    main()
