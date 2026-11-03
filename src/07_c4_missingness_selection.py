"""NACC Phase 1, Dataset C, task C.4: missingness and multimodal selection.

Runs on the *supplied* course subset, not the original full NACC cohort.
No participant-level rows are written. Blank cells, documented special codes,
and potential quality flags are three distinct concepts. Raw files are read-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'derived' / 'phase1' / 'c4_missingness'

# Only field-specific codes already documented in task 5 are counted as missing
# *by code*. Neither a zero nor an arbitrary 88/99 in another field is missing.
UDS_FIELDS = {
    'NACCUDSD': {'special': set(), 'allowed': {1, 2, 3, 4}},
    'CDRSUM': {'special': set(), 'bounds': (0, 18)},
    'NACCAGE': {'special': set()},
    'SEX': {'special': set(), 'allowed': {1, 2}},
    'EDUC': {'special': {99}, 'bounds': (0, 36), 'integer_only': True},
    'MOCATOTS': {'special': {88, -4}, 'bounds': (0, 30), 'integer_only': True},
}
MRI_MAIN = ('LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH')


def load_frame(filename: str) -> pd.DataFrame:
    path = RAW / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing course dataset: {path}')
    return pd.read_csv(path, dtype={'NACCID': 'string'}, low_memory=False)


def ensure_unique(df: pd.DataFrame, label: str) -> None:
    if 'NACCID' not in df.columns or df.NACCID.isna().any() or (df.NACCID.str.strip() == '').any():
        raise ValueError(f'{label} missing/blank NACCID')
    if df.NACCID.duplicated().any():
        raise ValueError(f'{label} has repeated participants; do not create a many-to-many join')


def missing_report(series: pd.Series, *, dataset: str, special: set[int] | None = None,
                   bounds: tuple[float, float] | None = None,
                   allowed: set[int] | None = None, positive: bool = False,
                   integer_only: bool = False) -> dict:
    """Do not use global numeric sentinels or silently recode quality flags."""
    special = special or set()
    raw_blank = series.isna() | series.astype('string').str.strip().isin(['', 'NA', 'N/A', 'nan', 'None', 'null'])
    numbers = pd.to_numeric(series, errors='coerce')
    finite = pd.Series(np.isfinite(numbers.to_numpy(dtype=float)), index=series.index)
    nonnumeric = (~raw_blank & ~finite)
    code_special = (~raw_blank & finite & numbers.isin(special))
    plausible = (~raw_blank & finite & ~code_special)
    outside = pd.Series(False, index=series.index)
    if bounds is not None:
        outside |= plausible & ((numbers < bounds[0]) | (numbers > bounds[1]))
    if allowed is not None:
        outside |= plausible & (~numbers.isin(allowed))
    if positive:
        outside |= plausible & (numbers <= 0)
    noninteger = plausible & (abs(numbers - numbers.round()) > 1e-8) if integer_only else pd.Series(False, index=series.index)
    bad = outside | noninteger
    n = len(series)
    return {
        'dataset': dataset,
        'variable': series.name,
        'n_rows': n,
        'raw_blank_n': int(raw_blank.sum()),
        'raw_blank_pct': round(float(100 * raw_blank.sum() / n), 3),
        'documented_special_code_n': int(code_special.sum()),
        'documented_special_pct': round(float(100 * code_special.sum() / n), 3),
        'nonnumeric_or_nonfinite_n': int(nonnumeric.sum()),
        'quality_flag_n': int(bad.sum()),
        'noninteger_score_flag_n': int(noninteger.sum()),
        'out_of_plausible_range_or_nonpositive_n': int(outside.sum()),
        'analysis_eligible_n': int((plausible & ~bad).sum()),
        'analysis_eligible_pct': round(float(100 * (plausible & ~bad).sum() / n), 3),
    }


def main() -> None:
    uds, mri = load_frame('udslong_imputed.csv'), load_frame('mriscanclean.csv')
    ensure_unique(uds, 'UDS')
    ensure_unique(mri, 'MRI')
    for col in UDS_FIELDS:
        if col not in uds:
            raise ValueError(f'Required UDS variable not found: {col}')
    features = [c for c in mri if c not in ('NACCID', 'NACCUDSD')]
    if len(features) != 124 or not all(c in mri for c in MRI_MAIN):
        raise ValueError('MRI feature layout changed; inspect mapping before analysing')
    shared = set(uds.NACCID) & set(mri.NACCID)
    only_uds = len(set(uds.NACCID) - shared)
    only_mri = len(set(mri.NACCID) - shared)

    key = [missing_report(uds[c], dataset='UDS', **UDS_FIELDS[c]) for c in UDS_FIELDS]
    key += [missing_report(mri[c], dataset='MRI', positive=True) for c in MRI_MAIN]
    all_mri = [missing_report(mri[c], dataset='MRI', positive=True) for c in features]

    # The file contains a single record for *every* UDS participant. It cannot
    # identify or describe unscanned people from the full NACC source population.
    availability = pd.DataFrame([{
        'category': 'UDS participants with a supplied MRI record',
        'n_people': len(shared), 'pct_of_uds_file': round(100 * len(shared) / len(uds), 3)},
        {'category': 'UDS participants without a supplied MRI record',
         'n_people': only_uds, 'pct_of_uds_file': round(100 * only_uds / len(uds), 3)},
        {'category': 'MRI participants without a supplied UDS record',
         'n_people': only_mri, 'pct_of_uds_file': ''},
    ])
    # Describe the MRI-covered group, *without inventing* a non-MRI comparison.
    status_counts = uds['NACCUDSD'].value_counts().sort_index()
    baseline = pd.DataFrame([{'variable': 'NACCUDSD', 'category': str(int(c)),
                              'n': int(v), 'pct_of_mri_available': round(100 * v / len(shared), 3)}
                             for c, v in status_counts.items()] +
                            [{'variable': 'SEX', 'category': str(int(c)),
                              'n': int(v), 'pct_of_mri_available': round(100 * v / len(shared), 3)}
                             for c, v in uds['SEX'].value_counts().sort_index().items()])
    age = pd.to_numeric(uds['NACCAGE'])
    cdr = pd.to_numeric(uds['CDRSUM'])
    overview = {'n_uds': len(uds), 'n_mri': len(mri), 'n_matched': len(shared),
                'n_uds_without_mri': only_uds, 'n_mri_without_uds': only_mri,
                'can_compare_mri_vs_no_mri_in_supplied_subset': only_uds > 0,
                'missingness_scope': 'observed course CSV files after any unknown upstream imputation',
                'original_missingness_recoverable': False,
                'imputation_method_known': False,
                'scan_uds_time_alignment_known': False,
                'mri_feature_columns': len(features),
                'mri_features_with_raw_blanks': sum(r['raw_blank_n'] > 0 for r in all_mri),
                'mri_raw_blank_cells_total': sum(r['raw_blank_n'] for r in all_mri),
                'mri_nonpositive_cells_flagged': sum(r['out_of_plausible_range_or_nonpositive_n'] for r in all_mri),
                'moca_fractional_score_flags': next(r['noninteger_score_flag_n'] for r in key if r['variable']=='MOCATOTS'),
                'moca_out_of_range_flags': next(r['out_of_plausible_range_or_nonpositive_n'] for r in key if r['variable']=='MOCATOTS'),
                'age_mean_mri_group': float(age.mean()),
                'age_median_mri_group': float(age.median()),
                'cdrsum_mean_mri_group': float(cdr.mean()),
                'cdrsum_median_mri_group': float(cdr.median()),
                'limitations': [
                    'No group without an MRI record exists in supplied UDS subset; comparative selection analysis is not identifiable.',
                    'A record present for everyone does not prove absence of selection upstream when preparing the subset.',
                    'Raw blank count is not equivalent to original missingness in imputed UDS; special codes are variable-specific.',
                    'Nonpositive MRI feature values are quality flags, not automatically imputed or deleted observations.',
                ]}
    # Consistency with C.3 and common task 6.
    prior = ROOT / 'data' / 'derived' / 'phase1' / 'c123' / 'c123_summary.json'
    if prior.is_file():
        previous = json.loads(prior.read_text(encoding='utf-8'))
        if previous['c3']['joined_people'] != len(shared) or previous['c3']['uds_only_people'] != only_uds:
            raise ValueError('C4 MRI availability does not agree with task C3 merge audit')

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key).to_csv(OUT / 'c4_main_variable_missingness.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_mri).to_csv(OUT / 'c4_all_mri_feature_missingness.csv', index=False, encoding='utf-8-sig')
    availability.to_csv(OUT / 'c4_mri_availability.csv', index=False, encoding='utf-8-sig')
    baseline.to_csv(OUT / 'c4_uds_profile_with_mri.csv', index=False, encoding='utf-8-sig')
    (OUT / 'c4_summary.json').write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'C4: {len(shared)} UDS people with MRI, {only_uds} without MRI; no non-MRI comparator in this subset.')
    print(f'C4: all {len(features)} MRI features audited; raw blanks={overview["mri_raw_blank_cells_total"]}, '
          f'nonpositive MRI values flagged={overview["mri_nonpositive_cells_flagged"]}')
    print(f'C4: field-specific codes EDU=99: {next(r for r in key if r["variable"]=="EDUC")["documented_special_code_n"]}, '
          f'MoCA special: {next(r for r in key if r["variable"]=="MOCATOTS")["documented_special_code_n"]}.')
    print(f'C4: summary files: {OUT}')


if __name__ == '__main__':
    main()
