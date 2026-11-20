"""NACC Phase 2, Stage 3: distributions, normal approximation, and covariance.

Covers common tasks 6 and 8, and NACC-specific tasks C.2 and C.4.
Run as Stage 3: python src/12_phase2_distributions_covariance.py
The original CSV files are read-only. Only aggregate results are exported.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'derived' / 'phase2' / 'distributions'
FIG = ROOT / 'figures'
FIGURES = {
    'cdrsum': 'phase2_03_cdrsum_distribution.png',
    'mri_hist': 'phase2_04_mri_thickness_distribution.png',
    'mri_qq': 'phase2_05_mri_normal_qq.png',
    'covariance': 'phase2_06_cdrsum_mri_covariance.png',
}


def describe(series: pd.Series, variable: str, source: str, unit: str) -> dict:
    """Empirical moments use 1/N; quartiles and IQR flags match Phase 1."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    sd_population = float(np.sqrt(variance))
    sd_sample = float(np.std(values, ddof=1)) if n > 1 else float('nan')
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75], method='linear')
    iqr = float(q3 - q1)
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)
    lower_outliers = values[values < lower_fence]
    upper_outliers = values[values > upper_fence]
    standardized = (values - mean) / sd_population if sd_population else np.zeros(n)
    skewness = float(np.mean(standardized ** 3)) if sd_population else 0.0

    all_outliers = np.concatenate([lower_outliers, upper_outliers])
    return {
        'variable': variable,
        'source': source,
        'unit': unit,
        'n': int(n),
        'mean': mean,
        'variance_empirical_ddof0': variance,
        'sd_empirical_ddof0': sd_population,
        'sd_sample_ddof1': sd_sample,
        'median': float(median),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': iqr,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'lower_fence': lower_fence,
        'upper_fence': upper_fence,
        'n_lower_outliers': int(len(lower_outliers)),
        'n_upper_outliers': int(len(upper_outliers)),
        'n_iqr_outliers': int(len(all_outliers)),
        'outlier_min': float(np.min(all_outliers)) if len(all_outliers) else None,
        'outlier_max': float(np.max(all_outliers)) if len(all_outliers) else None,
        'skewness_empirical': skewness,
    }


def load_analysis_data() -> tuple[pd.Series, pd.Series, dict]:
    """Load and pair the supplied UDS/MRI records for Stage 3 calculations."""
    uds = pd.read_csv(RAW / 'udslong_imputed.csv', dtype={'NACCID': 'string'})
    mri = pd.read_csv(RAW / 'mriscanclean.csv', dtype={'NACCID': 'string'})

    for name, frame in [('UDS', uds), ('MRI', mri)]:
        if (frame['NACCID'].isna().any() or
                frame['NACCID'].str.strip().eq('').any() or
                frame['NACCID'].duplicated().any()):
            raise ValueError(f'{name}: missing, empty, or repeated NACCID')

    if set(uds['NACCID']) != set(mri['NACCID']):
        raise ValueError('UDS and MRI do not contain the same participant IDs')

    visit_date_fields = {
        'UDS': ['NACCVNUM', 'NACCAVST', 'NACCNVST', 'VISITMO', 'VISITDAY', 'VISITYR', 'NACCFDYS'],
        'MRI': ['MRIMO', 'MRIYR', 'MRIDY', 'STUDYDATE', 'SCANDT'],
    }
    visit_date_fields_found = {
        'UDS': [field for field in visit_date_fields['UDS'] if field in uds.columns],
        'MRI': [field for field in visit_date_fields['MRI'] if field in mri.columns],
    }

    data = uds[['NACCID', 'CDRSUM']].merge(
        mri[['NACCID', 'LH_ENTORHINAL_AVGTH', 'RH_ENTORHINAL_AVGTH']],
        on='NACCID', how='inner', validate='one_to_one')

    if len(data) != len(uds) or len(data) != len(mri):
        raise ValueError('The one-to-one UDS/MRI merge changed the participant count')

    x = pd.to_numeric(data['CDRSUM'], errors='coerce')
    left = pd.to_numeric(data['LH_ENTORHINAL_AVGTH'], errors='coerce')
    right = pd.to_numeric(data['RH_ENTORHINAL_AVGTH'], errors='coerce')
    y = (left + right) / 2

    valid = x.notna() & left.notna() & right.notna()
    valid &= np.isfinite(x) & np.isfinite(left) & np.isfinite(right) & np.isfinite(y)
    valid &= x.between(0, 18) & np.isclose(x * 2, np.round(x * 2))
    valid &= (left > 0) & (right > 0) & (y > 0)
    if not valid.all():
        raise ValueError('Unexpected incomplete or invalid CDRSUM/MRI records')

    x = x.astype(float).reset_index(drop=True)
    y = y.astype(float).reset_index(drop=True)
    data_structure = {
        'uds_rows': int(len(uds)),
        'mri_rows': int(len(mri)),
        'paired_participants': int(len(data)),
        'visit_or_scan_date_fields_found': visit_date_fields_found,
    }
    return x, y, data_structure


def main() -> None:
    x, y, data_structure = load_analysis_data()
    n = len(x)

    # C.2: empirical distributions, moments, and IQR flags for the two variables.
    x_summary = describe(x, 'CDRSUM', 'UDS', 'score')
    y_summary = describe(y, 'ENTORHINAL_MEAN_AVGTH', 'MRI-derived', 'mm (assumed)')
    summaries = pd.DataFrame([x_summary, y_summary])

    # Full empirical probability mass function for the discrete CDR Sum of Boxes.
    counts = x.value_counts().sort_index()
    cdrsum_pmf = pd.DataFrame({
        'CDRSUM': counts.index.astype(float),
        'count': counts.values.astype(int),
        'probability': counts.values / n,
        'percent': 100 * counts.values / n,
    })
    # C.4: empirical population covariance uses 1/N, as fixed in the roadmap.
    x_centered = x.to_numpy() - x.mean()
    y_centered = y.to_numpy() - y.mean()
    covariance = float(np.mean(x_centered * y_centered))
    direction = 'negative' if covariance < 0 else ('positive' if covariance > 0 else 'zero')

    # Common task 6: compare the continuous MRI distribution with a fitted normal curve.
    y_mean = float(y.mean())
    y_sd = float(y.std(ddof=0))
    y_skew = y_summary['skewness_empirical']
    normal_conclusion = (
        'The histogram and Q-Q plot show a departure from the normal reference in the lower tail. '
        'The normal curve is not a complete description of the empirical distribution.'
    )

    # Stage 3 outputs: aggregate tables only; never export NACCID or participant rows.
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(OUT / 'distribution_summary.csv', index=False,
                     encoding='utf-8-sig', float_format='%.10f')
    cdrsum_pmf.to_csv(OUT / 'cdrsum_empirical_distribution.csv', index=False,
                      encoding='utf-8-sig', float_format='%.10f')
    pd.DataFrame([{
        'variable_x': 'CDRSUM',
        'variable_y': 'ENTORHINAL_MEAN_AVGTH',
        'n_paired_participants': n,
        'covariance_empirical_ddof0': covariance,
        'direction': direction,
        'unit': 'score x mm (mm assumed)',
        'correlation_calculated': False,
        'causal_interpretation': False,
    }]).to_csv(OUT / 'covariance_summary.csv', index=False,
              encoding='utf-8-sig', float_format='%.10f')

    # Figure 1: empirical PMF of the discrete CDRSUM score.
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    ax.bar(cdrsum_pmf['CDRSUM'], cdrsum_pmf['probability'], width=0.36)
    ax.set(title=f'Empirical distribution of CDR Sum of Boxes (n = {n:,})',
           xlabel='CDR Sum of Boxes (score; increments of 0.5)',
           ylabel='Empirical probability', xticks=np.arange(0, 16, 1))
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    fig.savefig(FIG / FIGURES['cdrsum'], dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Figure 2: histogram and normal curve for the continuous MRI-derived variable.
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    ax.hist(y, bins=28, density=True, alpha=0.62, edgecolor='white',
            label='Empirical MRI distribution')
    grid = np.linspace(float(y.min()) - 0.12, float(y.max()) + 0.12, 350)
    normal_pdf = np.exp(-0.5 * ((grid - y_mean) / y_sd) ** 2) / (y_sd * np.sqrt(2 * np.pi))
    ax.plot(grid, normal_pdf, linewidth=2,
            label=f'Normal curve (mean={y_mean:.3f}, SD={y_sd:.3f})')
    ax.set(title='Bilateral entorhinal thickness and a normal reference curve',
           xlabel='Mean left/right entorhinal thickness (mm; assumed)',
           ylabel='Density')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.16)
    ax.set_axisbelow(True)
    fig.savefig(FIG / FIGURES['mri_hist'], dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Figure 3: normal Q-Q plot; the reference line is y = x after standardization.
    sorted_standardized_y = np.sort((y.to_numpy() - y_mean) / y_sd)
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf(float(p)) for p in probabilities])
    fig, ax = plt.subplots(figsize=(7.2, 6.0), layout='constrained')
    ax.scatter(theoretical, sorted_standardized_y, s=13, alpha=0.48)
    low = min(float(theoretical.min()), float(sorted_standardized_y.min()))
    high = max(float(theoretical.max()), float(sorted_standardized_y.max()))
    ax.plot([low, high], [low, high], linestyle='--', linewidth=1.5,
            label='Normal reference line')
    ax.set(title='Normal Q-Q plot: bilateral entorhinal thickness',
           xlabel='Theoretical standard normal quantiles',
           ylabel='Observed standardized quantiles')
    ax.legend(frameon=False)
    ax.grid(alpha=0.16)
    fig.savefig(FIG / FIGURES['mri_qq'], dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Figure 4: paired scatter plot. Jitter affects only the visual x-coordinates.
    rng = np.random.default_rng(1405)
    x_jittered = x.to_numpy() + rng.uniform(-0.09, 0.09, size=n)
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout='constrained')
    ax.scatter(x_jittered, y.to_numpy(), alpha=0.34, s=14, edgecolors='none')
    ax.set(title='Clinical impairment and bilateral entorhinal thickness (descriptive)',
           xlabel='CDR Sum of Boxes (score; visual jitter +/-0.09)',
           ylabel='Mean left/right entorhinal thickness (mm; assumed)')
    ax.text(0.98, 0.97, f'Empirical covariance = {covariance:.4f}',
            transform=ax.transAxes, ha='right', va='top')
    ax.grid(alpha=0.16)
    fig.savefig(FIG / FIGURES['covariance'], dpi=185, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    summary = {
        'phase': 2,
        'stage': 3,
        'path': 'C (NACC UDS + MRI)',
        'n_participants': n,
        'variables': {
            'X': {'name': 'CDRSUM', 'source': 'UDS', 'type': 'quantitative discrete',
                  'distribution_file': 'cdrsum_empirical_distribution.csv', **x_summary},
            'Y': {'name': 'ENTORHINAL_MEAN_AVGTH', 'source': 'MRI-derived',
                  'type': 'quantitative continuous',
                  'definition': '(LH_ENTORHINAL_AVGTH + RH_ENTORHINAL_AVGTH) / 2',
                  **y_summary}
        },
        'normal_approximation': {
            'variable': 'ENTORHINAL_MEAN_AVGTH',
            'reference_curve_mean': y_mean,
            'reference_curve_sd_ddof0': y_sd,
            'empirical_skewness': y_skew,
            'CDRSUM_skewness': x_summary['skewness_empirical'],
            'CDRSUM_normal_approximation_note': 'Not suitable as a full-distribution approximation: the score is discrete, concentrated at zero, and strongly right-skewed.' ,
            'n_lower_iqr_outliers': y_summary['n_lower_outliers'],
            'n_upper_iqr_outliers': y_summary['n_upper_outliers'],
            'histogram_figure': FIGURES['mri_hist'],
            'qq_plot_figure': FIGURES['mri_qq'],
            'visual_conclusion': normal_conclusion,
            'formal_normality_test_performed': False
        },
        'covariance': {
            'variable_x': 'CDRSUM',
            'variable_y': 'ENTORHINAL_MEAN_AVGTH',
            'formula': '1/N * sum((X_i - mean(X)) * (Y_i - mean(Y)))',
            'denominator': n,
            'value': covariance,
            'direction': direction,
            'unit': 'score x mm (mm assumed)',
            'interpretation': (
                'The negative covariance indicates that higher-than-average CDRSUM values '
                'tend to occur with lower-than-average bilateral entorhinal thickness values '
                'in this supplied subset. Covariance is scale-dependent and does not establish causation.'
            ),
            'correlation_or_regression_performed': False
        },
        'figures': FIGURES,
        'outlier_rule': 'Values strictly below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged; none are removed.',
        'data_structure': data_structure,
        'scope_controls': {
            'hypothesis_test_performed': False,
            'normality_test_performed': False,
            'correlation_calculated': False,
            'regression_performed': False,
            'bootstrap_performed': False,
            'confidence_interval_calculated': False,
            'outliers_removed': False,
            'raw_data_modified': False,
            'participant_level_data_exported': False
        }
    }
    (OUT / 'stage3_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Phase 2, Stage 3: {n} paired participants; excluded=0')
    print(f'CDRSUM: mean={x_summary["mean"]:.6f}; variance(1/N)={x_summary["variance_empirical_ddof0"]:.6f}; IQR outliers={x_summary["n_iqr_outliers"]}')
    print(f'MRI Y: mean={y_summary["mean"]:.6f}; variance(1/N)={y_summary["variance_empirical_ddof0"]:.6f}; IQR outliers={y_summary["n_iqr_outliers"]}')
    print(f'Normal approximation: {normal_conclusion}')
    print(f'Covariance CDRSUM x MRI={covariance:.8f} ({direction})')
    print(f'Created: {OUT} and four figures in figures/')
    print('Only aggregate results were exported; no Phase 1 or earlier Phase 2 files were changed.')


if __name__ == '__main__':
    main()
