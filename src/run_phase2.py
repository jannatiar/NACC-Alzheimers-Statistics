"""Reproduce all Phase-2 tasks for Dataset C (NACC).

Run from the project root: python src/run_phase2.py
Phase 1 scripts and files are not executed or modified.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
REQUIRED = (
    ROOT / 'data' / 'raw' / 'udslong_imputed.csv',
    ROOT / 'data' / 'raw' / 'mriscanclean.csv',
)
SCRIPTS = (
    '10_phase2_events_probabilities.py',
    '11_phase2_independence_binomial_bayes.py',
    '12_phase2_distributions_covariance.py',
    '13_build_phase2_tex.py',
)


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'Phase 2 needs the original UDS and MRI CSV files. '
            f'Missing: {missing}'
        )

    for filename in SCRIPTS:
        print(f'Running {filename}...', flush=True)
        subprocess.run([sys.executable, str(SRC / filename)], check=True, cwd=ROOT)

    xelatex = shutil.which('xelatex')
    if xelatex:
        print('Compiling reports/phase2.pdf (XeLaTeX, two passes)...', flush=True)
        for _ in range(2):
            subprocess.run(
                [xelatex, '-interaction=nonstopmode', '-halt-on-error',
                 '-output-directory=reports', 'reports/phase2.tex'],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
            )
    else:
        print('XeLaTeX not found; reports/phase2.tex was created. '
              'Install XeLaTeX and a Persian font to compile the PDF.')

    print('Phase 2 complete: all 9 common tasks and 5 NACC-specific tasks are covered.')
    print('Report: reports/phase2.pdf and reports/phase2.tex')
    print('Aggregate tables: data/derived/phase2/; figures: figures/phase2_*.png')
    print('Raw data and participant-level records are not modified or exported.')


if __name__ == '__main__':
    main()
