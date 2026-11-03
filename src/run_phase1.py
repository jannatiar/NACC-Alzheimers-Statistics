"""Reproduce all 10 common and 5 NACC-specific Phase-1 tasks.

Run from the project root: python src/run_phase1.py
The original CSV files remain read-only in data/raw/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REQUIRED = ("udslong_imputed.csv", "mriscanclean.csv", "combined_adjacency_matrix.csv")
# Sorted in actual execution order, with report generation LAST.
SCRIPTS = (
    "01_inventory.py", "02_quality_audit.py", "03_descriptive_tables.py",
    "04_center_spread.py", "05_visualize_and_compare.py",
    "06_subset_structure_merge.py", "07_c4_missingness_selection.py",
    "08_c5_joint_uds_mri.py", "09_build_phase1_tex.py",
)


def main() -> None:
    raw = ROOT / "data" / "raw"
    missing = [name for name in REQUIRED if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Copy the course CSV files unchanged into {raw}. Missing: {missing}"
        )
    for filename in SCRIPTS:
        print(f"Running {filename}...", flush=True)
        subprocess.run([sys.executable, str(SRC / filename)], check=True, cwd=ROOT)

    # A complete report PDF is distributed in the archive; regenerate it when
    # TeX is present without demanding a TeX installation for data analysis.
    xelatex = shutil.which("xelatex")
    if xelatex:
        print("Compiling reports/phase1.pdf (XeLaTeX, two passes)...", flush=True)
        for _ in range(2):
            subprocess.run(
                [xelatex, "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory=reports", "reports/phase1.tex"],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
            )
    else:
        print("XeLaTeX not found; reports/phase1.tex was refreshed. "
              "The included PDF is available; compile again after installing XeLaTeX and XB Niloofar.")
    print("Phase 1 complete: 10 common + 5 NACC-specific tasks.")
    print("Figures: figures/ (4 PNG files). Report: reports/phase1.pdf and reports/phase1.tex")
    print("Local intermediate data: data/derived/phase1/ (do NOT upload).")


if __name__ == "__main__":
    main()
