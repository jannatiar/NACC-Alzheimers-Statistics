"""
NACC Project - Phase 1
Script: 01_inventory.py

Purpose:
    Create an initial inventory of the supplied UDS, MRI, and adjacency CSV files.
    This script does NOT clean, impute, modify, or overwrite raw data.

Run from the project root:
    python src/01_inventory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "phase1" / "inventory"

UDS_FILE = RAW_DIR / "udslong_imputed.csv"
MRI_FILE = RAW_DIR / "mriscanclean.csv"
ADJ_FILE = RAW_DIR / "combined_adjacency_matrix.csv"


# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and stop with a clear message if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}\n"
            "Check that the CSV file is inside data/raw."
        )

    print(f"Reading: {path.name}")
    return pd.read_csv(path)


def json_safe(value):
    """Convert NumPy/Pandas values into JSON-compatible Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def make_column_inventory(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Create one row of metadata for every column in a dataset."""
    rows = []

    for column in df.columns:
        series = df[column]
        non_missing = series.dropna()
        samples = non_missing.astype(str).drop_duplicates().head(5).tolist()

        rows.append(
            {
                "dataset": dataset_name,
                "column_name": column,
                "pandas_dtype": str(series.dtype),
                "row_count": len(df),
                "non_missing_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 3),
                "unique_non_missing": int(non_missing.nunique()),
                "sample_values_first_5_unique": " | ".join(samples),
            }
        )

    return pd.DataFrame(rows)


def dataset_overview(df: pd.DataFrame, dataset_name: str) -> dict:
    """Return high-level information about a dataset."""
    return {
        "dataset": dataset_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing_values": int((df.isna().sum() > 0).sum()),
    }


def id_audit(uds: pd.DataFrame, mri: pd.DataFrame) -> dict:
    """Audit NACCID uniqueness and overlap without merging the datasets."""
    for dataset_name, df in [("UDS", uds), ("MRI", mri)]:
        if "NACCID" not in df.columns:
            raise KeyError(f"NACCID column is missing from {dataset_name}.")

    uds_ids = uds["NACCID"].dropna().astype(str)
    mri_ids = mri["NACCID"].dropna().astype(str)

    uds_set = set(uds_ids)
    mri_set = set(mri_ids)

    result = {
        "UDS": {
            "rows": int(len(uds)),
            "non_missing_NACCID": int(uds_ids.size),
            "unique_NACCID": int(uds_ids.nunique()),
            "duplicate_NACCID_rows": int(uds_ids.duplicated().sum()),
            "missing_NACCID_rows": int(uds["NACCID"].isna().sum()),
        },
        "MRI": {
            "rows": int(len(mri)),
            "non_missing_NACCID": int(mri_ids.size),
            "unique_NACCID": int(mri_ids.nunique()),
            "duplicate_NACCID_rows": int(mri_ids.duplicated().sum()),
            "missing_NACCID_rows": int(mri["NACCID"].isna().sum()),
        },
        "overlap": {
            "shared_unique_NACCID": int(len(uds_set & mri_set)),
            "UDS_only_unique_NACCID": int(len(uds_set - mri_set)),
            "MRI_only_unique_NACCID": int(len(mri_set - uds_set)),
        },
        "interpretation_warning": (
            "Matching NACCID values indicate participant-level identifier overlap. "
            "They do NOT prove that UDS and MRI measurements were collected at "
            "the same visit or date."
        ),
    }

    # Compare NACCUDSD values for participants appearing exactly once in each file.
    if "NACCUDSD" in uds.columns and "NACCUDSD" in mri.columns:
        uds_unique = uds.loc[~uds["NACCID"].duplicated(keep=False), ["NACCID", "NACCUDSD"]]
        mri_unique = mri.loc[~mri["NACCID"].duplicated(keep=False), ["NACCID", "NACCUDSD"]]

        comparison = uds_unique.merge(
            mri_unique,
            on="NACCID",
            how="inner",
            suffixes=("_UDS", "_MRI"),
            validate="one_to_one",
        )

        comparable = comparison["NACCUDSD_UDS"].notna() & comparison["NACCUDSD_MRI"].notna()
        result["NACCUDSD_comparison"] = {
            "matched_unique_participants": int(len(comparison)),
            "both_values_non_missing": int(comparable.sum()),
            "equal_values": int(
                (
                    comparison.loc[comparable, "NACCUDSD_UDS"]
                    == comparison.loc[comparable, "NACCUDSD_MRI"]
                ).sum()
            ),
            "different_values": int(
                (
                    comparison.loc[comparable, "NACCUDSD_UDS"]
                    != comparison.loc[comparable, "NACCUDSD_MRI"]
                ).sum()
            ),
        }

    return result


def adjacency_audit(adjacency: pd.DataFrame) -> dict:
    """Inspect the adjacency matrix structure without interpreting its meaning."""
    if adjacency.shape[1] < 2:
        raise ValueError("Adjacency file does not appear to contain a label plus matrix.")

    row_labels = adjacency.iloc[:, 0].astype(str).tolist()
    matrix_df = adjacency.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    matrix = matrix_df.to_numpy(dtype=float)
    column_labels = matrix_df.columns.astype(str).tolist()

    is_square = matrix.shape[0] == matrix.shape[1]
    has_no_non_numeric = bool(np.isfinite(matrix).all())

    result = {
        "file_rows": int(adjacency.shape[0]),
        "file_columns_including_row_label": int(adjacency.shape[1]),
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "row_label_column": str(adjacency.columns[0]),
        "is_square": bool(is_square),
        "all_matrix_values_numeric": has_no_non_numeric,
        "row_labels_match_column_labels_in_same_order": bool(row_labels == column_labels),
        "row_label_count": int(len(row_labels)),
        "column_label_count": int(len(column_labels)),
    }

    if has_no_non_numeric and is_square:
        result.update(
            {
                "unique_matrix_values": sorted(
                    [float(x) for x in np.unique(matrix)]
                ),
                "is_symmetric": bool(np.allclose(matrix, matrix.T)),
                "diagonal_values": sorted(
                    [float(x) for x in np.unique(np.diag(matrix))]
                ),
                "sum_of_all_matrix_values": float(matrix.sum()),
                "off_diagonal_ones_or_nonzero_sum": float(
                    matrix.sum() - np.trace(matrix)
                ),
            }
        )

    result["interpretation_warning"] = (
        "This audit checks matrix structure only. It does not establish what "
        "zero/one means, how the matrix was generated, or whether it represents "
        "anatomical adjacency, connectivity, or another relationship."
    )

    return result


# ---------------------------------------------------------------------
# 3. Main program
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== NACC DATA INVENTORY: START ===\n")

    uds = load_csv(UDS_FILE)
    mri = load_csv(MRI_FILE)
    adjacency = load_csv(ADJ_FILE)

    # Dataset overview
    overview = pd.DataFrame(
        [
            dataset_overview(uds, "UDS"),
            dataset_overview(mri, "MRI"),
            dataset_overview(adjacency, "Adjacency"),
        ]
    )
    overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False, encoding="utf-8-sig")

    # Column inventories
    make_column_inventory(uds, "UDS").to_csv(
        OUTPUT_DIR / "uds_column_inventory.csv", index=False, encoding="utf-8-sig"
    )
    make_column_inventory(mri, "MRI").to_csv(
        OUTPUT_DIR / "mri_column_inventory.csv", index=False, encoding="utf-8-sig"
    )
    make_column_inventory(adjacency, "Adjacency").to_csv(
        OUTPUT_DIR / "adjacency_column_inventory.csv", index=False, encoding="utf-8-sig"
    )

    # Identifier audit
    ids_result = id_audit(uds, mri)
    with open(OUTPUT_DIR / "id_audit.json", "w", encoding="utf-8") as file:
        json.dump(ids_result, file, ensure_ascii=False, indent=4, default=json_safe)

    # MRI variable family counts
    mri_columns = mri.columns.astype(str).tolist()
    mri_family_result = {
        "GVOL_columns": int(sum(column.endswith("_GVOL") for column in mri_columns)),
        "AVGTH_columns": int(sum(column.endswith("_AVGTH") for column in mri_columns)),
        "other_columns": [
            column
            for column in mri_columns
            if column != "NACCID"
            and column != "NACCUDSD"
            and not column.endswith("_GVOL")
            and not column.endswith("_AVGTH")
        ],
    }
    with open(OUTPUT_DIR / "mri_variable_families.json", "w", encoding="utf-8") as file:
        json.dump(mri_family_result, file, ensure_ascii=False, indent=4)

    # Adjacency audit
    adjacency_result = adjacency_audit(adjacency)
    with open(OUTPUT_DIR / "adjacency_audit.json", "w", encoding="utf-8") as file:
        json.dump(adjacency_result, file, ensure_ascii=False, indent=4, default=json_safe)

    # Human-readable summary
    summary_lines = [
        "NACC DATA INVENTORY — RUN SUMMARY",
        "=" * 45,
        "",
        "DATASET OVERVIEW",
        overview.to_string(index=False),
        "",
        "IDENTIFIER AUDIT",
        json.dumps(ids_result, ensure_ascii=False, indent=2, default=json_safe),
        "",
        "MRI VARIABLE FAMILIES",
        json.dumps(mri_family_result, ensure_ascii=False, indent=2),
        "",
        "ADJACENCY MATRIX AUDIT",
        json.dumps(adjacency_result, ensure_ascii=False, indent=2, default=json_safe),
        "",
        "IMPORTANT:",
        "This is an inventory/audit only. No raw data were modified.",
        "NACCID overlap does not establish temporal matching of UDS and MRI.",
        "The meaning of adjacency values has not been inferred.",
    ]

    (OUTPUT_DIR / "inventory_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print("\n=== INVENTORY COMPLETED ===")
    print(f"Output folder:\n{OUTPUT_DIR}")
    print("\nFiles created:")
    for output_file in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {output_file.name}")

    print("\nDataset overview:")
    print(overview.to_string(index=False))

    print("\nShared unique NACCID:", ids_result["overlap"]["shared_unique_NACCID"])
    print("UDS-only NACCID:", ids_result["overlap"]["UDS_only_unique_NACCID"])
    print("MRI-only NACCID:", ids_result["overlap"]["MRI_only_unique_NACCID"])
    print("\n=== NACC DATA INVENTORY: FINISHED ===")


if __name__ == "__main__":
    main()