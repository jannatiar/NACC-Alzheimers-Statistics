"""Phase 1, Dataset C, tasks C.1-C.3: document UDS/MRI, visits, and ID merge.

Run from anywhere: python src/06_subset_structure_merge.py
Only aggregated tables and metadata are saved. Raw NACC participant records are NEVER exported.
The original CSV files are read-only inputs.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "phase1" / "c123"

# These are groups inferred from the columns supplied for this *course subset*,
# not full official NACC data-dictionary definitions or replacements for the RDD.
# Slice endpoints are column *names* so that a changed layout fails visibly.
UDS_BLOCKS = [
    ("جمعیت‌شناختی/محیط زندگی", "SEX", "NACCAGE"),
    ("خانوادگی", "NACCFAM", "NACCFFTD"),
    ("مصرف دارو/سبک زندگی", "ANYMEDS", "QUITSMOK"),
    ("سابقه پزشکی و روان‌پزشکی", "CVHATT", "PSYCDIS"),
    ("گروه‌های دارویی", "NACCAAAS", "NACCVASD"),
    ("اندازه‌گیری/معاینه بالینی", "HEIGHT", "NORMEXAM"),
    ("رفتاری/روان‌پزشکی NPI-Q", "NPIQINF", "APPSEV"),
    ("پرسشنامه خلق GDS", "NOGDS", "NACCGDS"),
    ("عملکرد روزمره", "BILLS", "TRAVEL"),
    ("آزمون‌های شناختی", "DECSUB", "NACCAPOE"),
    ("تشخیص/وضعیت شناختی", "NACCUDSD", "NACCUDSD"),
    ("امتیازهای CDR بالینی/عملکردی", "MEMORY", "CDRGLOB"),
]

# Exact MRI region stems contained in this course file. The source data do not
# include an atlas version, scanner protocol or raw-volume units.
REGIONS = {
    "CAUDALANTERIORCINGULATE": "caudal anterior cingulate",
    "CAUDALMIDDLEFRONTAL": "caudal middle frontal",
    "CUNEUS": "cuneus",
    "ENTORHINAL": "entorhinal cortex",
    "FUSIFORM": "fusiform gyrus",
    "INFERIORPARIETAL": "inferior parietal cortex",
    "INFERIORTEMPORAL": "inferior temporal cortex",
    "ISTHMUSCINGULATE": "isthmus cingulate",
    "LATERALOCCIPITAL": "lateral occipital cortex",
    "LATERALORBITOFRONTAL": "lateral orbitofrontal cortex",
    "LINGUAL": "lingual gyrus",
    "MEDIALORBITOFRONTAL": "medial orbitofrontal cortex",
    "MIDDLETEMPORAL": "middle temporal cortex",
    "PARAHIPPOCAMPAL": "parahippocampal cortex",
    "PARACENTRAL": "paracentral cortex",
    "PARSOPERCULARIS": "pars opercularis",
    "PARSORBITALIS": "pars orbitalis",
    "PARSTRIANGULARIS": "pars triangularis",
    "PERICALCARINE": "pericalcarine cortex",
    "POSTCENTRAL": "postcentral cortex",
    "POSTERIORCINGULATE": "posterior cingulate cortex",
    "PRECENTRAL": "precentral cortex",
    "PRECUNEUS": "precuneus",
    "ROSTRALANTERIORCINGULATE": "rostral anterior cingulate",
    "ROSTRALMIDDLEFRONTAL": "rostral middle frontal",
    "SUPERIORFRONTAL": "superior frontal cortex",
    "SUPERIORPARIETAL": "superior parietal cortex",
    "SUPERIORTEMPORAL": "superior temporal cortex",
    "SUPRAMARGINAL": "supramarginal gyrus",
    "TRANSVERSETEMPORAL": "transverse temporal cortex",
    "INSULA": "insula",
}

UDS_VISIT_FIELDS = ("NACCVNUM", "NACCAVST", "NACCNVST", "VISITMO", "VISITDAY", "VISITYR", "NACCFDYS", "FORMVER")
MRI_DATE_FIELDS = ("MRIMO", "MRIYR", "MRIDY", "STUDYDATE", "SCANDT")
CORE_UDS = ("NACCID", "NACCUDSD", "CDRSUM")
CORE_MRI = ("NACCID", "NACCUDSD", "LH_ENTORHINAL_AVGTH", "RH_ENTORHINAL_AVGTH")


def read_csv(filename: str) -> tuple[list[str], list[dict[str, str]]]:
    path = RAW / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required raw file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        rows = list(reader)
    if "NACCID" not in columns:
        raise ValueError(f"{filename}: NACCID column is missing")
    return columns, rows


def write_csv(name: str, columns: list[str], rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def unique_index(rows: list[dict[str, str]], dataset: str) -> dict[str, dict[str, str]]:
    """Prevent silent many-to-many merges or use of an empty person identifier."""
    indexed = {}
    for row in rows:
        key = row["NACCID"].strip()
        if not key:
            raise ValueError(f"{dataset}: empty NACCID; no merge performed")
        if key in indexed:
            raise ValueError(f"{dataset}: repeated NACCID; first define a documented visit selection rule")
        indexed[key] = row
    return indexed


def uds_categories(columns: list[str]) -> list[dict]:
    group_for = {"NACCID": "شناسه/اتصال"}
    for label, begin, end in UDS_BLOCKS:
        if begin not in columns or end not in columns:
            raise ValueError(f"UDS variable-map boundary missing: {begin}, {end}")
        i, j = columns.index(begin), columns.index(end)
        if i > j:
            raise ValueError(f"Unexpected UDS column order: {begin}, {end}")
        for key in columns[i:j+1]:
            if key in group_for:
                raise ValueError(f"UDS column assigned to multiple families: {key}")
            group_for[key] = label
    if set(group_for) != set(columns):
        raise ValueError(f"UDS columns not categorized: {sorted(set(columns)-set(group_for))}")
    # Two functions and a diagnosis can appear in demographic/clinical blocks;
    # override with more useful analytic roles without changing source variables.
    specific = {"INDEPEND": "عملکرد روزمره", "NACCLIVS": "محیط زندگی/عملکرد",
                "RESIDENC": "محیط زندگی", "CDRSUM": "امتیازهای CDR بالینی/عملکردی",
                "NACCUDSD": "تشخیص/وضعیت شناختی"}
    out = []
    for key in columns:
        family = specific.get(key, group_for[key])
        role = "کلید اتصال" if key == "NACCID" else (
            "متغیر گروه‌بندی اصلی" if key == "NACCUDSD" else (
            "نمره بالینی اصلی" if key == "CDRSUM" else "متغیر تکمیلی در زیرمجموعه"))
        note = (
            "کدگذاری و معنای مقدارها باید با فرهنگ داده نسخه صحیح UDS بررسی شود؛ "
            "این دسته‌بندی براساس نام ستون و جایگاه آن در فایل ارسالی است."
        )
        out.append({"variable": key, "family": family, "role": role, "definition_scope": note})
    return out


def mri_categories(columns: list[str]) -> list[dict]:
    rows = []
    metrics = Counter()
    regions_seen = set()
    for key in columns:
        if key in {"NACCID", "NACCUDSD"}:
            rows.append({"variable": key, "kind": "metadata / not MRI measurement",
                         "hemisphere": "", "region_code": "", "region_name_en": "",
                         "measurement": "participant ID" if key == "NACCID" else "UDS cognitive-status code copied to MRI file",
                         "unit": "not applicable", "interpretation": "Not an independently measured MRI feature"})
            continue
        bits = key.split("_")
        if len(bits) != 3 or bits[0] not in {"LH", "RH"} or bits[2] not in {"GVOL", "AVGTH"}:
            raise ValueError(f"MRI naming pattern is unknown: {key}")
        side, region, metric = bits
        if region not in REGIONS:
            raise ValueError(f"Unknown MRI region: {region}")
        metrics[metric] += 1
        regions_seen.add(region)
        rows.append({"variable": key, "kind": "MRI regional feature",
                     "hemisphere": "left" if side == "LH" else "right",
                     "region_code": region, "region_name_en": REGIONS[region],
                     "measurement": "regional gray-matter volume" if metric == "GVOL" else "mean cortical thickness",
                     "unit": "not supplied / verify with TA" if metric == "GVOL" else "mm (project documentation)",
                     "interpretation": (
                         "A regional imaging feature, not a clinical diagnosis. "
                         "No scanner/processing protocol or GVOL volume unit was supplied."
                         if metric == "GVOL" else
                         "Mean thickness of the named cortical region in the given hemisphere; "
                         "pipeline and acquisition metadata were not supplied.")})
    if len(regions_seen) != 31 or metrics != Counter({"GVOL": 62, "AVGTH": 62}):
        raise ValueError(f"Unexpected MRI regional structure: regions={len(regions_seen)}, metrics={dict(metrics)}")
    return rows


def status_code(raw: str) -> str:
    """Normalize equivalent codes like '1', '1.0' without changing their meaning."""
    try:
        x = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError("Non-numeric NACCUDSD code in supplied data") from exc
    if not x.is_finite() or x != int(x) or int(x) not in (1, 2, 3, 4):
        raise ValueError("Unexpected NACCUDSD code: inspect source coding")
    return str(int(x))


def main() -> None:
    ucols, urows = read_csv("udslong_imputed.csv")
    mcols, mrows = read_csv("mriscanclean.csv")
    for col in CORE_UDS:
        if col not in ucols:
            raise ValueError(f"UDS primary variable is missing: {col}")
    for col in CORE_MRI:
        if col not in mcols:
            raise ValueError(f"MRI primary variable is missing: {col}")

    # C.1: complete dictionaries, including all 124 MRI measurement columns.
    umap = uds_categories(ucols)
    mmap = mri_categories(mcols)
    write_csv("c1_uds_variable_map.csv", ["variable", "family", "role", "definition_scope"], umap)
    # Publish only a concise variable glossary; never export participant rows or IDs.
    public = ROOT / "docs"
    public.mkdir(parents=True, exist_ok=True)
    with (public / "uds_variable_map.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variable", "family", "role"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(umap)
    write_csv("c1_mri_variable_map.csv", ["variable", "kind", "hemisphere", "region_code",
                                      "region_name_en", "measurement", "unit", "interpretation"], mmap)
    with (public / "mri_variable_map.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["variable", "hemisphere", "region_name_en", "measurement", "unit"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mmap)
    udf = Counter(x["family"] for x in umap)
    mdf = Counter(x["measurement"] for x in mmap)
    write_csv("c1_family_counts.csv", ["dataset", "family", "n_columns"],
              ([{"dataset": "UDS", "family": k, "n_columns": v} for k, v in sorted(udf.items())] +
               [{"dataset": "MRI", "family": k, "n_columns": v} for k, v in sorted(mdf.items())]))

    # C.2: the supplied subset has one *row* per person; historical visits are not recoverable.
    uids = unique_index(urows, "UDS")
    mids = unique_index(mrows, "MRI")
    ucounts = Counter(row["NACCID"].strip() for row in urows)
    mcounts = Counter(row["NACCID"].strip() for row in mrows)
    rows_distribution = []
    for dataset, counts in (("UDS", ucounts), ("MRI", mcounts)):
        for nrows, npeople in sorted(Counter(counts.values()).items()):
            rows_distribution.append({"dataset": dataset, "rows_per_person_in_supplied_file": nrows,
                                      "participant_count": npeople})
    write_csv("c2_rows_per_participant.csv", ["dataset", "rows_per_person_in_supplied_file", "participant_count"], rows_distribution)
    visit_columns = ([{"dataset": "UDS", "column": k, "present": str(k in ucols).lower()} for k in UDS_VISIT_FIELDS] +
                     [{"dataset": "MRI", "column": k, "present": str(k in mcols).lower()} for k in MRI_DATE_FIELDS])
    write_csv("c2_visit_metadata_availability.csv", ["dataset", "column", "present"], visit_columns)

    # C.3: join by unique NACCID and validate diagnostic-code agreement.
    common = set(uids) & set(mids)
    only_uds = set(uids) - set(mids)
    only_mri = set(mids) - set(uids)
    diagnostic_mismatches = sum(status_code(uids[k]["NACCUDSD"]) != status_code(mids[k]["NACCUDSD"]) for k in common)
    group_counts = Counter(status_code(uids[k]["NACCUDSD"]) for k in common)
    core_complete = sum(all(uids[k][c].strip() for c in CORE_UDS[1:]) and
                        all(mids[k][c].strip() for c in CORE_MRI[1:]) for k in common)
    flow = [
        ("UDS: raw rows", len(urows)), ("UDS: unique participants", len(uids)),
        ("MRI: raw rows", len(mrows)), ("MRI: unique participants", len(mids)),
        ("shared participant IDs", len(common)), ("UDS-only participant IDs", len(only_uds)),
        ("MRI-only participant IDs", len(only_mri)), ("joined participant rows (1:1)", len(common)),
        ("NACCUDSD disagreements in joined rows", diagnostic_mismatches),
        ("joined rows with nonblank core UDS/MRI fields", core_complete),
    ]
    write_csv("c3_merge_flow.csv", ["stage", "count"], [{"stage": k, "count": v} for k, v in flow])
    write_csv("c3_joined_status_frequency.csv", ["NACCUDSD", "n_participants"],
              [{"NACCUDSD": k, "n_participants": group_counts[k]} for k in sorted(group_counts)])

    summary = {
        "source_files": {"UDS": "udslong_imputed.csv", "MRI": "mriscanclean.csv"},
        "c1": {"uds_columns": len(ucols), "mri_columns": len(mcols),
               "uds_variable_families": dict(sorted(udf.items())), "mri_features": {"gray_volume": 62,
               "mean_thickness": 62, "regions": 31, "hemispheres": 2},
               "notes": ["MRI GVOL measurement unit and preprocessing/atlas provenance not supplied",
                         "Column categories are course-subset classifications; consult NACC official RDD for exact codes"]},
        "c2": {"uds_rows": len(urows), "uds_unique_people": len(uids),
               "mri_rows": len(mrows), "mri_unique_people": len(mids),
               "multiple_rows_in_supplied_uds": sum(v > 1 for v in ucounts.values()),
               "multiple_rows_in_supplied_mri": sum(v > 1 for v in mcounts.values()),
               "uds_visit_fields_available": [k for k in UDS_VISIT_FIELDS if k in ucols],
               "mri_date_fields_available": [k for k in MRI_DATE_FIELDS if k in mcols],
               "actual_longitudinal_visit_counts_observable": False,
               "representative_observation_rule": "Use the one provided UDS row and one provided MRI row per unique NACCID; cannot establish original visit-selection or time proximity"},
        "c3": {"shared_people": len(common), "uds_only_people": len(only_uds),
               "mri_only_people": len(only_mri), "joined_people": len(common),
               "diagnostic_disagreements": diagnostic_mismatches,
               "joined_core_nonblank": core_complete, "status_frequencies": dict(sorted(group_counts.items())),
               "connection_key": "NACCID", "cardinality": "one-to-one in the supplied subset",
               "scan_uds_temporal_match_verified": False,
               "mri_status_is_independent_evidence": False},
        "pending": ["Common task 10 question finalization"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c123_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(only_uds) or len(only_mri) or diagnostic_mismatches:
        raise ValueError("UDS/MRI alignment changed: review c3_merge_flow.csv before continuing")
    # Guard against accidental different participant populations in the existing analyses.
    from_previous = ROOT / "data" / "derived" / "phase1" / "descriptive_tables" / "naccudsd_frequency.csv"
    if from_previous.exists():
        with from_previous.open("r", encoding="utf-8-sig", newline="") as fh:
            previous = {str(r["code"]): int(r["frequency"]) for r in csv.DictReader(fh)}
        if previous != dict(group_counts):
            raise ValueError("C3 status frequencies differ from common task 6")
    print(f"C1: {len(ucols)} UDS variables + {len(mcols)} MRI columns documented")
    print(f"C2: {len(uids)} UDS people; one supplied row/person; original visit count unknown")
    print(f"C3: shared={len(common)}, UDS-only={len(only_uds)}, MRI-only={len(only_mri)}, status mismatches={diagnostic_mismatches}")
    print(f"Saved aggregated CSV/JSON outputs in: {OUT}")


if __name__ == "__main__":
    main()
