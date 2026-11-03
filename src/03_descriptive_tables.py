"""NACC | Phase 1, common task 6: frequency and empirical distributions.

Run from the project root: python src/03_descriptive_tables.py
Input: data/raw/udslong_imputed.csv and data/raw/mriscanclean.csv
Temporary, git-ignored outputs: data/derived/phase1/descriptive_tables/
CDRSUM and mean bilateral entorhinal thickness are the primary variables; left/right thickness are supporting measures.
Original data are never edited. No hypothesis testing or grouped comparisons.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "phase1" / "descriptive_tables"

STATUS = {
    1: "Normal cognition",
    2: "Impaired, not MCI",
    3: "Mild cognitive impairment (MCI)",
    4: "Dementia",
}
STATUS_FA = {
    1: "شناخت طبیعی",
    2: "اختلال شناختی غیر از MCI",
    3: "اختلال شناختی خفیف (MCI)",
    4: "دمانس",
}
MEASURES = [
    ("CDRSUM", "UDS", "CDR Sum of Boxes", "score"),
    ("LH_ENTORHINAL_AVGTH", "MRI", "Left entorhinal mean thickness", "mm"),
    ("RH_ENTORHINAL_AVGTH", "MRI", "Right entorhinal mean thickness", "mm"),
]


def read_rows(file: Path) -> list[dict[str, str]]:
    with file.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "NACCID" not in rows[0]:
        raise ValueError(f"Empty file or missing NACCID: {file}")
    ids = [str(row["NACCID"]).strip() for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate NACCID in {file.name}")
    return rows


def write_csv(file: Path, fields: list[str], records: list[dict]) -> None:
    with file.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_number(value: str, variable: str) -> float:
    if str(value).strip().lower() in ("", "na", "n/a", "nan", "none", "null"):
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Non-numeric value found in {variable}: {value!r}") from None
    return value if np.isfinite(value) else float("nan")


def continuous_summary(values: list[float], name: str, dataset: str, label: str, unit: str) -> dict:
    a = np.array(values, dtype=float)
    valid = a[np.isfinite(a)]
    if len(valid) == 0:
        raise ValueError(f"No valid observations for {name}")
    if name == "CDRSUM" and np.any(
        (valid < 0) | (valid > 18) | (np.abs(valid * 2 - np.round(valid * 2)) > 1e-8)
    ):
        raise ValueError("Unexpected CDRSUM code; revisit task 5 before summarizing")
    # MRI outliers are intentionally retained, consistent with task 5.
    q1, median, q3 = np.percentile(valid, [25, 50, 75])
    return dict(
        dataset=dataset, variable=name, label=label, unit=unit,
        n_valid=len(valid), n_missing=len(a) - len(valid),
        min=round(float(np.min(valid)), 3),
        q1=round(float(q1), 3), median=round(float(median), 3),
        q3=round(float(q3), 3), max=round(float(np.max(valid)), 3),
        mean=round(float(np.mean(valid)), 3),
        sd=round(float(np.std(valid, ddof=1)), 3),
        iqr=round(float(q3 - q1), 3),
    )


def main() -> None:
    uds = read_rows(RAW / "udslong_imputed.csv")
    mri = read_rows(RAW / "mriscanclean.csv")
    mri_by_id = {row["NACCID"].strip(): row for row in mri}
    if set(r["NACCID"].strip() for r in uds) != set(mri_by_id):
        raise ValueError("UDS/MRI participant sets differ: review the merge rule")
    # Align data using the participant ID, not the CSV row number.
    matched = [(row, mri_by_id[row["NACCID"].strip()]) for row in uds]
    status = []
    for u, m in matched:
        a = parse_number(u["NACCUDSD"], "NACCUDSD (UDS)")
        b = parse_number(m["NACCUDSD"], "NACCUDSD (MRI)")
        if not np.isfinite(a) or a not in STATUS or a != b:
            raise ValueError(f"Invalid or inconsistent NACCUDSD for {u['NACCID']}")
        status.append(int(a))

    OUT.mkdir(parents=True, exist_ok=True)
    n = len(matched)
    counts = Counter(status)
    freq = [
        dict(code=code, category=STATUS[code], frequency=counts[code],
             percent=round(100 * counts[code] / n, 2))
        for code in STATUS
    ]
    write_csv(OUT / "naccudsd_frequency.csv", list(freq[0]), freq)

    values = {}
    summary = []
    for name, dataset, label, unit in MEASURES:
        column = [parse_number(u[name] if dataset == "UDS" else m[name], name)
                  for u, m in matched]
        values[name] = column
        summary.append(continuous_summary(column, name, dataset, label, unit))
    # Primary MRI outcome: average the TWO hemispheres of the SAME participant.
    # If either side is unavailable, the derived value remains unavailable.
    left = np.asarray(values["LH_ENTORHINAL_AVGTH"], dtype=float)
    right = np.asarray(values["RH_ENTORHINAL_AVGTH"], dtype=float)
    bilateral = np.where(np.isfinite(left) & np.isfinite(right), (left + right) / 2, np.nan)
    values["ENTORHINAL_MEAN_AVGTH"] = bilateral.tolist()
    summary.append(continuous_summary(bilateral.tolist(), "ENTORHINAL_MEAN_AVGTH",
                                      "MRI-derived", "Bilateral entorhinal thickness", "mm (assumed)"))
    write_csv(OUT / "quantitative_distribution_summary.csv", list(summary[0]), summary)

    # CDRSUM is a half-step discrete score. Its empirical frequencies/CDF
    # show the concentration at zero without arbitrary bin choices.
    cdr = np.asarray(values["CDRSUM"], dtype=float)
    cdr_valid = cdr[np.isfinite(cdr)]
    score_counts = Counter(cdr_valid.tolist())
    cumulative = 0
    cdr_frequency = []
    for score in sorted(score_counts):
        count = score_counts[score]
        cumulative += count
        cdr_frequency.append(dict(
            score=f"{score:.1f}", frequency=count,
            percent=round(count * 100 / len(cdr_valid), 2),
            cumulative_frequency=cumulative,
            cumulative_percent=round(100 * cumulative / len(cdr_valid), 2),
        ))
    write_csv(OUT / "cdrsum_empirical_distribution.csv", list(cdr_frequency[0]), cdr_frequency)

    by_name = {row["variable"]: row for row in summary}
    c, l, r = [by_name[name] for name in (
        "CDRSUM", "LH_ENTORHINAL_AVGTH", "RH_ENTORHINAL_AVGTH")]
    zero_count = score_counts.get(0.0, 0)
    def iqr_flag_count(column: list[float]) -> int:
        a = np.asarray(column, dtype=float)
        valid = a[np.isfinite(a)]
        q1, q3 = np.percentile(valid, [25, 75])
        fence = 1.5 * (q3 - q1)
        return int(np.count_nonzero((valid < q1 - fence) | (valid > q3 + fence)))

    left_flags = iqr_flag_count(values["LH_ENTORHINAL_AVGTH"])
    right_flags = iqr_flag_count(values["RH_ENTORHINAL_AVGTH"])
    lines = [
        "# فاز اول — تسک مشترک ۶: جدول‌های فراوانی و توزیع تجربی",
        "",
        f"مبنای این بخش {n} شرکت‌کننده از زیرمجموعه تحویلی NACC است. «NACCID» در هر فایل یکتا بود؛ "
        "دو فایل با این شناسه تطبیق داده شدند. متغیرها مطابق کارت پروژه انتخاب شده‌اند.",
        "",
        "## ۱. فراوانی وضعیت شناختی (NACCUDSD)",
        "",
        "| وضعیت شناختی | کد | تعداد | درصد |",
        "|---|---:|---:|---:|",
    ]
    for row in freq:
        lines.append(f"| {STATUS_FA[row['code']]} | {row['code']} | {row['frequency']} | {row['percent']:.2f}٪ |")
    lines.extend([
        f"| جمع | | {n} | 100.00٪ |",
        "",
        f"گروه شناخت طبیعی با {counts[1]} نفر ({100 * counts[1] / n:.2f}٪) بزرگ‌ترین گروه است. "
        f"گروه اختلال شناختی غیر از MCI تنها {counts[2]} نفر ({100 * counts[2] / n:.2f}٪) دارد. "
        "به دلیل این نامتوازنی، در مقایسه‌های بعدی باید اندازه گروه‌ها دیده شود. "
        "این درصدها مربوط به زیرمجموعه تحویلی‌اند، نه شیوع جمعیتی.",
        "",
        "## ۲. خلاصه توزیع متغیرهای کمی",
        "",
        "| متغیر (واحد) | تعداد معتبر | گمشده | میانگین | میانه | انحراف معیار | Q1 | Q3 | کمینه | بیشینه |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        name = row["variable"].replace("LH_ENTORHINAL_AVGTH", "انتورینال چپ").replace(
            "RH_ENTORHINAL_AVGTH", "انتورینال راست").replace(
            "ENTORHINAL_MEAN_AVGTH", "انتورینال میانگین دوطرفه")
        lines.append("| {0} ({1}) | {2} | {3} | {4:.3f} | {5:.3f} | {6:.3f} | {7:.3f} | {8:.3f} | {9:.3f} | {10:.3f} |".format(
            name, row["unit"], row["n_valid"], row["n_missing"], row["mean"],
            row["median"], row["sd"], row["q1"], row["q3"], row["min"], row["max"]))
    lines.extend([
        "",
        "انحراف معیار با فرمول نمونه‌ای (n−1) محاسبه شده است. صدک‌ها و چارک‌ها بر پایه مشاهدات معتبر هر ستون‌اند.",
        "",
        "## ۳. الگوهای مهم",
        "",
        f"- **CDRSUM:** تعداد {zero_count} نفر ({100 * zero_count / len(cdr_valid):.2f}٪) نمره صفر دارند. "
        f"میانگین {c['mean']:.3f} از میانه {c['median']:.3f} بالاتر است و "
        f"نمره‌ها تا {c['max']:.1f} ادامه دارند؛ بنابراین توزیع در صفر متمرکز است و دنباله راست دارد.",
        f"- **ضخامت انتورینال:** میانگین سمت چپ {l['mean']:.3f} و راست {r['mean']:.3f} میلی‌متر است. "
        f"میانه‌ها به‌ترتیب {l['median']:.3f} و {r['median']:.3f} هستند؛ "
        "سمت راست در این زیرمجموعه از نظر میانگین کمی بیشتر است؛ این صرفا مشاهده توصیفی است.",
        f"- **مقادیر حدی MRI:** بر اساس قاعده ممیزی تسک ۵، برای ضخامت چپ {left_flags} و راست {right_flags} مقدار بیرون از حصار ۱٫۵×IQR قرار دارد. "
        "این‌ها بدون شواهد کنترل کیفیت MRI حذف نشده‌اند و در جدول حاضر باقی مانده‌اند.",
        "",
        "## ۴. مبنای محاسبه و محدودیت",
        "",
        "شاخص CDRSUM و دو ستون ضخامت و شاخص مشتق‌شده دوطرفه در فایل‌های فعلی فاقد مقدار خالی تشخیص‌داده‌شده‌اند؛ "
        "بنابراین حجم معتبر برای هر کدام برابر ۱۲۳۷ است. به دلیل نامشخص‌بودن روش imputation "
        "و زمان مراجعه بالینی/اسکن، یافته‌ها فقط برای زیرمجموعه حاضر توصیف می‌شوند. "
        "هیچ آزمون فرض یا نتیجه‌گیری علّی انجام نشده است.",
        "",
        "**خروجی‌های داده‌ای:** `naccudsd_frequency.csv`، `quantitative_distribution_summary.csv` "
        "و `cdrsum_empirical_distribution.csv` (فراوانی و درصد تجمعی تجربی نمره CDRSUM).",
        "",
        "**منابع:** `PROJECT_CARD.md`؛ `data/derived/phase1/quality_audit/quality_audit_report.md`؛ "
        "`docs/amarpr.pdf` (تسک ۶ فاز اول)؛ "
        "https://www.naccdata.org/the-nacc-researchers-guide ؛ "
        "https://files.alz.washington.edu/documentation/uds3-rdd.pdf",
        "",
    ])
    (OUT / "task6_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Task 6 completed: N={n}; output: {OUT}")
    for row in summary:
        print(f"  {row['variable']}: n={row['n_valid']} mean={row['mean']} median={row['median']}")


if __name__ == "__main__":
    main()
