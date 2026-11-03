"""Phase 1 / Common task 7 — Center and spread of preselected UDS + MRI variables.

Run: python src/04_center_spread.py  (from the project root)
Inputs: data/raw/udslong_imputed.csv, data/raw/mriscanclean.csv
Temporary, git-ignored outputs: data/derived/phase1/center_spread/

No raw data edits, participant-level deduplication, hypothesis testing, or new variables.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "derived" / "phase1" / "center_spread"
TASK6 = ROOT / "data" / "derived" / "phase1" / "descriptive_tables" / "quantitative_distribution_summary.csv"
VARIABLES = [
    ("CDRSUM", "UDS", "CDR Sum of Boxes", "score"),
    ("LH_ENTORHINAL_AVGTH", "MRI", "Entorhinal thickness — left", "mm"),
    ("RH_ENTORHINAL_AVGTH", "MRI", "Entorhinal thickness — right", "mm"),
    ("ENTORHINAL_MEAN_AVGTH", "MRI-derived", "Bilateral mean entorhinal thickness", "mm (assumed)"),
]


def read_data(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "NACCID" not in reader.fieldnames:
            raise ValueError(f"NACCID column is absent in {path.name}")
        data: dict[str, dict[str, str]] = {}
        for row in reader:
            identifier = row["NACCID"].strip()
            if not identifier or identifier in data:
                raise ValueError(f"Blank or duplicate NACCID in {path.name}: {identifier!r}")
            data[identifier] = row
    if not data:
        raise ValueError(f"Empty data file: {path}")
    return data


def number(text: str | None, variable: str) -> float:
    if text is None or str(text).strip().lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return float("nan")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Unrecognized numeric value in {variable}: {text!r}") from exc
    return value if np.isfinite(value) else float("nan")


def describe(values: np.ndarray, variable: str, source: str, label: str, unit: str) -> dict:
    good = values[np.isfinite(values)]
    if good.size < 2:
        raise ValueError(f"Fewer than 2 observations for {variable}")
    if variable == "CDRSUM":
        if np.any((good < 0) | (good > 18) | (abs(2 * good - np.rint(2 * good)) > 1e-8)):
            raise ValueError("CDRSUM has an unrecognized score. Revisit task 5 first.")
    elif np.any(good <= 0):
        raise ValueError(f"Nonpositive MRI cortical thickness in {variable}: review task 5")
    q1, median, q3 = np.percentile(good, [25, 50, 75], method="linear")
    iqr = float(q3 - q1)
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    return {
        "source": source,
        "variable": variable,
        "label": label,
        "unit": unit,
        "n_valid": int(good.size),
        "n_missing": int(values.size - good.size),
        "mean": float(good.mean()),
        "median": float(median),
        "sd_sample": float(good.std(ddof=1)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "min": float(good.min()),
        "max": float(good.max()),
        "n_zero": int(np.count_nonzero(good == 0)),
        "iqr_lower_fence": lower,
        "iqr_upper_fence": upper,
        "iqr_outlier_count": int(np.count_nonzero((good < lower) | (good > upper))),
    }


def check_task6(rows: list[dict]) -> None:
    """If the task-6 CSV exists, ensure both tasks use the same observations/results."""
    if not TASK6.exists():
        print("Note: task-6 CSV not found; task-7 outputs computed independently.")
        return
    with TASK6.open(encoding="utf-8-sig", newline="") as handle:
        older = {r["variable"]: r for r in csv.DictReader(handle)}
    for row in rows:
        prior = older.get(row["variable"])
        if prior is None:
            raise ValueError(f"Task-6 summary misses {row['variable']}")
        for key in ("n_valid", "n_missing"):
            if int(prior[key]) != row[key]:
                raise ValueError(f"Task 6 and 7 differ in {row['variable']} / {key}")
        for oldkey, newkey in (("mean", "mean"), ("median", "median"),
                               ("sd", "sd_sample"), ("q1", "q1"),
                               ("q3", "q3"), ("iqr", "iqr")):
            if abs(float(prior[oldkey]) - row[newkey]) > 0.00101:
                raise ValueError(f"Task 6 and 7 differ in {row['variable']} / {oldkey}")
    print("Task-6 summaries: consistent.")


def write_summary(rows: list[dict]) -> None:
    # Keep numeric precision in CSV; report is rounded to 3 decimals.
    headers = list(rows[0])
    with (OUT / "center_spread_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})


def write_report(rows: list[dict]) -> None:
    cdr, left, right, bilateral = rows
    n = cdr["n_valid"] + cdr["n_missing"]
    def cell(r: dict, key: str) -> str:
        return f"{r[key]:.3f}"

    lines = [
        "# فاز اول — تسک مشترک ۷: مرکز و پراکندگی",
        "",
        "## ۱. داده و روش محاسبه",
        "",
        f"محاسبات روی {n} شرکت‌کننده از زیرمجموعه تحویلی NACC انجام شد. "
        "داده‌های UDS و MRI با `NACCID` در سطح شرکت‌کننده تطبیق داده شدند. "
        "چهار شاخص کمی (سه ستون اصلی و میانگین دوطرفه مشتق‌شده) مطابق کارت پروژه و تسک ۶ هستند؛ "
        "`NACCUDSD` طبقه‌ای است و میانگین یا انحراف معیار کدهای آن محاسبه نمی‌شود.",
        "",
        "انحراف معیار به‌صورت نمونه‌ای (`ddof=1`)، چارک‌ها با روش صدک خطی، و دامنه بین‌چارکی "
        "به صورت `IQR = Q3 - Q1` محاسبه شدند. در گزارش، اعداد به سه رقم اعشار گرد شده‌اند.",
        "",
        "## ۲. جدول شاخص‌های مرکز و پراکندگی",
        "",
        "| شاخص | CDRSUM (امتیاز) | ضخامت چپ | ضخامت راست | میانگین دوطرفه |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, numeric in [
        ("تعداد معتبر", "n_valid", False), ("گمشده", "n_missing", False),
        ("میانگین", "mean", True), ("میانه", "median", True),
        ("انحراف معیار نمونه‌ای", "sd_sample", True),
        ("چارک اول (Q1)", "q1", True), ("چارک سوم (Q3)", "q3", True),
        ("دامنه بین‌چارکی (IQR)", "iqr", True),
        ("کمینه", "min", True), ("بیشینه", "max", True),
    ]:
        values = [(cell(r, key) if numeric else str(r[key])) for r in rows]
        lines.append(f"| {label} | {' | '.join(values)} |")

    zero_percent = 100 * cdr["n_zero"] / cdr["n_valid"]
    lines.extend([
        "",
        "## ۳. مناسب‌بودن شاخص‌ها و تفسیر",
        "",
        "**نمره CDRSUM (UDS):**",
        f"{cdr['n_zero']} نفر ({zero_percent:.2f}٪) نمره صفر دارند. "
        f"میانه {cell(cdr,'median')} و میانگین {cell(cdr,'mean')} است؛ "
        f"دامنه مشاهدات از {cell(cdr,'min')} تا {cell(cdr,'max')} امتداد دارد. "
        "تمرکز زیاد داده‌ها در صفر و دنباله راست باعث می‌شود میانه و IQR "
        "از میانگین و انحراف معیار برای نمایش وضعیت معمول و پراکندگی مرکزی مناسب‌تر باشند؛ "
        "به‌ویژه بازه میانگین ± انحراف معیار حتی حد پایینی منفی می‌دهد که برای این نمره ممکن نیست. "
        "میانگین و انحراف معیار نیز برای کامل‌بودن توصیف گزارش شدند.",
        "",
        "**ضخامت انتورینال چپ (MRI):**",
        f"میانگین {cell(left,'mean')} و میانه {cell(left,'median')} میلی‌متر هستند. "
        f"انحراف معیار {cell(left,'sd_sample')} و IQR برابر {cell(left,'iqr')} میلی‌متر است. "
        f"{left['iqr_outlier_count']} مقدار بیرون از حصارهای ۱٫۵×IQR قرار می‌گیرد. "
        "میانگین و انحراف معیار خلاصه کلی مناسبی می‌دهند، اما با توجه به مقادیر انتهایی، "
        "میانه و IQR نیز برای توصیف مقاوم‌تر لازم‌اند.",
        "",
        "**ضخامت انتورینال راست (MRI):**",
        f"میانگین {cell(right,'mean')} و میانه {cell(right,'median')} میلی‌متر هستند. "
        f"انحراف معیار {cell(right,'sd_sample')} و IQR برابر {cell(right,'iqr')} میلی‌متر است. "
        f"{right['iqr_outlier_count']} مقدار بیرون از حصارهای ۱٫۵×IQR قرار می‌گیرد. "
        "برای این متغیر نیز گزارش هم‌زمان میانگین/انحراف معیار و میانه/IQR مناسب است. "
        "این آمار به‌تنهایی توزیع نرمال را اثبات نمی‌کنند.",
        "",
        "**ضخامت میانگین دوطرفه (MRI):**",
        f"میانگین {cell(bilateral,'mean')} و انحراف معیار {cell(bilateral,'sd_sample')} (با فرض واحد میلی‌متر) است؛ "
        "برای هر فرد از متوسط ضخامت انتورینال چپ و راست همان فرد ساخته شده و پاسخ اصلی پژوهش است.",
        "",
        "## ۴. نحوه برخورد با موارد خاص",
        "",
        "- همه شاخص‌های اصلی و میانگین دوطرفه مشتق‌شده در داده تحویلی برای ۱۲۳۷ فرد معتبرند.",
        f"- مطابق تسک ۵، {left['iqr_outlier_count']} مقدار در چپ و {right['iqr_outlier_count']} مقدار در راست "
        "با قاعده ۱٫۵×IQR علامت‌گذاری شدند؛ بدون شواهد کنترل کیفیت تصویربرداری "
        "هیچ‌کدام حذف یا جایگزین نشدند.",
        "- تفاوت مشاهده‌شده میان خلاصه‌های دو نیمکره صرفا توصیفی است و آزمون فرض یا ادعای علّی انجام نشده است.",
        "- روش `imputation` فایل UDS و تطابق زمانی UDS/MRI در زیرمجموعه تحویلی مستند نیست.",
        "",
        "## منابع",
        "",
        "- `docs/amarpr.pdf`، صفحه ۶، تسک مشترک ۷ فاز اول.",
        "- `PROJECT_CARD.md` و گزارش‌های تسک‌های ۵ و ۶ همین پروژه.",
        "- https://www.naccdata.org/about-nacc-data",
        "- https://www.naccdata.org/data-forms-and-documentation/",
        "- https://www.naccdata.org/the-nacc-researchers-guide",
        "",
    ])
    (OUT / "task7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    uds = read_data(RAW / "udslong_imputed.csv")
    mri = read_data(RAW / "mriscanclean.csv")
    if set(uds) != set(mri):
        raise ValueError("Participant ID sets differ; review merge rule from tasks 1–6")
    ids = sorted(uds)
    for ident in ids:
        a = number(uds[ident].get("NACCUDSD"), "NACCUDSD (UDS)")
        b = number(mri[ident].get("NACCUDSD"), "NACCUDSD (MRI)")
        if not np.isfinite(a) or a not in (1, 2, 3, 4) or a != b:
            raise ValueError(f"Unmatched/invalid NACCUDSD for participant {ident}")
    rows: list[dict] = []
    for name, source, label, unit in VARIABLES:
        if name == "ENTORHINAL_MEAN_AVGTH":
            l = np.array([number(mri[ident].get("LH_ENTORHINAL_AVGTH"), "left thickness") for ident in ids])
            r = np.array([number(mri[ident].get("RH_ENTORHINAL_AVGTH"), "right thickness") for ident in ids])
            values = np.where(np.isfinite(l) & np.isfinite(r), (l + r) / 2, np.nan)
        else:
            dataset = uds if source == "UDS" else mri
            values = np.array([number(dataset[ident].get(name), name) for ident in ids], dtype=float)
        rows.append(describe(values, name, source, label, unit))
    check_task6(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_summary(rows)
    write_report(rows)
    print(f"Task 7 complete: {len(ids)} participants. Outputs: {OUT}")
    for row in rows:
        print(f"  {row['variable']}: mean={row['mean']:.3f}, median={row['median']:.3f}, "
              f"انحراف معیار={row['sd_sample']:.3f}, IQR={row['iqr']:.3f}")


if __name__ == "__main__":
    main()
