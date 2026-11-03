"""Regenerate the sole version-controlled Phase-1 report, reports/phase1.tex.

Run `python src/run_phase1.py` (recommended) after putting the supplied CSVs
inside data/raw/. All intermediate diagnostic files live in git-ignored
`data/derived/phase1/`; the report is one TeX file (with four referenced PNG figures).
Common tasks 1–10 and Dataset-C C.1–C.5 are covered; analysis remains descriptive.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / 'data' / 'derived' / 'phase1'
OUT = ROOT / 'reports' / 'phase1.tex'
C123 = AUD / 'c123'


def csvrows(relative):
    with (AUD / relative).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jsonread(relative):
    return json.loads((AUD / relative).read_text(encoding='utf8'))


def coded(name):
    return r'\Code{' + str(name) + '}'


def trow(*cells):
    return ' & '.join(str(c) for c in cells) + r' \\' + '\n'


def table(headers, rows, specs, *, small=True):
    s = ('\\small\n' if small else '') + '\\begin{longtable}{' + specs + '}\n'
    s += '\\toprule\n' + trow(*headers) + '\\midrule\n\\endfirsthead\n'
    s += '\\toprule\n' + trow(*headers) + '\\midrule\n\\endhead\n'
    for r in rows:
        s += trow(*r)
    s += '\\bottomrule\n\\end{longtable}\n\\normalsize\n'
    return s


def nf(num, places=3):
    return f'{float(num):,.{places}f}'



# Editorial style for all regenerated documents: no hamza/tanwin in Persian prose,
# and inline bold exclusively for a paragraph-opening "label:".
_HAMZA_MAP = {
    0x0623: "ا", 0x0625: "ا", 0x0624: "و", 0x0626: "ی", 0x0621: "",
    0x06C0: "ه", 0x0671: "ا", 0x0654: "", 0x064B: "", 0x064C: "", 0x064D: "",
}


def editorial_tex_style(tex: str) -> str:
    tex = tex.translate(_HAMZA_MAP)
    token = r"\textbf{"
    pos = 0
    while True:
        start = tex.find(token, pos)
        if start == -1:
            break
        opening = start + len(token) - 1
        depth = 1
        end = opening + 1
        while end < len(tex) and depth:
            if tex[end] == "{":
                depth += 1
            elif tex[end] == "}":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("Unbalanced textbf braces")
        content = tex[opening + 1:end - 1]
        prefix = tex[tex.rfind("\n", 0, start) + 1:start].strip()
        # Allow a preceding indentation or a LaTeX noindent command only.
        paragraph_start = prefix in ("", r"\noindent")
        keep = paragraph_start and content.strip().endswith((":", "："))
        if keep:
            pos = end
        else:
            tex = tex[:start] + content + tex[end:]
            pos = start + len(content)
    return tex


def build():
    inventory = csvrows('inventory/dataset_overview.csv')
    ids = jsonread('inventory/id_audit.json')
    mrifam = jsonread('inventory/mri_variable_families.json')
    adj = jsonread('inventory/adjacency_audit.json')
    quality = jsonread('quality_audit/quality_summary.json')
    details = csvrows('quality_audit/column_quality_overview.csv')
    flags = csvrows('quality_audit/flagged_cases.csv')
    freq = csvrows('descriptive_tables/naccudsd_frequency.csv')
    cdr = csvrows('descriptive_tables/cdrsum_empirical_distribution.csv')
    task6 = csvrows('descriptive_tables/quantitative_distribution_summary.csv')
    spread = csvrows('center_spread/center_spread_summary.csv')
    groups = csvrows('visual_comparison/task9_group_descriptive_summary.csv')
    chartinfo = jsonread('visual_comparison/task8_9_summary.json')
    c123 = json.loads((C123 / 'c123_summary.json').read_text(encoding='utf-8'))
    c4 = jsonread('c4_missingness/c4_summary.json')
    c4key = csvrows('c4_missingness/c4_main_variable_missingness.csv')
    c4mri = csvrows('c4_missingness/c4_all_mri_feature_missingness.csv')
    c4profile = csvrows('c4_missingness/c4_uds_profile_with_mri.csv')
    c5 = jsonread('c5_joint/c5_summary.json')
    c5all = csvrows('c5_joint/c5_joint_overall.csv')
    c5group = csvrows('c5_joint/c5_joint_by_status.csv')
    with (C123 / 'c1_family_counts.csv').open(encoding='utf-8-sig', newline='') as f:
        c1families = list(csv.DictReader(f))
    with (C123 / 'c1_mri_variable_map.csv').open(encoding='utf-8-sig', newline='') as f:
        c1mri = list(csv.DictReader(f))
    with (C123 / 'c2_rows_per_participant.csv').open(encoding='utf-8-sig', newline='') as f:
        c2rows = list(csv.DictReader(f))
    with (C123 / 'c3_merge_flow.csv').open(encoding='utf-8-sig', newline='') as f:
        c3flow = list(csv.DictReader(f))

    assert len(inventory) == 3 and len(spread) == 4 and len(task6) == 4
    assert len(groups) == 16 and chartinfo['n_figures'] == 3
    assert c123['c1']['uds_columns'] == 180 and c123['c1']['mri_columns'] == 126
    assert len(c1mri) == 126 and len(c2rows) == 2
    assert len(c4key) == 8 and len(c4mri) == 124 and len(c5all) == 4 and len(c5group) == 16
    assert c4['n_uds'] > 0 and c4['n_uds'] == c4['n_mri'] == c4['n_matched'] == c5['n_matched_people'] == c5['n_joint_valid_people']
    assert c4['n_uds_without_mri'] == 0 and not c4['can_compare_mri_vs_no_mri_in_supplied_subset']
    assert c4['mri_raw_blank_cells_total'] == sum(int(x['raw_blank_n']) for x in c4mri)
    assert sum(int(x['n_joint_pairs']) for x in c5all) == 4 * c5['n_joint_valid_people']
    assert c5['n_excluded_from_joint_by_field_rules'] == 0
    if not (ROOT / 'figures' / c5['scatter_figure']).is_file():
        raise FileNotFoundError('Task C5 scatter figure missing; run task C5 before report')
    for x in c5group:
        matches = [y for y in groups if y['variable']==x['variable'] and y['category_code']==x['group_code']]
        assert len(matches) == 1 and abs(float(matches[0]['mean']) - float(x['mean'])) < 0.0011
    assert c123['c3']['joined_people'] == int(c123['c2']['uds_unique_people'])
    assert c123['c3']['joined_people'] == int(c123['c2']['mri_unique_people'])
    assert chartinfo['n_participants'] == 1237
    assert len(chartinfo['figures']) == 3 and chartinfo['n_figures_in_final_report'] == 4
    for file in chartinfo['figures']:
        if not (ROOT / 'figures' / file).is_file():
            raise FileNotFoundError(f'Figure missing: {file}. Run task 8/9 first.')
    assert len(freq) == 4 and len(details) == sum(int(x["columns"]) for x in inventory if x["dataset"] != "Adjacency"), 'Unexpected column counts; inspect source CSV.'
    assert int(ids['overlap']['shared_unique_NACCID']) == sum(int(x['frequency']) for x in freq)
    assert int(quality['overlap_naccid']) == int(ids['overlap']['shared_unique_NACCID'])
    assert {x['variable'] for x in spread} == {x['variable'] for x in task6}
    # Verify that the reported statistics agree across independently generated scripts.
    for x in spread:
        y = next(y for y in task6 if y['variable'] == x['variable'])
        for key, newer in [('mean','mean'),('median','median'),('sd','sd_sample'),('q1','q1'),('q3','q3')]:
            assert abs(float(y[key]) - float(x[newer])) < 0.0011

    freq_names = {'1':'شناخت طبیعی', '2':'اختلال شناختی غیر از MCI', '3':'اختلال شناختی خفیف (MCI)', '4':'دمانس'}
    n = sum(int(x['frequency']) for x in freq)
    uds_columns = next(int(x['columns']) for x in inventory if x['dataset']=='UDS')
    zero_count = next(int(v['frequency']) for v in cdr if float(v['score']) == 0)
    shown_special = [
        ('NACCBMI','888.8'),('HEIGHT','88.8'),('WEIGHT','888'),
        ('EDUC','99'),('MOCATOTS','88'),('NACCGDS','88'),
        ('DIABETES','9'),('HYPERTEN','9'),('BILLS','8 / 9'),('SHOPPING','8 / 9')
    ]
    special_by_var = Counter(x['column'] for x in flags if x['issue'] == 'documented_nonanalytic_code')
    flagged_breakdown = Counter((x['column'], x['issue']) for x in flags)
    invalid_summary = [
        ('WEIGHT', 'out_of_documented_range', 'دامنه مستند'),
        ('NACCBMI', 'out_of_documented_range', 'دامنه مستند'),
        ('MOCATOTS', 'out_of_documented_range', 'دامنه مستند'),
        ('MOCATOTS', 'noninteger_score', 'نمره اعشاری'),
        ('NACCGDS', 'out_of_documented_range', 'دامنه مستند'),
        ('NACCGDS', 'noninteger_score', 'نمره اعشاری'),
        ('DIABETES', 'invalid_category', 'کد طبقه‌ای نامجاز'),
        ('HYPERTEN', 'invalid_category', 'کد طبقه‌ای نامجاز'),
        ('BILLS', 'invalid_category', 'کد طبقه‌ای نامجاز'),
        ('SHOPPING', 'invalid_category', 'کد طبقه‌ای نامجاز'),
    ]
    source = r"""
\documentclass[a4paper,11pt]{article}
\usepackage[left=2cm, right=2cm, top=2cm]{geometry}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage[colorlinks = true]{hyperref}
\usepackage{graphicx}
\usepackage{float}
\usepackage{needspace}
\usepackage{fancyhdr}
\usepackage{color,xcolor}
\usepackage{enumitem}
\usepackage{longtable,array,booktabs}
\usepackage{setspace}
\usepackage{xcolor}
\usepackage{xepersian}
\usepackage{mathtools}
\usepackage{flexisym}
\usepackage{fontspec}


\settextfont{XB Niloofar}
\setdigitfont{XB Niloofar}
\setlatintextfont{XB Niloofar}
\setmonofont{DejaVu Sans Mono}

\newcommand{\Code}[1]{\lr{\texttt{\detokenize{#1}}}}
\newcommand{\En}[1]{\lr{#1}}
\graphicspath{{figures/}{../figures/}}
\setstretch{1.19}
\setlength{\parindent}{0pt}\setlength{\emergencystretch}{2em}
\setlength{\parskip}{0.35em}
\renewcommand{\arraystretch}{1.23}
\hypersetup{hidelinks,pdfauthor={NACC Dataset C -- Statistics Course},pdftitle={Phase 1 NACC UDS and MRI}}
\begin{document}

\thispagestyle{fancy}
\fancyhf{}
\lhead{

 \includegraphics[width=3cm]{logo.PNG}
 \vspace{-2pt}
}
\chead{\leavevmode{\Huge به نام خدا}\vspace{15pt}\\  {\Large علیرضا جنتی}\vspace{10pt}\\{\small 403101672}\vspace{5pt}}
\rhead{

  {\large   پروژه درس آمار}

\vspace{10pt}   

{\large  فاز اول}

\vspace{10pt}

ترم 14051
\vspace{-7pt}

}
\cfoot{\thepage}

\leavevmode
\vspace{20pt}


\vspace{1cm}

\begingroup
\hypersetup{linkcolor=blue}
\tableofcontents
\pagenumbering{gobble}
\endgroup
\clearpage
\pagenumbering{arabic} 
\pagebreak
\section{دامنه، سوال پژوهش و مبنای گزارش}
این گزارش فقط مربوط به فاز اول مسیر ج و زیرمجموعه تحویلی تیم درس از داده‌های \En{NACC} است.

\textbf{عنوان نهایی:} بررسی ارتباط شدت اختلال بالینی با میانگین ضخامت قشر انتورینال در زیرمجموعه \En{UDS--MRI} مرکز \En{NACC}.

\textbf{سوال اصلی نهایی:} در زیرمجموعه تحویلی \En{NACC}، چه ارتباط آماری‌ای میان نمره شدت اختلال بالینی \Code{CDRSUM} و میانگین ضخامت قشر انتورینال مغز در دو نیمکره وجود دارد و این ارتباط با درنظرگرفتن سن و جنس شرکت‌کنندگان چگونه تغییر می‌کند؟

\textbf{واحد تحلیل:} یک شرکت‌کننده با شناسه یکتای \Code{NACCID}. هر فایل تحویلی از هر فرد یک ردیف دارد؛ این موضوع، یکتایی مراجعه در پایگاه اصلی یا هم‌زمانی MRI و ارزیابی بالینی را ثابت نمی‌کند.


\subsection*{نقش متغیرها در سوال نهایی}
"""
    source += table(['متغیر', 'مقیاس/منبع', 'نقش'], [
        (coded('CDRSUM'), 'کمی گسسته / UDS', 'توضیحی اصلی (X): شدت بالینی'),
        (coded('ENTORHINAL_MEAN_AVGTH'), 'کمی پیوسته / MRI مشتق‌شده', 'پاسخ اصلی (Y): میانگین ضخامت دوطرفه'),
        (coded('NACCAGE'), 'کمی / UDS', 'توضیحی کمکی: سن'),
        (coded('SEX'), 'طبقه‌ای / UDS', 'توضیحی کمکی: جنس'),
        (coded('NACCUDSD'), 'طبقه‌ای / UDS', 'گروه‌بندی صرفا توصیفی'),
        (coded('NACCID'), 'شناسه / هر دو', 'کلید اتصال؛ نه متغیر تحلیلی'),
    ], r' p{4.9cm} p{4.9cm} p{5.0cm} ')
    source += r"""\textbf{فرضیه علمی اولیه:} انتظار می‌رود نمره‌های بالاتر \Code{CDRSUM} با میانگین ضخامت انتورینال پایین‌تر همراه باشند؛ این انتظار از پیش نتیجه تحلیل را تعیین نمی‌کند. نقش سن و جنس در این فاز صرفا معرفی می‌شود و بررسی آماری تعدیل‌شده به مراحل بعدی موکول است.

\section{منشا داده و سازوکار جمع‌آوری}
\En{NACC} داده‌های چندمرکزی مراکز پژوهش بیماری آلزایمر (\En{ADRC}) را نگه‌داری می‌کند.
\En{UDS} شامل اطلاعات جمعیت‌شناختی، وضعیت بالینی، عملکرد و شناخت است؛ داده MRI
از فرایند تصویربرداری و پردازش تصویری حاصل می‌شود. گردآوری ماهیت \textbf{مشاهده‌ای}
دارد و مداخله یا تخصیص تصادفی برای این پرسش انجام نشده است.
تاریخِ انجماد داده، بازه دقیق ورود نمونه به زیرمجموعه، روش ساخت آن و منشا پردازش ویژگی‌های MRI در CSVهای تحویلی مستند نشده‌اند.
برای شاخص‌های ضخامت، واحد \En{mm} بر پایه قرارداد رایج این نوع خروجی MRI نوشته شده.
\subsection*{فهرست و ابعاد فایل‌های تحویلی}
"""
    source += table(['فایل/بخش', 'ردیف', 'ستون'], [
        (coded('udslong_imputed.csv') if x['dataset']=='UDS' else coded('mriscanclean.csv') if x['dataset']=='MRI' else coded('combined_adjacency_matrix.csv'),
         x['rows'], x['columns']) for x in inventory
    ], r' p{9.0cm} r r ')
    source += rf"""
ماتریس مجاورت در CSV دارای \textbf{{۶۲ ردیف و ۶۳ ستون شامل ستون برچسب}} است؛
بخش عددی آن ماتریسی \En{{62 $\times$ 62}} است.

\section{{واحد مشاهده و ساختار داده}}
در داده‌های طولی اصلی NACC ممکن است یک شرکت‌کننده چند مراجعه \En{{UDS}} داشته باشد.
اما هر یک از فایل‌های تحویلی \En{{UDS}} و \En{{MRI}} دقیقا \textbf{{{n}}} ردیف و
\textbf{{{n}}} شناسه یکتا دارند. تعداد مراجعات واقعی هر فرد، شماره مراجعه انتخاب‌شده،
تاریخ معاینه و تاریخ اسکن در زیرمجموعه فعلی در دسترس نیستند؛ پس نباید
ساختار طولی پایگاه اصلی را به‌اشتباه به رکوردهای تحویلی نسبت داد.

"""
    source += table(['کنترل شناسه', 'مقدار'], [
        ('شناسه یکتای UDS',ids['UDS']['unique_NACCID']),
        ('شناسه یکتای MRI',ids['MRI']['unique_NACCID']),
        ('شناسه مشترک UDS و MRI',ids['overlap']['shared_unique_NACCID']),
        ('فقط در UDS',ids['overlap']['UDS_only_unique_NACCID']),
        ('فقط در MRI',ids['overlap']['MRI_only_unique_NACCID']),
        ('اختلاف مقدار NACCUDSD میان دو فایل',ids['NACCUDSD_comparison']['different_values']),
        ('شناسه تکراری UDS', ids['UDS']['duplicate_NACCID_rows']),
        ('شناسه تکراری MRI', ids['MRI']['duplicate_NACCID_rows']),
    ], r' p{10cm} r ')
    source += r"""
\textbf{رابطه رکوردها:}\quad
\En{NACCID} در یک فایل \En{UDS}، به همان \En{NACCID} در فایل \En{MRI} متصل می‌شود؛
\textbf{اتصال یک‌به‌یک در سطح شناسه} با \textbf{تطبیق زمانی تصویر و مراجعه} متفاوت است.

\section{نمونه‌گیری، انتخاب و قابلیت تعمیم}
ورود افراد به مراکز ADRC به طراحی‌ها و مسیرهای ارجاع و مشارکت وابسته است.
راهنمای رسمی NACC تاکید می‌کند مجموعه داده آن نمونه احتمالی و نماینده
جمعیت شهر، ایالت یا کشور نیست. افزون بر این، قاعده گزینش زیرمجموعه آماده این درس
نامشخص است. بنابراین درصدها، میانگین‌ها و نمودارهای این پروژه فقط توصیف
\textbf{همین زیرمجموعه تحویلی} هستند؛ نمی‌توان آن‌ها را شیوع آلزایمر یا دمانس
در کل جامعه دانست. «دمانس» در \Code{NACCUDSD} نیز لزوما «دمانس ناشی از آلزایمر» نیست.

\section{طبقه‌بندی متغیرها و نقش آن‌ها}
"""
    source += table(['متغیر','تعریف و واحد','مقیاس','نقش'], [
        (coded('NACCID'),'شناسه شرکت‌کننده','اسمی/شناسه','اتصال'),
        (coded('NACCUDSD'),'۱ طبیعی، ۲ اختلال غیر MCI، ۳ MCI، ۴ دمانس','طبقه‌ای؛ کدها کمیت نیستند','گروه‌بندی'),
        (coded('CDRSUM'),'مجموع نمرات شش حوزه CDR، امتیاز','کمی گسسته؛ گام ۰٫۵','شاخص بالینی'),
        (coded('LH_ENTORHINAL_AVGTH'),'میانگین ضخامت انتورینال چپ، میلی‌متر','کمی پیوسته','شاخص MRI'),
        (coded('RH_ENTORHINAL_AVGTH'),'میانگین ضخامت انتورینال راست، میلی‌متر','کمی پیوسته','شاخص MRI'),
    ],r' p{4.1cm} p{5.0cm} p{3.0cm} p{2.2cm} ')
    source += rf"""

\section{{ممیزی کیفیت داده و تصمیم‌ها}}
\subsection*{{خلاهای ظاهری و کدهای ویژه}}
بررسی خام خانه‌های خالی و مقدار \En{{NaN}} در هر دو جدول UDS و MRI عدد \textbf{{صفر}}
را نشان داد؛ اما \textbf{{این معادل صفر بودن گمشدگی مفهومی نیست}}. مقادیر خاصی مانند ۸، ۹، ۸۸، ۹۹
یا ۸۸۸ فقط با توجه به \textbf{{تعریف همان متغیر}} می‌توانند معنای «نامعلوم» یا «انجام نشده» داشته باشند.
در ده ستون منتخب با تعریف بررسی‌شده، \textbf{{{quality['issues'].get('documented_nonanalytic_code',0)}}}
سلول دارای کد خاص غیرقابل استفاده مستقیم در محاسبه عددی علامت‌گذاری شده است.

"""
    source += table(['متغیر','کد خاص بررسی‌شده','تعداد علامت'], [
        (coded(var), code, special_by_var[var]) for var,code in shown_special
    ], r' p{5cm} p{6cm} r ')
    source += r"""
\textbf{تصمیم:} کدهای خاص فقط برای متغیرهایی که تعریف معتبر دارند در خلاصه کمی کنار گذاشته
می‌شوند. مقدار خام تغییر نمی‌کند؛ برای جدول‌های طبقه‌ای، دسته نامعلوم/نامربوط
می‌تواند جداگانه نشان داده شود. حذف همگانی عدد ۹ یا ۸ از تمام ستون‌ها غلط است.

\subsection*{غربالگری دامنه، نمره اعشاری و کد طبقه‌ای}
"""
    source += table(['متغیر', 'نوع نشانه', 'تعداد پرچم'], [
        (coded(v), reason, flagged_breakdown[(v,issue)]) for v, issue,reason in invalid_summary
    ], r' p{5cm} p{7cm} r ')
    source += rf"""
این اعداد \textbf{{پرچم سلول/رخداد}} هستند؛ در یک سطر ممکن است چند نشانه وجود داشته باشد.
از جمع‌کردن آن‌ها برای به‌دست‌آوردن شمار افراد مسیله‌دار اجتناب شد.
به‌ویژه بخشی از نمره‌های \Code{{MOCATOTS}} و \Code{{NACCGDS}}
اعشاری‌اند که با نمره خام صحیح‌مقدار ناسازگارند.
برای نمره‌های خام یا کدهای طبقه‌ای، گردکردن خودکار انجام نشده است.
\subsection*{{تکرار، سازگاری و MRI}}
"""
    source += table(['کنترل','نتیجه'], [
        ('سطر کامل تکراری UDS',quality['UDS']['duplicate_full_rows']),
        ('سطر کامل تکراری MRI',quality['MRI']['duplicate_full_rows']),
        ('عدم تطابق NACCUDSD میان دو وجه',quality['NACCUDSD_mismatch']),
        ('ناسازگاری CDRSUM و مجموع شش حوزه',quality['CDRSUM_sum_mismatch']),
        ('ناسازگاری BMI در مشاهدات غربال‌شده',quality['BMI_formula_mismatch_within_valid_ranges']),
        ('ناسازگاری فشار خون در محدوده‌های کنترل‌شده',quality['blood_pressure_logic_failure']),
        ('حجم MRI صفر یا منفی (پرچم سلول)',quality['issues'].get('nonpositive_mri_volume',0)),
        ('پرت IQR ضخامت انتورینال چپ',flagged_breakdown[('LH_ENTORHINAL_AVGTH','entorhinal_iqr_outlier')]),
        ('پرت IQR ضخامت انتورینال راست',flagged_breakdown[('RH_ENTORHINAL_AVGTH','entorhinal_iqr_outlier')]),
    ], r' p{11cm} r ')
    source += rf"""
برای ضخامت انتورینال چپ و راست، تعداد پرچم‌های \En{{IQR}} به‌ترتیب
{flagged_breakdown[('LH_ENTORHINAL_AVGTH','entorhinal_iqr_outlier')]} و
{flagged_breakdown[('RH_ENTORHINAL_AVGTH','entorhinal_iqr_outlier')]} است؛
مجموع {quality['issues'].get('entorhinal_iqr_outlier',0)} پرچم الزاما همین تعداد شرکت‌کننده یکتا نیست.
پرت آماری به‌خودی‌خود خطای اندازه‌گیری نیست و \textbf{{هیچ فردی صرفا به خاطر IQR حذف نمی‌شود}}.
دو حجم MRI غیرمثبت برای بررسی کیفیت قطعه‌بندی نشان‌دارند.
در {mrifam['GVOL_columns']+mrifam['AVGTH_columns']} ستون ویژگی MRI، بررسی فنی ساختار و مقدار انجام شده است؛
بدون فراداده کیفیت اسکن، صحت زیستی/بالینی این ویژگی‌ها اثبات نمی‌شود.

\textbf{{تصمیم نهایی تسک ۵:}}
سه ستون مبنای سوال اصلی و میانگین دوطرفه مشتق‌شده در کنترل‌های پایه برای توصیف مناسب‌اند؛
داده خام دستکاری نمی‌شود، کدهای نامعتبر متغیرهای فرعی وارد خلاصه عددی نمی‌شوند،
و منشا جایگذاری و تطبیق زمانی UDS/MRI «نامشخص» ثبت می‌ماند.

\section{{جداول فراوانی و توزیع تجربی}}
\subsection*{{فراوانی وضعیت شناختی}}
"""
    source += table(['وضعیت','کد','فراوانی','درصد'], [
        (freq_names[x['code']],x['code'],x['frequency'],nf(x['percent'],2)+r'\%') for x in freq
    ]+[('جمع','',n,'100.00'+r'\%')],r' p{8.0cm} r r r ')
    source += rf"""
گروه «شناخت طبیعی» با {freq[0]['frequency']} نفر بیشترین حجم و گروه
«اختلال شناختی غیر از MCI» با {freq[1]['frequency']} نفر کمترین حجم را دارد.
این درصدها فراوانی در نمونه‌اند و هیچ ادعای شیوع جمعیتی ندارند.

\subsection*{{خلاصه توزیع متغیرهای کمی}}
"""
    source += table(['متغیر','n معتبر','میانگین','میانه','انحراف معیار نمونه‌ای'], [
        (coded(x['variable']),x['n_valid'],nf(x['mean']),nf(x['median']),nf(x['sd'])) for x in task6
    ],r' p{5.8cm} r r r r ')
    source += rf"""
نمره \Code{{CDRSUM}} در \textbf{{{zero_count}}} مورد از {n} شرکت‌کننده،
یعنی {nf(zero_count/n*100,2)}\%، برابر صفر است؛ میانه صفر، میانگین مثبت
و دنباله راست مشخص دارد. ضخامت انتورینال راست در این مجموعه
میانگین بزرگ‌تری از چپ دارد؛ این تفاوت صرفا توصیفی است.

\subsection*{{توزیع تجربی کامل نمره CDRSUM}}
"""
    source += table(['امتیاز','فراوانی','درصد','فراوانی تجمعی','درصد تجمعی'], [
        (nf(x['score'],1),x['frequency'], nf(x['percent'],2)+r'\%',x['cumulative_frequency'],nf(x['cumulative_percent'],2)+r'\%') for x in cdr
    ],r' r r r r r ')
    source += r"""
این جدول شکل تجربی توزیع نمره‌های نیم‌واحدی را نمایش می‌دهد. گردکردن نمایش درصدها
ممکن است باعث اختلاف ناچیز با جمع دقیق مقادیر محاسبه‌شده شود.

\section{مرکز، پراکندگی و تناسب شاخص‌ها}
آماره‌ها روی همان چهار شاخصِ تسک قبل و در سطح شناسه یکتا محاسبه شده‌اند.
انحراف معیار با مخرج \En{$n-1$}، چارک‌ها با روش صدک خطی و
\En{$IQR=Q_3-Q_1$} به دست آمده‌اند؛ اعشارهای جدول فقط گردکردن \textbf{نمایش} هستند.

"""
    sdict = {x['variable']:x for x in spread}
    keys = ['CDRSUM','ENTORHINAL_MEAN_AVGTH','LH_ENTORHINAL_AVGTH','RH_ENTORHINAL_AVGTH']
    spec = [
        ('تعداد معتبر','n_valid',False),('گمشده','n_missing',False),
        ('میانگین','mean',True),('میانه','median',True),
        ('انحراف معیار نمونه‌ای','sd_sample',True),
        ('چارک اول Q1','q1',True),('چارک سوم Q3','q3',True),
        ('دامنه بین‌چارکی IQR','iqr',True),
        ('کمینه','min',True),('بیشینه','max',True),
    ]
    source += table(['شاخص','CDRSUM','میانگین دوطرفه','انتورینال چپ','انتورینال راست'], [
        (name,*[(nf(sdict[k][key]) if isfloat else sdict[k][key]) for k in keys])
        for name,key,isfloat in spec
    ], r' p{5.2cm} r r r r ')
    source += r"""
\textbf{تعبیر آماری:} توزیع \Code{CDRSUM} در صفر متراکم و به راست چوله است؛
پس برای وضعیت معمول و تغییرپذیری مرکزی، میانه و دامنه بین‌چارکی در کنار
میانگین و انحراف معیار مناسب‌اند. برای میانگین دوطرفه و دو ضخامت MRI نیز گزارش هم‌زمان
میانگین/انحراف معیار و میانه/IQR، به‌ویژه به‌علت مقادیر انتهایی، مناسب است.
از این شاخص‌ها به‌تنهایی نمی‌توان نرمال‌بودن توزیع را نتیجه گرفت.
پرت‌های MRI در همه خلاصه‌ها و نمودارهای تسک‌های ۸ و ۹ حفظ شده‌اند.

"""
    source += r"""\clearpage
\section{مصورسازی درست}
چهار نمودار معنادار در کل این گزارش به کار رفته است: نمودار فراوانی وضعیت شناختی
\Code{NACCUDSD}، نمودار توزیع تجربی نمره گسسته \Code{CDRSUM}،
هیستوگرام چگالی ضخامت انتورینال چپ و راست \En{MRI} با بازه‌های یکسان،
و نمودار پراکنش جفت‌شده \En{UDS/MRI} که به‌عنوان نمودار چهارم،
نمره بالینی را در برابر میانگین ضخامت دوطرفه (پاسخ اصلی) نمایش می‌دهد.
این چهار نمودار علاوه بر تسک مشترک مصورسازی، توصیف مشترک خواسته‌شده در بخش ج.۵ را هم پوشش می‌دهند.
همه شکل‌ها محور و واحد دارند و هیچ آزمون آماری یا نتیجه علّی از آن‌ها گرفته نمی‌شود.
در هیستوگرام MRI، محور عمودی چگالی در هر میلی‌متر است، نه تعداد افراد.

\begin{figure}[H]\centering
\includegraphics[width=.82\linewidth]{phase1_08_01_cognitive_group_counts.png}
\caption{فراوانی چهار گروه وضعیت شناختی در زیرمجموعه تحویلی؛ تفاوت اندازه گروه‌ها آشکار است.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=.82\linewidth]{phase1_08_02_cdrsum_discrete_distribution.png}
\caption{فراوانی تجربی امتیاز نیم‌واحدی \Code{CDRSUM}؛ تمرکز روی صفر و دنباله راست.}
\end{figure}
\clearpage
\begin{figure}[H]\centering
\includegraphics[width=.82\linewidth]{phase1_08_03_mri_thickness_distribution.png}
\caption{توزیع ضخامت انتورینال چپ و راست؛ محور افقی برحسب میلی‌متر و محور عمودی چگالی است.}
\end{figure}

\begin{figure}[H]\centering
\includegraphics[width=.82\linewidth]{phase1_c5_joint_uds_mri_scatter.png}
\caption{نمایش جفت‌شده نمره بالینی UDS و میانگین ضخامت انتورینال دوطرفه MRI. رنگ‌ها گروه‌های شناختی‌اند؛ جابه‌جایی افقی تصادفی با بذر ثابت در بازه \En{$\pm0.09$} صرفا برای نمایش نقاط هم‌پوشان است و در محاسبات اعمال نشده است.}
\label{fig:joint-uds-mri}
\end{figure}
\textbf{دلیل انتخاب:} نمودار میله‌ای برای متغیر طبقه‌ای و نمره گسسته، هیستوگرام برای متغیر پیوسته
و نمودار پراکنش برای نمایش توصیفی دو وجه مناسب است. مقایسه گروه‌ها در بخش بعدی به کمک
جداول جداگانه انجام شده؛ در آن جداول تمام مقادیر، از جمله نقاط دورافتاده، حفظ شده‌اند.

\clearpage

\section{مقایسه توصیفی اولیه}
طبق سوال نهایی، چهار شاخص کمّی روی \textbf{همان چهار گروه}\,
\Code{NACCUDSD} مقایسه شده‌اند. همه خلاصه‌ها با اتصال یک‌به‌یک بر اساس
\Code{NACCID} محاسبه می‌شوند؛ هر شرکت‌کننده یک بار در هر جدول حضور دارد.
در جدول‌های زیر انحراف معیار با مخرج \En{$n-1$} و دامنه بین‌چارکی \En{$IQR=Q_3-Q_1$}
گزارش شده‌اند.
"""
    group_titles = [
        ('CDRSUM', 'نمره بالینی CDRSUM (امتیاز)'),
        ('LH_ENTORHINAL_AVGTH', 'ضخامت انتورینال چپ (میلی‌متر)'),
        ('RH_ENTORHINAL_AVGTH', 'ضخامت انتورینال راست (میلی‌متر)'),
        ('ENTORHINAL_MEAN_AVGTH', 'میانگین ضخامت انتورینال دوطرفه (میلی‌متر فرضی)'),
    ]
    for var, title in group_titles:
        source += r'\Needspace{12\baselineskip}\subsection*{' + title + '}'+ '\n'
        rows = [x for x in groups if x['variable'] == var]
        assert len(rows) == 4
        assert sum(int(x['n_group']) for x in rows) == n
        source += table(['گروه', 'تعداد', 'میانگین', 'میانه', 'انحراف معیار', 'IQR'], [
            (freq_names[x['category_code']], x['n_valid'],
             nf(x['mean']), nf(x['median']), nf(x['sd_sample']), nf(x['iqr']))
            for x in rows
        ], r' p{3.5cm} r r r r r ')
    source += r"""
\Needspace{12\baselineskip}
\noindent\textbf{تفسیر مقایسه‌ای:}
در گروه شناخت طبیعی (۸۴۷ نفر)، میانه \Code{CDRSUM} برابر صفر است؛
در گروه \En{MCI} (۲۵۲ نفر) میانه برابر ۱ و در گروه دمانس (۱۳۱ نفر) برابر ۴٫۵ است.
در همین ترتیب، میانگین ضخامت انتورینال چپ از حدود ۳٫۰۴۶ به ۲٫۸۶۳
و ۲٫۶۷۴ میلی‌متر و میانگین سمت راست از حدود ۳٫۱۳۲ به ۲٫۹۵۳ و
۲٫۷۶۴ میلی‌متر تغییر می‌کند. این تفاوت‌ها فقط درباره زیرمجموعه درس صادق‌اند.
گروه اختلال غیر \En{MCI} تنها ۷ مشاهده دارد و مقادیر آن با نوسان نمونه‌ای
زیاد و بدون فرض ترتیب بالینی عددی کدهای گروه‌بندی تفسیر می‌شوند.
"""
    # Dataset-C C1-C3: all numbers are read from the generated aggregated files.
    source += r"""
\section{نقشه کامل زیرمجموعه UDS و MRI}
\textbf{قلمرو:} دو فایل واقعی تحویلی تیم درس مبنای طبقه‌بندی هستند؛ کد \Code{NACCID}
شناسه فرد و \Code{NACCUDSD} برچسب وضعیت شناختی است.

\subsection*{طبقه‌بندی تمام ستون‌های UDS}
"""
    source += table(['خانواده متغیرها', 'تعداد ستون'], [
        (x['family'], x['n_columns']) for x in c1families if x['dataset']=='UDS'
    ], r' p{11cm} r ')
    source += r"""فهرست \textbf{تمام ۱۸۰ ستون} با گروه و نقش در فایل
\path{data/derived/phase1/c123/c1_uds_variable_map.csv} قابل مشاهده و بازتولید است.
گروه‌های اصلی شامل جمعیت‌شناختی و زندگی، سابقه پزشکی و خانوادگی، مصرف دارو،
عملکرد روزمره، خلق و رفتار، آزمون‌های شناختی و نمره‌های \En{CDR} هستند.
در سوال نهایی، \Code{CDRSUM} متغیر توضیحی، میانگین ضخامت دوطرفه پاسخ اصلی و \Code{NACCUDSD} گروه‌بندی توصیفی است.

\subsection*{معنای ۱۲۴ ویژگی MRI}
\textbf{الگوی نام‌گذاری:} پیشوند \Code{LH} نیمکره چپ، \Code{RH} نیمکره راست؛
پسوند \Code{AVGTH} «میانگین ضخامت قشر ناحیه» (میلی‌متر در گزارش درس) و
پسوند \Code{GVOL} «حجم ماده خاکستری همان ناحیه» است. واحد خام ستون‌های حجمی
و نوع پردازش و نسخه اطلس از فایل‌ها قابل احراز نیست؛ هیچ واحد حجمی حدسی ثبت نشده است.
\Code{NACCUDSD} در فایل MRI یک برچسب بالینی است، نه اندازه‌گیری مستقل تصویری.
"""
    source += table(['نوع ستون MRI', 'تعداد'], [
        ('ویژگی حجمی ناحیه‌ای، دو نیمکره', 62),
        ('میانگین ضخامت قشر ناحیه‌ای، دو نیمکره', 62),
        ('شناسه و برچسب شناختی (نه ویژگی تصویر)', 2),
        ('جمع', c123['c1']['mri_columns']),
    ], r' p{11cm} r ')
    region_names = sorted({x['region_name_en'] for x in c1mri if x['region_name_en']})
    assert len(region_names) == 31
    source += r"""\textbf{۳۱ ناحیه MRI} (هرکدام در دو نیمکره و با دو نوع اندازه‌گیری):
"""
    names = [r'\En{' + name + '}' for name in region_names]
    source += table(['ناحیه', 'ناحیه'], [
        (names[i], names[i+16] if i+16 < len(names) else '') for i in range(16)
    ], r' p{7cm} p{7cm} ')
    source += r"""فایل \Code{c1_mri_variable_map.csv} در پوشه \Code{data/derived/phase1/c123/}
برای \textbf{تمام ۱۲۶ ستون MRI} نام ناحیه، نیمکره، نوع اندازه‌گیری و واحد معلوم یا نامشخص را جداگانه ثبت می‌کند.
دو ستون مورد استفاده این پروژه، \Code{LH_ENTORHINAL_AVGTH} و
\Code{RH_ENTORHINAL_AVGTH}، میانگین ضخامت قشر انتورینال چپ و راست‌اند.
ماتریس مجاورت ۶۲ ناحیه نیز در موجودی‌گیری ثبت شده اما متغیرهای پاسخ این پرسش را نمی‌سازد.

\section{ساختار شرکت‌کننده و مراجعه}
در پایگاه کامل \En{NACC}، \En{UDS} عموما ساختار طولی دارد و یک ردیف
می‌تواند نماینده یک \textbf{مراجعه} یک فرد باشد؛ اما در زیرمجموعه درسی کنونی
\textbf{هر فرد در هر فایل دقیقا یک ردیف دارد}. این شمارش به معنی آن نیست که
هر فرد در پایگاه اصلی فقط یک مراجعه داشته است.
"""
    source += table(['فایل', 'تعداد سطر هر فرد در فایل تحویلی', 'تعداد افراد'], [
        (r'\En{' + x['dataset'] + '}', x['rows_per_person_in_supplied_file'], x['participant_count'])
        for x in c2rows
    ], r' p{3.2cm} p{7.5cm} r ')
    source += r"""\textbf{فراداده غایب:} ستون‌های شماره یا شمار مراجعات
\Code{NACCVNUM}، \Code{NACCAVST}، \Code{NACCNVST}، تاریخ ارزیابی
\Code{VISITMO}، \Code{VISITDAY}، \Code{VISITYR} و تاریخ‌های
اسکن مانند \Code{MRIYR} و \Code{MRIMO} در فایل‌های تحویلی وجود ندارند.
پس \textbf{تعداد واقعی مراجعات و توزیع آن قابل برآورد از این فایل‌ها نیست}.

\textbf{قاعده مشاهده شاخص:} برای فازهای بعدی فقط همان یک سطر تحویلی از
\En{UDS} و یک سطر از \En{MRI} به‌ازای هر \Code{NACCID}
استفاده می‌شود. این یک \textbf{قاعده استفاده از زیرمجموعه آماده} است، نه
قاعده تازه برای انتخاب آخرین یا اولین مراجعه واقعی. هر فرد یک مشاهده مستقل
در سطح شناسه خواهد بود؛ ادعایی درباره هم‌زمانی اسکن و معاینه نداریم.

\section{اتصال یک‌به‌یک UDS و MRI}
اتصال بر اساس \Code{NACCID} و \textbf{پس از کنترل یکتایی و تهی نبودن شناسه}
انجام شده است. کد اگر شناسه تکراری ببیند، به‌جای ساختن اتصال چندبه‌چند
و چندبار شمردن فرد، با خطا متوقف می‌شود. کدهای \Code{NACCUDSD}
در دو فایل به شکل عددی هم‌معنا مقایسه شده‌اند؛ به‌عنوان مثال \En{1.0}
در فایل UDS با \En{1} در MRI یک طبقه است و عدم تطابق واقعی نیست.
"""
    flow_labels = {
        'UDS: raw rows':'سطرهای خام UDS', 'UDS: unique participants':'شرکت‌کنندگان یکتای UDS',
        'MRI: raw rows':'سطرهای خام MRI', 'MRI: unique participants':'شرکت‌کنندگان یکتای MRI',
        'shared participant IDs':'شرکت‌کنندگان مشترک', 'UDS-only participant IDs':'فقط UDS',
        'MRI-only participant IDs':'فقط MRI', 'joined participant rows (1:1)':'ردیف‌های اتصال نهایی یک‌به‌یک',
        'NACCUDSD disagreements in joined rows':'اختلاف واقعی کد وضعیت شناختی',
        'joined rows with nonblank core UDS/MRI fields':'سطرهای دارای پنج ستون کنترل‌شده غیرخالی',
    }
    source += table(['مرحله اتصال', 'تعداد'], [
        (flow_labels[x['stage']], x['count']) for x in c3flow
    ], r' p{11cm} r ')
    source += r"""
\textbf{محدودیت زمانی:} راهنمای رسمی \En{NACC} صریح می‌گوید اسکن‌ها
لزومی ندارد با همان مراجعه \En{UDS} ثبت شده باشند. نبودن تاریخ‌ها در
زیرمجموعه ما مانع کنترل فاصله زمانی و تطبیق مراجعه بالینی با اسکن است.

\textbf{فایل‌های بازتولیدپذیر:} همه جدول‌های تجمیعی سه تسک ویژه در
\Code{data/derived/phase1/c123/} قرار می‌گیرند و هیچ شناسه واقعیِ فرد در این
جدول‌ها صادر نمی‌شود. برای اجرای دوباره، اسکریپت
\Code{src/06_subset_structure_merge.py} همراه اجرای اصلی استفاده می‌شود.

"""

    # C.4 and C.5: read new generated aggregate files. No individual IDs enter report.
    source += r"""\section{گمشدگی و انتخاب نمونه چندوجهی}
\textbf{تعریف سه نوع وضعیت:} نخست «خانه واقعا خالی در CSV تحویلی»،
دوم «کد ویژه نامعتبر برای تحلیل همان متغیر» مطابق نگاشت متغیر-ویژه،
و سوم «مقدار قابل پرچم‌گذاری برای کنترل کیفیت» جداگانه شمارش شده‌اند.
این سه وضعیت نباید با هم جمع بسته و بدون توضیح «داده گمشده اولیه» نامیده شوند.
\subsection*{درصد گمشدگی در متغیرهای منتخب هر دو وجه}
"""
    source += table(['وجه / متغیر', 'خالی خام', 'درصد خام', 'کد ویژه', 'پرچم کیفیت', 'معتبر'], [
        (coded(x['variable']), x['raw_blank_n'], nf(x['raw_blank_pct'],1),
         x['documented_special_code_n'], x['quality_flag_n'], x['analysis_eligible_n'])
        for x in c4key
    ], r' p{5.0cm} r r r r r ')
    edu = next(x for x in c4key if x['variable'] == 'EDUC')
    moca = next(x for x in c4key if x['variable'] == 'MOCATOTS')
    source += rf"""
\textbf{{یافته‌های متغیر-ویژه:}} در \Code{{EDUC}} تعداد {edu['documented_special_code_n']}
رکورد با کد \En{{99}} و در \Code{{MOCATOTS}} تعداد {moca['documented_special_code_n']}
رکورد با کد \En{{88}} یا \En{{-4}} قابل استفاده به‌عنوان مقدار نمره نیستند.
برای \Code{{MOCATOTS}} تعداد {moca['out_of_plausible_range_or_nonpositive_n']}
مقدار خارج از دامنه نمره و {moca['noninteger_score_flag_n']} مقدار اعشاری
به‌عنوان پرچم کیفیت ثبت شده‌اند (این دو دسته می‌توانند همپوشانی داشته باشند).
با توجه به نام \En{{imputed}} فایل، این مقادیر خودکار اصلاح یا گرد نشده‌اند و
تا روشن‌شدن روش جایگذاری، در آمار نمره MoCA استفاده نمی‌شوند.
در مقابل چهار شاخص اصلی سوال، \Code{{CDRSUM}} و ضخامت انتورینال چپ و راست،
در فایل تحویلی برای تمام افراد مقدار عددی و در بازه کنترل‌شده دارند.

\subsection*{{پوشش MRI و عدم امکان مقایسه «دارای / فاقد MRI»}}
"""
    source += table(['معیار پوشش', 'تعداد'], [
        ('افراد UDS دارای رکورد MRI', c4['n_matched']),
        ('افراد UDS بدون رکورد MRI در فایل تحویلی', c4['n_uds_without_mri']),
        ('ستون ویژگی واقعی MRI بررسی‌شده', c4['mri_feature_columns']),
        ('خانه خالی در همه ویژگی‌های MRI', c4['mri_raw_blank_cells_total']),
        ('مقادیر غیرمثبت در ویژگی‌های MRI: پرچم کنترل کیفیت', c4['mri_nonpositive_cells_flagged']),
    ], r' p{11cm} r ')
    source += r"""
تمام افراد \En{UDS} در این \textbf{زیرمجموعه آماده} یک رکورد MRI دارند.
بنابراین شاخص «داشتن MRI» ثابت و مقایسه توصیفی افراد \textbf{دارای MRI با
فاقد MRI در این فایل}\textbf{ ناممکن} است؛ از داده‌های موجود نمی‌توان
درباره انتخاب‌شدن برای اسکن در کل پایگاه NACC نتیجه گرفت.
برای توصیف گروه موجود، فراوانی‌های کدهای شناختی UDS در افراد دارای MRI ثبت شدند:
"""
    source += table(['کد وضعیت شناختی', 'تعداد', 'درصد'], [
        (x['category'], x['n'], nf(x['pct_of_mri_available'], 1))
        for x in c4profile if x['variable']=='NACCUDSD'
    ], r' p{8.0cm} r r ')
    source += rf"""
در همان زیرنمونه دارای MRI، میانگین سن {nf(c4['age_mean_mri_group'], 2)} سال
و میانگین \Code{{CDRSUM}} برابر {nf(c4['cdrsum_mean_mri_group'], 3)} است.
این خلاصه‌ها یک \textbf{{گروه موجود}} را توصیف می‌کنند؛ جدول مقایسه دو گروه نیستند.
تعداد {c4['mri_nonpositive_cells_flagged']} مقدار غیرمثبت در کل ۱۲۴ ستون
MRI صرفا \textbf{{پرچم کنترل کیفیت}} است: بدون شرح پردازش تصویر، صفر را
به‌صورت خودکار «گمشده» یا خطای قطعی فرض نمی‌کنیم. این دو مورد به دو
شاخص ضخامت انتورینال منتخب مربوط نیستند. هیچ رکوردی به این دلیل حذف نشده است.

\section{{خلاصه مشترک شناختی و تصویربرداری}}
شرکت‌کنندگان با کلید \Code{{NACCID}} و فقط یک مشاهده تحویلی از هر وجه
به‌صورت یک‌به‌یک هم‌تراز شدند. در زیرمجموعه مشترک \textbf{{{c5['n_joint_valid_people']}}} نفر
دارای \Code{{CDRSUM}} معتبر از UDS و دو ضخامت انتورینال معتبر از MRI هستند؛ شاخص دوطرفه از میانگین همان دو مقدار ساخته شد؛
تعداد رکوردهای کنارگذاشته‌شده به‌علت نبود/نامعتبربودن این چهار شاخص برابر
\textbf{{{c5['n_excluded_from_joint_by_field_rules']}}} است.

\subsection*{{خلاصه توزیع چهار شاخص در افراد مشترک}}
"""
    names = {'CDRSUM': 'امتیاز بالینی CDRSUM',
             'LH_ENTORHINAL_AVGTH': 'ضخامت انتورینال چپ (mm)',
             'RH_ENTORHINAL_AVGTH': 'ضخامت انتورینال راست (mm)',
             'ENTORHINAL_MEAN_AVGTH': 'میانگین دوطرفه (mm؛ فرضی)'}
    source += table(['متغیر','میانگین','میانه','انحراف معیار','IQR','دامنه','پرت IQR'], [
        (names[x['variable']], nf(x['mean']), nf(x['median']), nf(x['sd_sample']),
         nf(x['iqr']), nf(x['min'])+' تا '+nf(x['max']), x['n_iqr_outliers'])
        for x in c5all
    ], r' p{3.5cm} r r r r p{2.3cm} r ')
    source += r"""
\textbf{روش:} انحراف معیار با مخرج \En{$n-1$}، چارک‌ها با درون‌یابی خطی
و نقاط پرت با حصارهای \En{$Q_1-1.5IQR$} و \En{$Q_3+1.5IQR$}
محاسبه شده‌اند. موارد بیرون از حصار صرفا توصیف شده و در میانگین و نمودارها
باقی مانده‌اند؛ به‌ویژه تراکم \Code{CDRSUM} روی صفر باعث می‌شود این قاعده
برای این امتیاز صرفا یک شاخص توصیفی، نه ملاک غیرعادی‌بودن بالینی، باشد.

\subsection*{خلاصه گروهی هم‌تراز بین UDS و MRI}
"""
    source += table(['گروه شناختی', 'تعداد', 'میانه CDRSUM', 'میانگین دوطرفه', 'انحراف معیار دوطرفه'], [
        (freq_names[str(code)],
         next(x['n_group'] for x in c5group if int(x['group_code'])==code and x['variable']=='CDRSUM'),
         nf(next(float(x['median']) for x in c5group if int(x['group_code'])==code and x['variable']=='CDRSUM')),
         nf(next(float(x['mean']) for x in c5group if int(x['group_code'])==code and x['variable']=='ENTORHINAL_MEAN_AVGTH')),
         nf(next(float(x['sd_sample']) for x in c5group if int(x['group_code'])==code and x['variable']=='ENTORHINAL_MEAN_AVGTH')))
        for code in (1,2,3,4)
    ], r' p{4.1cm} r r r r ')
    source += r"""
نمودار چهارم (شکل \ref{fig:joint-uds-mri}) جفت‌های نمره بالینی و ضخامت دوطرفه را
همراه جدول‌های این بخش نمایش می‌دهد.
"""
    source += rf"""
\section{{نهایی‌سازی سوال آماری و تناسب داده}}
\textbf{{سوال نهایی:}} آیا نمره \Code{{CDRSUM}} با میانگین ضخامت قشر انتورینال چپ و راست مرتبط است و این ارتباط با درنظرگرفتن سن و جنس چگونه تغییر می‌کند؟ متغیر پاسخ اصلی \Code{{ENTORHINAL_MEAN_AVGTH}} و توضیحی اصلی \Code{{CDRSUM}} است. سن و جنس فقط برای تحلیل‌های مراحل بعدی معرفی شده‌اند و در فاز اول هیچ مدل تعدیل‌شده‌ای برازش نشده است.

\textbf{{تناسب داده:}} {n} شرکت‌کننده یکتا در هر دو وجه \En{{UDS}} و \En{{MRI}} ثبت شده‌اند. طبق جدول‌های قابل بازتولید، هر {n} نفر در \Code{{CDRSUM}} و هر دو ستون ضخامت انتورینال مقدار معتبر دارند. میانگین ضخامت دوطرفه برای هر فرد از همین دو ستون ساخته شده است. میانگین نمونه‌ای آن \textbf{{{nf(next(float(x['mean']) for x in c5all if x['variable']=='ENTORHINAL_MEAN_AVGTH'))}}} و انحراف معیار نمونه‌ای \textbf{{{nf(next(float(x['sd_sample']) for x in c5all if x['variable']=='ENTORHINAL_MEAN_AVGTH'))}}} است (واحد میلی‌متر فرض شده است). نمودار چهارم نمایش توصیفی جفت‌های واقعی را فراهم می‌کند.

\textbf{{دامنه و محدودیت:}} این زیرمجموعه نماینده جامعه عمومی نیست؛ تاریخ اسکن و مراجعه، فرایند انتخاب نمونه و روش جایگذاری UDS در فایل‌ها مشخص نشده‌اند. رابطه مشاهده‌ای دلیل بر علیت یا هم‌زمانی نیست. مقایسه تعدیل‌شده بر حسب سن و جنس، آزمون فرض، فاصله اطمینان و رگرسیون موضوع فاز اول نیستند.

\section{{روش بازتولید و مراجع}}
"""
    source += r"""\textbf{داده خام:} سه CSV تحویلی در \Code{data/raw/} قرار دارند و اسکریپت‌ها
آن‌ها را تغییر نمی‌دهند. برای اجرای کامل بخش انجام‌شده، از ریشه مخزن:
\begin{latin}
\begin{verbatim}
python -m pip install -r requirements.txt
python src/run_phase1.py
\end{verbatim}
\end{latin}
نسخه قابل بازتولید گزارش در \Code{reports/phase1.tex} نوشته می‌شود.

فایل‌های ممیزی ریزدانه که ممکن است دارای شناسه \Code{NACCID} باشند،
فقط به‌صورت محلی در \Code{data/derived/} ذخیره می‌شوند.
این پوشه با \Code{.gitignore} از انتشار کنار گذاشته شده است.
برای نمایش چهار نمودارِ گزارش، به چهار فایل PNG در پوشه \Code{figures/} نیاز است.
جداول محاسباتی در پوشه \Code{data/derived/phase1/} ساخته می‌شوند.
گزارش با \En{XeLaTeX} قابل بازتولید است.

\textbf{مراجع اصلی:}
\begin{enumerate}
\item دستورالعمل رسمی تیم دستیاری، \Code{docs/amarpr.pdf}
\item \href{https://www.naccdata.org/about-nacc-data}{\En{NACC -- About NACC Data}}.
\item \href{https://www.naccdata.org/data-forms-and-documentation/}{\En{NACC -- Data Forms and Documentation}}.
\item \href{https://www.naccdata.org/the-nacc-researchers-guide}{\En{NACC -- The Researcher's Guide}}.
\end{enumerate}

\end{document}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = editorial_tex_style(source)
    OUT.write_text(source, encoding='utf-8')
    print(f'Created: {OUT} ({len(source):,} chars)')


if __name__ == '__main__':
    build()
