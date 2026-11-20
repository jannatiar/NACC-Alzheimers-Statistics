"""Build the Phase-2 Persian report in the same editorial style as Phase 1.

Run after the three Phase-2 analysis scripts:
    python src/13_build_phase2_tex.py
The script reads aggregate outputs only and writes reports/phase2.tex.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'derived' / 'phase2'
OUT = ROOT / 'reports' / 'phase2.tex'


def csvrows(relative: str) -> list[dict]:
    with (BASE / relative).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jsonread(relative: str) -> dict:
    return json.loads((BASE / relative).read_text(encoding='utf-8'))


def code(value: str) -> str:
    return r'\Code{' + str(value) + '}'


def row(*cells) -> str:
    return ' & '.join(str(x) for x in cells) + r' \\' + '\n'


def table(headers, rows, specs, *, small=True) -> str:
    text = ('\\small\n' if small else '')
    text += '\\begin{longtable}{' + specs + '}\n'
    text += '\\toprule\n' + row(*headers) + '\\midrule\n\\endfirsthead\n'
    text += '\\toprule\n' + row(*headers) + '\\midrule\n\\endhead\n'
    for item in rows:
        text += row(*item)
    text += '\\bottomrule\n\\end{longtable}\n\\normalsize\n'
    return text


def num(value) -> str:
    return r'\Num{' + str(value) + '}'


def nf(value, places=4) -> str:
    return num(f'{float(value):,.{places}f}')


def pct(value, places=2) -> str:
    return num(f'{100 * float(value):,.{places}f}\\%')


def pmf(value) -> str:
    number = float(value)
    formatted = f'{number:.6f}' if number >= 0.0001 else f'{number:.3e}'
    return num(formatted)


_HAMZA_MAP = {
    0x0623: 'ا', 0x0625: 'ا', 0x0624: 'و', 0x0626: 'ی', 0x0621: '',
    0x06C0: 'ه', 0x0671: 'ا', 0x0654: '', 0x064B: '', 0x064C: '', 0x064D: '',
}


def editorial_tex_style(tex: str) -> str:
    tex = tex.translate(_HAMZA_MAP)
    token = r'\textbf{'
    pos = 0
    while True:
        start = tex.find(token, pos)
        if start == -1:
            break
        opening = start + len(token) - 1
        depth = 1
        end = opening + 1
        while end < len(tex) and depth:
            if tex[end] == '{':
                depth += 1
            elif tex[end] == '}':
                depth -= 1
            end += 1
        if depth:
            raise ValueError('Unbalanced textbf braces')
        content = tex[opening + 1:end - 1]
        prefix = tex[tex.rfind('\n', 0, start) + 1:start].strip()
        keep = prefix in ('', r'\noindent') and content.strip().endswith((':', '：'))
        if keep:
            pos = end
        else:
            tex = tex[:start] + content + tex[end:]
            pos = start + len(content)
    return tex


def build() -> None:
    events = jsonread('events/definitions.json')
    joint = csvrows('events/joint_counts_probabilities.csv')
    conditional = csvrows('events/conditional_probabilities.csv')
    model = jsonread('models/model_summary.json')
    binomial = csvrows('models/binomial_pmf.csv')
    stage3 = jsonread('distributions/stage3_summary.json')
    cdr_pmf = csvrows('distributions/cdrsum_empirical_distribution.csv')
    covariance = csvrows('distributions/covariance_summary.csv')[0]
    data_structure = stage3['data_structure']
    uds_visit_fields = '، '.join(data_structure['visit_or_scan_date_fields_found']['UDS']) or 'هیچ‌کدام'
    mri_scan_fields = '، '.join(data_structure['visit_or_scan_date_fields_found']['MRI']) or 'هیچ‌کدام'

    x = stage3['variables']['X']
    y = stage3['variables']['Y']
    ind = model['independence']
    bayes = model['bayes']
    bino = model['binomial_model']
    n = int(events['n_valid_paired_participants'])
    n_a = int(events['event_A']['count'])
    n_b = int(events['event_B']['count'])
    n_equal = int(events['event_B']['n_equal_to_threshold'])
    threshold = float(events['event_B']['threshold'])

    source = r'''
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


\IfFontExistsTF{XB Niloofar}{
  \settextfont{XB Niloofar}
  \setdigitfont{XB Niloofar}
  \setlatintextfont{XB Niloofar}
}{
  \settextfont{Amiri}
  \setdigitfont{Amiri}
  \setlatintextfont{Amiri}
}
\setmonofont{DejaVu Sans Mono}

\newcommand{\Code}[1]{\lr{\texttt{\detokenize{#1}}}}
\newcommand{\En}[1]{\lr{#1}}
\newcommand{\Num}[1]{\ifmmode\text{\lr{#1}}\else\lr{#1}\fi}
\graphicspath{{reports/}{figures/}{../figures/}}
\setstretch{1.19}
\setlength{\parindent}{0pt}\setlength{\emergencystretch}{2em}
\setlength{\parskip}{0.35em}
\renewcommand{\arraystretch}{1.23}
\hypersetup{hidelinks,pdfauthor={NACC Dataset C -- Statistics Course},pdftitle={Phase 2 NACC UDS and MRI}}
\begin{document}

\thispagestyle{fancy}
\fancyhf{}
\lhead{

 \includegraphics[width=3cm]{reports/logo.png}
 \vspace{-2pt}
}
\chead{\leavevmode{\Huge به نام خدا}\vspace{15pt}\\  {\Large علیرضا جنتی}\vspace{10pt}\\{\small 403101672}\vspace{5pt}}
\rhead{

  {\large   پروژه درس آمار}

\vspace{10pt}   

{\large  فاز دوم}

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
'''

    source += r'''
\section{داده، متغیرها و تعریف رویدادها}
\subsection*{واحد مشاهده و کنترل مراجعات}

در فایل‌های تحویلی UDS و MRI هر فایل شامل ۱۲۳۷ ردیف
و ۱۲۳۷ شناسه یکتای \Code{NACCID} است. شناسه‌های دو فایل
کاملاً با یکدیگر تطابق دارند و اتصال آن‌ها با رابطه یک‌به‌یک
انجام شده است. بنابراین هر شرکت‌کننده در محاسبات این فاز
دقیقاً یک بار وارد شده است.

این قاعده صرفاً ساختار زیرمجموعه آماده‌شده درسی را توصیف
می‌کند و به معنی وجود تنها یک مراجعه برای هر فرد در پایگاه
اصلی NACC نیست. از آنجا که شماره و تاریخ مراجعات UDS و
تاریخ اسکن MRI در فایل‌های تحویلی موجود نیستند، تعیین اینکه
رکورد انتخاب‌شده مربوط به مراجعه اولیه، آخرین مراجعه یا
نزدیک‌ترین مراجعه به زمان MRI است، امکان‌پذیر نیست.
در نتیجه، در این فاز انتخاب مجدد مراجعه انجام نشده و
از همان یک رکورد موجود برای هر \Code{NACCID} استفاده شده است.
\small
\begin{longtable}{p{5.0cm} p{5.0cm} p{4.0cm}}
\toprule
متغیر & تعریف و منبع & نوع و نقش \\
\midrule
\endfirsthead
\toprule
متغیر & تعریف و منبع & نوع و نقش \\
\midrule
\endhead
\Code{CDRSUM} & نمره مجموع جعبه‌های CDR از UDS & کمی گسسته؛ متغیر شناختی X \\
\Code{ENTORHINAL_MEAN_AVGTH} & میانگین ضخامت چپ و راست از MRI & کمی پیوسته؛ متغیر MRI به‌نام Y \\
\Code{NACCUDSD} & وضعیت شناختی UDS کد ۴ برابر دمانس & طبقه‌ای؛ تعریف رویداد A \\
\Code{NACCID} & شناسه شرکت‌کننده & کلید اتصال؛ نه متغیر تحلیلی \\
\bottomrule
\end{longtable}
\normalsize

\textbf{رویداد :A} فرد در طبقه دمانس وضعیت شناختی UDS قرار داشته باشد؛ یعنی
\Code{NACCUDSD = 4}.

\textbf{رویداد :B} میانگین ضخامت انتورینال دوطرفه کمتر از میانه فاز اول، یعنی
'''
    source += rf'''{nf(threshold, 4)} باشد. مقدار مساوی آستانه در گروه \En{{NOT B}} قرار گرفته است.
در این داده {num(n_equal)} نفر دقیقا برابر آستانه هستند. 

\textbf{{تعداد افراد معتبر:}} {num(f'{n:,}')} نفر؛ برای هر نفر مقدار معتبر \Code{{CDRSUM}} و هر دو ضخامت
انتورینال موجود است. هیچ فردی برای تحلیل‌های این فاز حذف نشده است.


\section{{احتمال‌های حاشیه‌ای، مشترک و شرطی}}
احتمال تجربی هر رویداد از نسبت فراوانی آن به تعداد افراد معتبر محاسبه شده است:
\[
P_{{emp}}(E)=\frac{{n(E)}}{{N}},\qquad N={num(f'{n:,}')}.
\]
بنابراین درصدهای این بخش فقط فراوانی نسبی در زیرمجموعه درسی هستند و برآورد شیوع در جمعیت
عمومی محسوب نمی‌شوند.

\subsection*{{احتمال‌های حاشیه‌ای}}
'''
    marginal_rows = [
        (r'\(A\)', r'\En{' + f"{n_a}/{n}" + '}', pct(events['event_A']['empirical_probability'], 4)),
        (r'\(\mathrm{NOT}\ A\)', r'\En{' + f"{n - n_a}/{n}" + '}', pct((n - n_a) / n, 4)),
        (r'\(B\)', r'\En{' + f"{n_b}/{n}" + '}', pct(events['event_B']['empirical_probability'], 4)),
        (r'\(\mathrm{NOT}\ B\)', r'\En{' + f"{n - n_b}/{n}" + '}', pct((n - n_b) / n, 4)),
    ]
    source += table(['رویداد', 'فراوانی / کل', 'احتمال تجربی'], marginal_rows,
                    r' p{6cm} p{3cm} r ')
    source += r'''\subsection*{جدول توافق دو رویداد}
'''
    joint_rows = []
    for r in joint:
        label = r'\En{NOT A}' if r['UDS_event'] == 'NOT A' else (r'\En{A}' if r['UDS_event'] == 'A' else r'\En{Total}')
        joint_rows.append((label, num(int(float(r['n_B']))), num(int(float(r['n_not_B']))),
                           num(int(float(r['n_total']))), pct(float(r['P(event)']), 2)))
    source += table(['رویداد', r'\(B\)', r'\(\mathrm{NOT}\ B\)', 'مجموع', 'احتمال حاشیه‌ای'],
                    joint_rows, r' p{4.2cm} r r r r ')
    source += r''' چهار خانه جدول مشترک به‌صورت زیر هستند:
'''
    a_b = int(float(joint[0]['n_B']))
    a_not_b = int(float(joint[0]['n_not_B']))
    not_a_b = int(float(joint[1]['n_B']))
    not_a_not_b = int(float(joint[1]['n_not_B']))
    source += table(['رویداد مشترک', 'فراوانی', 'احتمال تجربی'], [
        (r'\(A\cap B\)', num(a_b), pct(a_b/n, 4)),
        (r'\(A\cap \mathrm{NOT}\ B\)', num(a_not_b), pct(a_not_b/n, 4)),
        (r'\(\mathrm{NOT}\ A\cap B\)', num(not_a_b), pct(not_a_b/n, 4)),
        (r'\(\mathrm{NOT}\ A\cap \mathrm{NOT}\ B\)', num(not_a_not_b), pct(not_a_not_b/n, 4)),
    ], r' p{6cm} r r ')
    source += r'''\enlargethispage{5pt}
\subsection*{احتمال‌های شرطی}
'''
    cond_rows = []
    conditional_labels = {
        'P(A | B)': r'\(P(A\mid B)\)',
        'P(A | NOT B)': r'\(P(A\mid \mathrm{NOT}\ B)\)',
    }
    for r in conditional:
        fraction = f"{int(float(r['numerator_n']))}/{int(float(r['denominator_n']))}"
        cond_rows.append((conditional_labels[r['probability']], r'\En{' + fraction + '}',
                          pct(float(r['value']), 4)))
    source += table(['احتمال شرطی', 'صورت / مخرج', 'درصد'], cond_rows, r' p{5cm} p{4cm} r ')
    source += rf'''
\textbf{{تفسیر:}} در گروه دارای ضخامت کمتر از میانه، احتمال تجربی وضعیت دمانس UDS برابر
{pct(float(conditional[0]['value']), 2)} است؛ در گروه دارای ضخامت مساوی یا بیشتر از میانه این مقدار
{pct(float(conditional[1]['value']), 2)} است. این دو احتمال شرطی نشان می‌دهند که محدودکردن محاسبه
به گروه MRI، فراوانی نسبی رویداد A را تغییر می‌دهد. این مقایسه توصیفی است و به‌تنهایی آزمون
معناداری یا رابطه علّی محسوب نمی‌شود.

\begin{{figure}}[H]
\centering
\includegraphics[width=.72\linewidth]{{phase2_01_conditional_cognitive_status.png}}
\caption{{احتمال تجربی وضعیت دمانس UDS در دو گروه تعریف‌شده با میانه MRI.}}
\end{{figure}}

\section{{استقلال رویدادها}}
برای استقلال A و B باید رابطه زیر برقرار باشد:
\[
P(A\cap B)=P(A)P(B).
\]
در داده حاضر، احتمال مشترک مشاهده‌شده برابر {pct(ind['observed_joint_probability'], 4)} و حاصل‌ضرب
احتمال‌های حاشیه‌ای برابر {pct(ind['product_of_marginals'], 4)} است. اختلاف این دو مقدار
{nf(100 * float(ind['difference_observed_minus_independence']), 4)} واحد درصد است.

\textbf{{نتیجه:}} فراوانی‌های تجربی این زیرمجموعه با استقلال A و B سازگار به نظر نمی‌رسند.

\section{{متغیر برنولی، مدل دوجمله‌ای و قضیه بیز}}
\subsection*{{متغیر برنولی و توزیع دوجمله‌ای}}
متغیر دودویی زیر برای رویداد A تعریف شد:
\[
Z=\begin{{cases}}1 & A\\0 & \mathrm{{NOT}}\ A.\end{{cases}}
\]
پارامتر تجربی آن برابر \(p=131/1237={nf(bino['p'], 6)}\) است. برای توضیح رفتار یک شمارش
دودویی، تعداد رخداد A در ۱۰ انتخاب فرضی و مستقل با K نشان داده شد:
\[
K\sim Binomial(n=10,p),\qquad P(K=k)=\binom{{10}}{{k}}p^k(1-p)^{{10-k}}.
\]
'''
    bin_rows = [(num(int(float(r['k']))), pmf(r['probability'])) for r in binomial]
    source += table(['تعداد رخداد A (k)', r'\(P(K=k)\)'], bin_rows, r' r r ')
    source += rf'''
جدول بالا تابع جرم احتمال شمار رخداد A را برای تمام مقادیر ممکن K در ده انتخاب فرضی نشان می‌دهد.

\textbf{{محدودیت مدل:}} ده انتخاب در اینجا یک مثال نظری است و به معنی نمونه‌گیری واقعی ده نفر
از NACC نیست. برای نمونه‌گیری بدون جایگذاری از یک مجموعه محدود، مدل دقیق فوق‌هندسی است؛
مدل دوجمله‌ای فقط در صورت ثابت‌بودن احتمال و استقلال انتخاب‌ها مناسب است. نسبت ۱۰ به ۱۲۳۷
حدود {pct(10/n, 2)} است، اما در فایل‌های تحویلی طرح نمونه‌گیری تصادفی برای این ده انتخاب ثبت نشده؛
پس مدل صرفا برای نمایش توزیع دوجمله‌ای به‌کار رفته است.
'''

    source += rf'''
\begin{{figure}}[H]
\centering
\includegraphics[width=.82\linewidth]{{phase2_02_binomial_pmf.png}}
\caption{{تابع جرم احتمال شمار رخداد A در ده انتخاب فرضی مستقل.}}
\end{{figure}}

\subsection*{{قضیه بیز}}
برای دو رویداد تعریف‌شده، به‌روزرسانی احتمال با قضیه بیز به شکل زیر است:
\[
P(A|B)=\frac{{P(B|A)P(A)}}{{P(B)}}.
\]
با جایگذاری فراوانی‌های تجربی، مقدار بیزی برابر {pct(bayes['P_A_given_B_bayes'], 4)} به دست آمد؛
احتمال شرطی مستقیم مرحله اول نیز {pct(bayes['P_A_given_B_direct'], 4)} است.

\section{{توزیع، امید ریاضی، واریانس و نقاط دورافتاده}}
برای هر دو متغیر، توزیع تجربی، میانگین و واریانس محاسبه شد. چارک‌ها با درون‌یابی خطی محاسبه شده‌اند.

'''
    dist_rows = [
        ('تعداد معتبر', num(int(float(x['n']))), num(int(float(y['n'])))),
        ('میانگین', nf(x['mean'], 4), nf(y['mean'], 4)),
        ('واریانس تجربی با مخرج N', nf(x['variance_empirical_ddof0'], 4), nf(y['variance_empirical_ddof0'], 4)),
        ('انحراف معیار نمونه‌ای با مخرج N-1', nf(x['sd_sample_ddof1'], 4), nf(y['sd_sample_ddof1'], 4)),
        ('میانه', nf(x['median'], 4), nf(y['median'], 4)),
        ('چارک اول Q1', nf(x['q1'], 4), nf(y['q1'], 4)),
        ('چارک سوم Q3', nf(x['q3'], 4), nf(y['q3'], 4)),
        ('دامنه', f"{nf(x['min'], 3)} تا {nf(x['max'], 3)}", f"{nf(y['min'], 3)} تا {nf(y['max'], 3)}"),
        ('تعداد نقاط خارج از حصار IQR', num(int(float(x['n_iqr_outliers']))), num(int(float(y['n_iqr_outliers'])))),
    ]
    source += table(['شاخص توصیفی', r'\Code{CDRSUM}', 'میانگین ضخامت MRI'],
                    dist_rows, r' p{7.2cm} r r ')
    source += rf'''
میانگین \Code{{CDRSUM}}، متوسط نمره در همین زیرمجموعه را نشان می‌دهد؛ واریانس {nf(x['variance_empirical_ddof0'], 4)}
با واحد امتیاز به توان دو، پراکندگی نمره‌ها را حول این میانگین خلاصه می‌کند. برای متغیر MRI،
میانگین ضخامت {nf(y['mean'], 4)} میلی‌متر فرض‌شده و واریانس {nf(y['variance_empirical_ddof0'], 4)}
میلی‌متر به توان دو است؛ انحراف معیار نمونه‌ای {nf(y['sd_sample_ddof1'], 4)} میلی‌متر، مقیاس پراکندگی را
در واحد اصلی متغیر نشان می‌دهد.

\textbf{{نمره CDRSUM}}: مقدار صفر برای {num(757)} نفر، یعنی {pct(757/n, 2)} از زیرمجموعه، ثبت شده است.
توزیع به‌شدت راست‌چوله است. با توجه به \(Q_1=0\)، \(Q_3=1\) و \(IQR=1\)،
حصارهای IQR از \Num{-1.5} تا \Num{2.5} هستند؛ {num(115)} مقدار بالاتر از حصار بالا قرار می‌گیرند.
برای این متغیر گسسته، تقریب نرمال توصیف مناسبی از توزیع کامل نیست.

\textbf{{متغیر MRI}}: میانگین ضخامت دوطرفه {nf(y['mean'], 4)} و میانه آن {nf(y['median'], 4)} است.
میانگین کمتر از میانه است و چولگی تجربی برابر {nf(y['skewness_empirical'], 3)} است. این شاخص
توصیفی با عدم تقارن به سمت مقادیر پایین سازگار است؛ نمودار Q-Q نیز انحراف دنباله پایین را نشان می‌دهد.

\textbf{{قاعده نقاط دورافتاده}}: نقاط خارج از حصارهای \(Q_1-1.5IQR\) و \(Q_3+1.5IQR\)
پرچم‌گذاری شده‌اند. برای \Code{{CDRSUM}} حصارها از {nf(x['lower_fence'], 3)} تا
{nf(x['upper_fence'], 3)} هستند؛ برای MRI از {nf(y['lower_fence'], 3)} تا
{nf(y['upper_fence'], 3)}. تعداد نقاط پرچم‌خورده به‌ترتیب {num(int(float(x['n_iqr_outliers'])))} و
{num(int(float(y['n_iqr_outliers'])))} است. این پرچم‌ها به‌معنی خطای قطعی داده نیستند؛ هیچ فرد یا
مقداری به دلیل دورافتاده‌بودن حذف نشده است.

\subsection*{{توزیع تجربی کامل CDRSUM}}
'''
    pmf_rows = [(nf(r['CDRSUM'], 1), num(int(float(r['count']))),
                 pct(float(r['probability']), 2)) for r in cdr_pmf]
    source += table(['نمره CDRSUM', 'تعداد', 'درصد'], pmf_rows,
                    r' r r r ')
    source += r'''
\begin{figure}[H]
\centering
\includegraphics[width=.82\linewidth]{phase2_03_cdrsum_distribution.png}
\caption{توزیع تجربی گسسته نمره CDRSUM.}
\end{figure}

\subsection*{بررسی تقریب نرمال برای متغیر پیوسته MRI}
برای متغیر پیوسته ضخامت انتورینال، هیستوگرام همراه منحنی نرمال مرجع و نمودار کوانتایل--کوانتایل
بررسی شد. این دو نمودار برای ارزیابی بصری شکل توزیع استفاده شده‌اند؛ آزمون رسمی نرمال‌بودن
اجرا نشده است.

\begin{figure}[H]
\centering
\includegraphics[width=.82\linewidth]{phase2_04_mri_thickness_distribution.png}
\caption{هیستوگرام ضخامت انتورینال دوطرفه و منحنی نرمال مرجع با میانگین و انحراف معیار تجربی.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=.75\linewidth]{phase2_05_mri_normal_qq.png}
\caption{نمودار Q-Q برای بررسی انحراف توزیع ضخامت MRI از توزیع نرمال.}
\end{figure}

بر اساس نمودارها، بخش مرکزی توزیع تا حدی به خط مرجع نزدیک است، اما دنباله پایین انحراف
مشخصی دارد و \Num{22} مقدار پایین‌تر از حصار IQR قرار گرفته‌اند. بنابراین منحنی نرمال، توصیف کاملی
از کل توزیع نیست و باید با احتیاط به‌عنوان تقریب استفاده شود.

\section{کوواریانس شناخت و تصویربرداری}
برای دو متغیر کمی اصلی، \(X=CDRSUM\) و \(Y=ENTORHINAL\_MEAN\_AVGTH\)، کوواریانس تجربی
با مخرج \(N\) محاسبه شد:
\[
Cov_{emp}(X,Y)=\frac{1}{N}\sum_{i=1}^{N}(X_i-\bar X)(Y_i-\bar Y).
\]
'''
    cov_value = float(covariance['covariance_empirical_ddof0'])
    source += rf'''
مقدار به‌دست‌آمده برابر است با:
\[
Cov_{{emp}}(X,Y)={nf(cov_value, 6)}.
\]
علامت منفی نشان می‌دهد در این زیرمجموعه، مقادیر بالاتر از میانگین \Code{{CDRSUM}} تمایل دارند
همراه با مقادیر پایین‌تر از میانگین ضخامت انتورینال دیده شوند. واحد کوواریانس «امتیاز CDR
ضرب‌در میلی‌متر فرضی» است و مقدار آن به مقیاس دو متغیر وابسته است. در این فاز ضریب همبستگی
یا رگرسیون محاسبه نشده و از کوواریانس نتیجه علّی گرفته نمی‌شود.

\begin{{figure}}[H]
\centering
\includegraphics[width=.82\linewidth]{{phase2_06_cdrsum_mri_covariance.png}}
\caption{{نمودار پراکنش جفت‌های واقعی CDRSUM و ضخامت دوطرفه؛ جابه‌جایی افقی صرفا برای نمایش نقاط هم‌پوشان است.}}
\end{{figure}}
'''

    source += r'''

\section{بحث، محدودیت‌ها و جمع‌بندی}
در زیرمجموعه ۱۲۳۷ نفری، احتمال تجربی وضعیت دمانس UDS در گروه MRI زیر میانه بیشتر از گروه
مساوی یا بالاتر از میانه است. مقایسه احتمال‌های مشترک و حاصل‌ضرب احتمال‌های حاشیه‌ای نیز
با استقلال رویدادهای تعریف‌شده سازگار نیست. مدل دوجمله‌ای و قضیه بیز، روابط احتمالی را با
همان فراوانی‌های تجربی روشن می‌کنند.

توزیع \Code{CDRSUM} دارای تراکم زیاد در صفر و چولگی راست است. توزیع ضخامت انتورینال دوطرفه
در بخش مرکزی تا حدی به منحنی نرمال نزدیک است، اما دنباله پایین و نقاط دورافتاده نشان می‌دهند
تقریب نرمال برای کل توزیع کامل نیست. کوواریانس منفی نیز جهت رابطه توصیفی بین شدت بالینی و
ضخامت انتورینال را نشان می‌دهد؛ اما اندازه آن به مقیاس متغیرها وابسته است.

\textbf{محدودیت‌ها:} این زیرمجموعه نمونه احتمالی از جمعیت عمومی نیست؛ فایل UDS از پیش
\En{imputed} شده و روش جایگذاری معلوم نیست؛ تاریخ MRI و ارزیابی UDS در دسترس نیست؛ واحد
میلی‌متر برای ضخامت فرض شده؛ وضعیت دمانس UDS علت آلزایمری را مشخص نمی‌کند. نقاط دورافتاده
حذف نشده‌اند و هیچ آزمون استنباطی یا نتیجه علّی ارائه نشده است.

\textbf{جمع‌بندی:} فاز دوم با تعریف ثابت رویدادها، محاسبه احتمال‌های تجربی و شرطی، بررسی
توصیفی استقلال، مدل دوجمله‌ای و قضیه بیز، توصیف توزیع دو متغیر اصلی و محاسبه کوواریانس
به پایان رسیده است. تحلیل‌های استنباطی و مدل‌سازی به فازهای بعدی موکول می‌شوند.

\subsection*{ردیابی تسک‌های رسمی فاز دوم}
جدول زیر محل پوشش هر ۹ تسک مشترک و ۵ الزام اختصاصی مسیر ج را در گزارش مشخص می‌کند.
\small
\begin{longtable}{p{2.0cm} p{9.0cm} r}
\toprule
تسک & موضوع & بخش گزارش \\
\midrule
\endfirsthead
\toprule
تسک & موضوع & بخش گزارش \\
\midrule
\endhead

مشترک ۱ & تعریف رویدادها و متغیرهای تصادفی & ۱ و ۴ \\
مشترک ۲ & احتمال‌های حاشیه‌ای و مشترک & ۲ \\
مشترک ۳ & احتمال شرطی & ۲ \\
مشترک ۴ & بررسی استقلال & ۳ \\
مشترک ۵ & توزیع گسسته و مدل دوجمله‌ای & ۴ \\
مشترک ۶ & متغیر پیوسته و تقریب نرمال & ۵ \\
مشترک ۷ & امید ریاضی و واریانس & ۵ \\
مشترک ۸ & دو متغیر تصادفی و کوواریانس & ۶ \\
مشترک ۹ & قضیه بیز & ۴ \\
ج.۱ & احتمال شرطی وضعیت شناختی & ۲ \\
ج.۲ & توزیع نمره شناختی و متغیر MRI & ۵ \\
ج.۳ & احتمال مشترک دو وجه & ۲ \\
ج.۴ & کوواریانس شناخت و تصویربرداری & ۶ \\
ج.۵ & کنترل مراجعات و یک مشاهده برای هر فرد & ۱ \\

\bottomrule
\end{longtable}

\end{document}
'''
    source = source.replace('__UDS_ROWS__', f"{data_structure['uds_rows']:,}")
    source = source.replace('__MRI_ROWS__', f"{data_structure['mri_rows']:,}")
    source = source.replace('__PAIRED_ROWS__', f"{data_structure['paired_participants']:,}")
    source = source.replace('__UDS_VISIT_FIELDS__', uds_visit_fields)
    source = source.replace('__MRI_SCAN_FIELDS__', mri_scan_fields)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    source = editorial_tex_style(source)
    OUT.write_text(source, encoding='utf-8')
    print(f'Created: {OUT} ({len(source):,} chars)')


if __name__ == '__main__':
    build()
