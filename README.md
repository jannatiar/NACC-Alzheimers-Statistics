# پروژه درس آمار — فاز اول، مسیر ج (NACC UDS + MRI)

## داده‌های موردنیاز و اجرا
سه CSV اصلی تیم درس را بدون هیچ تغییری و فقط روی سیستم شخصی در `data/raw/` کپی کن:

- `udslong_imputed.csv`
- `mriscanclean.csv`
- `combined_adjacency_matrix.csv`

این فایل‌ها و خروجی‌های ریزدانه دارای شناسه `NACCID` نباید در GitHub یا ZIP تحویل قرار بگیرند. شرح محدودیت‌های استفاده در `data/README.md` آمده است.

از ریشه پروژه در PowerShell (محیط مجازی فعال باشد) اجرا کن:

```powershell
python -m pip install -r requirements.txt
python src/run_phase1.py
```
## ترتیب اسکریپت‌های Python

| شماره | فایل | وظیفه |
|---|---|---|
| ۰۱ | `src/01_inventory.py` | منشا، ساختار، شناسه‌ها و ماتریس مجاورت |
| ۰۲ | `src/02_quality_audit.py` | ممیزی گمشدگی، کدهای ویژه و ناسازگاری |
| ۰۳ | `src/03_descriptive_tables.py` | فراوانی و خلاصه تجربی؛ میانگین دوطرفه |
| ۰۴ | `src/04_center_spread.py` | میانگین، میانه، انحراف معیار، چارک و IQR |
| ۰۵ | `src/05_visualize_and_compare.py` | سه نمودار و مقایسه توصیفی گروه‌ها |
| ۰۶ | `src/06_subset_structure_merge.py` | ج.۱ تا ج.۳: نقشه UDS/MRI، مراجعات و اتصال |
| ۰۷ | `src/07_c4_missingness_selection.py` | ج.۴: پوشش MRI و الگوی گمشدگی |
| ۰۸ | `src/08_c5_joint_uds_mri.py` | ج.۵: خلاصه مشترک و نمودار چهارم |
| ۰۹ | `src/09_build_phase1_tex.py` | ساخت گزارش LaTeX از خروجی‌های تاییدشده |

`src/run_phase1.py` این ۹ فایل را به‌ترتیب اجرا می‌کند و در صورت امکان PDF را می‌سازد. ویرایش دایمی متن گزارش باید در `src/09_build_phase1_tex.py` انجام شود، نه در فایل تولیدشده `reports/phase1.tex`.

## فایل‌های تحویلی
`PROJECT_CARD.md`، `README.md`، `reports/phase1.pdf`، `reports/phase1.tex`، چهار تصویر `figures/`، کدهای `src/`، فایل `requirements.txt`، نقشه‌های بدون رکورد فردی `docs/uds_variable_map.csv` و `docs/mri_variable_map.csv`، و راهنمای داده در `data/README.md`. برای تحویل در CW همین‌ها در ZIP بدون CSV خام بسته‌بندی شده‌اند.

## محدودیت‌های تفسیری
- تطبیق بر پایه `NACCID` صورت می‌گیرد؛ تاریخ MRI و معاینه بالینی و شمار مراجعات اصلی معلوم نیستند.
- UDS از پیش `imputed` است؛ درصد گمشدگی اولیه قابل بازیابی نیست. در زیرمجموعه تحویلی گروه فاقد MRI وجود ندارد.
- واحد ضخامت (میلی‌متر) فرض متعارف است و نیازمند تایید مستندات پردازش MRI است.
- گروه‌بندی `NACCUDSD` تشخیص اختصاصی علت آلزایمری نیست و نمونه قابل تعمیم به جمعیت عمومی نیست.