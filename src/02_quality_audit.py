"""Phase 1, common task 5 - NACC UDS/MRI data-quality AUDIT (not cleaning).

Usage (from project root): python src/02_quality_audit.py
Inputs: data/raw/{udslong_imputed,mriscanclean,combined_adjacency_matrix}.csv
Output: data/derived/phase1/quality_audit/
Does NOT modify data/raw or automatically impute, round, winsorize or remove rows.

Sources: NACC UDS RDD v3, https://files.alz.washington.edu/documentation/uds3-rdd.pdf
NACC Researcher's Guide: https://www.naccdata.org/the-nacc-researchers-guide
Only field-specific documented special codes are marked CONFIRMED missing.
Other common number patterns are CANDIDATES and require field-by-field RDD review.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'derived' / 'phase1' / 'quality_audit'
OUT.mkdir(parents=True, exist_ok=True)

# Each entry is derived from the RDD, NOT a generic rule for the whole UDS.
# If an additional UDS variable is used in analysis, check the appropriate RDD
# definition and extend this map first.
DOCUMENTED_SPECIAL = {
    'EDUC': {99: 'unknown education'},
    'NACCBMI': {888.8: 'unknown/not assessed', -4: 'not applicable'},
    'HEIGHT': {88.8: 'unknown/not assessed', -4: 'not applicable'},
    'WEIGHT': {888: 'unknown/not assessed', -4: 'not applicable'},
    'MOCATOTS': {88: 'not administered', -4: 'not applicable'},
    'NACCGDS': {88: 'not calculable', -4: 'not applicable'},
    'DIABETES': {9: 'unknown', -4: 'not applicable'},
    'HYPERTEN': {9: 'unknown', -4: 'not applicable'},
    # INDEPEND=9 denotes 'other or unknown': retain as a separate category.
    'BILLS': {8: 'not applicable', 9: 'unknown', -4: 'not applicable'},
    'SHOPPING': {8: 'not applicable', 9: 'unknown', -4: 'not applicable'},
}
# These code sets are confirmed from the relevant UDS RDD. A fractional
# value outside the set is a code error, not a value to round automatically.
CATEGORICAL_CODES = {
    'NACCUDSD': {1, 2, 3, 4},
    'SEX': {1, 2},
    'DIABETES': {0, 1, 2, 9, -4},
    'HYPERTEN': {0, 1, 2, 9, -4},
    'INDEPEND': {1, 2, 3, 4, 9, -4},
    'BILLS': {0, 1, 2, 3, 8, 9, -4},
    'SHOPPING': {0, 1, 2, 3, 8, 9, -4},
}
# Explicit allowable ranges from the UDS RDD. Scores are integers except
# CDRSUM, whose domain is in half steps.
BOUNDS = {
    'EDUC': (0, 36),
    'NACCBMI': (10, 100),
    'MOCATOTS': (0, 30),
    'NACCGDS': (0, 15),
    'CDRSUM': (0, 18),
    'HEIGHT': (36, 87.9),
    'WEIGHT': (50, 400),
}
INTEGERS_ONLY = {'MOCATOTS', 'NACCGDS', 'EDUC'}
SPECIAL_CANDIDATES = {-4, 8, 9, 88, 88.8, 99, 888, 888.8, 995, 996, 997, 998, 999, 9999}
CDR_COMPONENTS = ('MEMORY', 'ORIENT', 'JUDGMENT', 'COMMUN', 'HOMEHOBB', 'PERSCARE')


def load(filename):
    path = RAW / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}\nPut original files in data/raw/ and run from project root.')
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def norm_id(value):
    return str(value or '').strip()


def dump_csv(name, columns, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def pretty_counts(counter):
    return '; '.join(f'{x:g}:{n}' for x,n in sorted(counter.items())) or '-'


def main():
    uf, uds = load('udslong_imputed.csv')
    mf, mri = load('mriscanclean.csv')
    af, adj = load('combined_adjacency_matrix.csv')
    flags = []
    columns = []
    counts = Counter()

    def flag(dataset, row, field, value, issue, action, reason=''):
        counts[issue] += 1
        flags.append(dict(dataset=dataset, NACCID=row.get('NACCID',''), column=field,
                          value=value, issue=issue, decision=action, note=reason))

    for dataset, fields, rows in [('UDS', uf, uds), ('MRI', mf, mri)]:
        for field in fields:
            if field == 'NACCID':
                missing = sum(not norm_id(row[field]) for row in rows)
                columns.append(dict(dataset=dataset, variable=field, raw_blank_count=missing,
                                    documented_nonanalytic_code_count=0, candidate_special_counts='-',
                                    fractional_count=0, min_value='', max_value='',
                                    iqr_outlier_count='', comment='identifier, not numeric'))
                continue
            vals = np.array([number(row[field]) for row in rows], dtype=float)
            rawblank = sum(str(row[field]).strip().lower() in ('','na','n/a','nan','none','null') for row in rows)
            nonnumeric = ~np.isfinite(vals)
            if np.any(nonnumeric & ~np.array([not str(row[field]).strip() for row in rows])):
                for row,bad in zip(rows,nonnumeric):
                    if bad: flag(dataset,row,field,row[field],'non_numeric', 'investigate; do not convert silently')
            confirmed = DOCUMENTED_SPECIAL.get(field,{}) if dataset == 'UDS' else {}
            special_mask = np.array([v in confirmed for v in vals])
            for row,v,match in zip(rows,vals,special_mask):
                if match:
                    flag(dataset,row,field,row[field],'documented_nonanalytic_code',
                         ('keep as explicit N/A/unknown category for frequencies; omit from ordinal scoring' if v in (8,9) and field in ('BILLS','SHOPPING') else 'treat as missing or not-applicable for numeric summaries'),confirmed[v])
            candidates=Counter(float(v) for v in vals if np.isfinite(v) and v in SPECIAL_CANDIDATES)
            if dataset=='UDS' and field in BOUNDS:
                lo,hi=BOUNDS[field]
                for row,v,masked in zip(rows,vals,special_mask):
                    if np.isfinite(v) and not masked and not (lo<=v<=hi):
                        flag(dataset,row,field,row[field],'out_of_documented_range',
                             'exclude from numeric summary; inspect source')
            if dataset=='UDS' and field in INTEGERS_ONLY:
                for row,v,masked in zip(rows,vals,special_mask):
                    if np.isfinite(v) and not masked and abs(v-round(v))>1e-8:
                        flag(dataset,row,field,row[field],'noninteger_score',
                             'exclude from raw-score summary; do not round', 'integer-valued measure')
            if dataset=='UDS' and field in CATEGORICAL_CODES:
                allowed=CATEGORICAL_CODES[field]
                for row,v in zip(rows,vals):
                    if np.isfinite(v) and v not in allowed:
                        flag(dataset,row,field,row[field],'invalid_category',
                             'exclude from categorical frequencies; no automatic rounding')
            valid=vals[np.isfinite(vals) & ~special_mask]
            fractional=int(np.sum(np.isfinite(vals) & (np.abs(vals-np.round(vals))>1e-8)))
            outliers=''
            if dataset=='MRI' and field.endswith(('_AVGTH','_GVOL')):
                # Clinical plausibility of ROI measures needs MRI processing/QC metadata.
                # IQR is ONLY a flag for review, never an automatic exclusion rule.
                q1,q3=np.percentile(valid,[25,75]); iqr=q3-q1
                low,high=q1-1.5*iqr,q3+1.5*iqr
                outmask=(vals<low)|(vals>high)
                outliers=int(np.sum(outmask))
                if field.endswith('_GVOL'):
                    for row,v in zip(rows,vals):
                        if np.isfinite(v) and v<=0:
                            flag(dataset,row,field,row[field],'nonpositive_mri_volume',
                                 'MRI QC required; do not auto-replace with NA')
                else:
                    for row,v in zip(rows,vals):
                        if np.isfinite(v) and v<=0:
                            flag(dataset,row,field,row[field],'nonpositive_mri_thickness',
                                 'MRI QC required; do not auto-replace with NA')
                # For the two actual MRI outcomes, record individual IQR flags.
                if field in ('LH_ENTORHINAL_AVGTH','RH_ENTORHINAL_AVGTH'):
                    for row,v,b in zip(rows,vals,outmask):
                        if b:flag(dataset,row,field,row[field],'entorhinal_iqr_outlier',
                                  'inspect only; retain unless QC evidence supports exclusion')
            columns.append(dict(dataset=dataset,variable=field,raw_blank_count=rawblank,
                         documented_nonanalytic_code_count=int(special_mask.sum()),
                         candidate_special_counts=pretty_counts(candidates),
                         fractional_count=fractional,
                         min_value=f'{np.min(valid):g}' if valid.size else '',
                         max_value=f'{np.max(valid):g}' if valid.size else '',
                         iqr_outlier_count=outliers, comment=('Field-specific RDD codes needed' if dataset=='UDS' and field not in DOCUMENTED_SPECIAL else '')))

    def duplicates(dataset,rows):
        ids=[norm_id(row.get('NACCID')) for row in rows]
        dupl=sum(c-1 for c in Counter(ids).values() if c>1)
        full_dupl=len(rows)-len({tuple(row.values()) for row in rows})
        if dupl:
            for row in rows:
                if ids.count(norm_id(row['NACCID']))>1:
                    flag(dataset,row,'NACCID',row['NACCID'],'duplicate_id','resolve one-row-per-participant rule')
        return {'rows':len(rows),'unique_nonempty_id':len(set(ids)-{''}),
                'missing_id':sum(not x for x in ids),
                'extra_duplicate_id_rows':dupl,'duplicate_full_rows':full_dupl}

    ids_u=duplicates('UDS',uds);ids_m=duplicates('MRI',mri)
    u_map={norm_id(row['NACCID']):row for row in uds}
    m_map={norm_id(row['NACCID']):row for row in mri}
    shared=set(u_map)&set(m_map)
    category_disagreement=0
    for pid in shared:
        if number(u_map[pid]['NACCUDSD'])!=number(m_map[pid]['NACCUDSD']):
            category_disagreement+=1
            flag('UDS/MRI',u_map[pid],'NACCUDSD',u_map[pid]['NACCUDSD'],
                 'diagnosis_mismatch','investigate source mapping')

    cdr_fail,bmi_fail,bp_fail=0,0,0
    for row in uds:
        cdr=number(row['CDRSUM'])
        total=sum(number(row[name]) for name in CDR_COMPONENTS)
        if not np.isclose(cdr,total,atol=1e-9):
            cdr_fail+=1
            flag('UDS',row,'CDRSUM',str(cdr),'cdr_sum_inconsistent',
                 'investigate item-score inconsistency', f'sum of 6 boxes = {total}')
        h,w,b=[number(row[k]) for k in ('HEIGHT','WEIGHT','NACCBMI')]
        # Only check BMI equation where all 3 values are within documented ranges.
        if 36<=h<=87.9 and 50<=w<=400 and 10<=b<=100:
            computed=703*w/(h*h)
            if abs(b-computed)>0.4: # allows normal 1-decimal BMI rounding
                bmi_fail+=1
                flag('UDS',row,'NACCBMI',str(b),'bmi_formula_inconsistent',
                     'check derived data and source before using',
                     f'calculated={computed:.2f}; reported={b:.2f}')
        syst,diast=number(row['BPSYS']),number(row['BPDIAS'])
        if 70<=syst<=230 and 40<=diast<=150 and syst<=diast:
            bp_fail+=1
            flag('UDS',row,'BPSYS/BPDIAS',f'{syst:g}/{diast:g}',
                 'blood_pressure_inconsistent', 'inspect original reading')

    adjlabels=[str(x[af[0]]) for x in adj]
    headerlabels=af[1:]
    adjmatrix=np.array([[number(row[k]) for k in headerlabels] for row in adj],dtype=float)
    adjstats=dict(shape=[len(adj),len(headerlabels)],square=len(adj)==len(headerlabels),
                  row_col_labels_identical=adjlabels==headerlabels,
                  all_binary=bool(np.isin(adjmatrix,[0.,1.]).all()),
                  symmetric=bool(adjmatrix.shape[0]==adjmatrix.shape[1] and np.allclose(adjmatrix,adjmatrix.T)),
                  diagonal_unique=sorted(set(np.diag(adjmatrix))) if adjmatrix.shape[0]==adjmatrix.shape[1] else [])

    summary=dict(UDS=ids_u,MRI=ids_m,overlap_naccid=len(shared),
                 uds_only=len(set(u_map)-set(m_map)),mri_only=len(set(m_map)-set(u_map)),
                 NACCUDSD_mismatch=category_disagreement,CDRSUM_sum_mismatch=cdr_fail,
                 BMI_formula_mismatch_within_valid_ranges=bmi_fail,
                 blood_pressure_logic_failure=bp_fail,
                 adjacency=adjstats,issues=dict(counts),
                 important_notes=[
                     'Zero empty cells does not imply zero missing values.',
                     'Special codes require per-variable RDD checks; 8 and 9 can be legitimate response categories.',
                     'Column-quality candidate counts are screening signals, NOT automatically missing.',
                     'The imputation mechanism and UDS/MRI visit pairing are not documented in supplied files.',
                     'MRI IQR outliers are not automatically invalid or deleted.',
                     'FORMVER is absent; UDSv3 codebook checks are provisional for this course subset.',
                     'All original CSVs are unchanged.'])
    dump_csv('column_quality_overview.csv',
             ['dataset','variable','raw_blank_count','documented_nonanalytic_code_count',
              'candidate_special_counts','fractional_count','min_value','max_value',
              'iqr_outlier_count','comment'],columns)
    dump_csv('flagged_cases.csv',
             ['dataset','NACCID','column','value','issue','decision','note'],flags)
    (OUT/'quality_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nOutput directory:',OUT)

if __name__=='__main__':main()
