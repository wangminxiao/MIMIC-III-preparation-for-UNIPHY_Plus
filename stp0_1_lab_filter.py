import polars as pl

"""
"ROW_ID","ITEMID","LABEL","FLUID","CATEGORY","LOINC_CODE"
172,50971,"Potassium","Blood","Chemistry","2823-3"
94,50893,"Calcium, Total","Blood","Chemistry","2000-8"
184,50983,"Sodium","Blood","Chemistry","2951-2"
132,50931,"Glucose","Blood","Chemistry","2345-7"
14,50813,"Lactate","Blood","Blood Gas","32693-4"
113,50912,"Creatinine","Blood","Chemistry","2160-0"
"""

cmp_ids = [
    # ID,    Lab name,  LOINC_CODE
	50971, # Potassium, "2823-3"
	50893, # Calcium,   "2000-8" / in emory data the code is 
	50983, # Sodium,    "2951-2"
	50931, # Glucose,   "2345-7"
	50813, # Lactate,   "32693-4" / in emory data we may choose "2518-9"
	50912, # Creatinine "2160-0"
	# 51000, # Lipids: Triglycerides, "1644-4" / not exist in emory data
    # 50906, # Lipids: Cholesterol LDL, "18262-6" / not exist in emory data
]

lf = (
    pl.scan_csv("/labs/hulab/MIMICIII-v1.4/LABEVENTS.csv", infer_schema_length=None)
    .select([
        "SUBJECT_ID",
        "HADM_ID",
        pl.col("CHARTTIME").str.to_datetime(format="%Y-%m-%d %H:%M:%S"),
        "ITEMID",
        "VALUENUM"
    ])
    .filter(pl.col("ITEMID").is_in(cmp_ids))
)

# Group by SUBJECT_ID, HADM_ID, CHARTTIME
agg_exprs = [
    (
        pl.when(pl.col("ITEMID") == iid)
          .then(pl.col("VALUENUM"))
          .otherwise(None)
          .max()
          .alias(str(iid))
    )
    for iid in cmp_ids
]

lf_wide = lf.group_by(["SUBJECT_ID", "HADM_ID", "CHARTTIME"]).agg(agg_exprs)

df_wide = lf_wide.collect()

# Rename columns for readability
df_final = df_wide.rename({
	"50971": "Potassium",
	"50893": "Calcium",
	"50983": "Sodium",
	"50931": "Glucose",
	"50813": "Lactate",
	"50912": "Creatinine",
	# "51000": "Triglycerides",
    # "50906": "Cholesterol_LDL",
})

df_final.write_parquet("cmp_panel_wide.parquet")
