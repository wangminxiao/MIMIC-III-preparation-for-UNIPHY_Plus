import polars as pl
"""
"ROW_ID","ITEMID","LABEL","ABBREVIATION","DBSOURCE","LINKSTO","CATEGORY","UNITNAME","PARAM_TYPE","CONCEPTID"

12716,220050,"Arterial Blood Pressure systolic","ABPs","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",
12717,220051,"Arterial Blood Pressure diastolic","ABPd","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",
12718,220052,"Arterial Blood Pressure mean","ABPm","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",

12734,220179,"Non Invasive Blood Pressure systolic","NBPs","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",
12735,220180,"Non Invasive Blood Pressure diastolic","NBPd","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",
12736,220181,"Non Invasive Blood Pressure mean","NBPm","metavision","chartevents","Routine Vital Signs","mmHg","Numeric",
"""

# ITEMID list
vital_ids = [

    # Arterial Blood Pressure (invasive)
    220050, # ABPs (systolic)
    220051, # ABPd (diastolic)
    220052, # ABPm (mean)

    # Non-Invasive Blood Pressure (cuff)
    220179, # NBPs (systolic)
    220180, # NBPd (diastolic)
    220181, # NBPm (mean)
]


# Load CHARTEVENTS
lf = (
    pl.scan_csv("/labs/hulab/MIMICIII-v1.4/CHARTEVENTS.csv", infer_schema_length=None)
    .select([
        "SUBJECT_ID",
        "HADM_ID",
        pl.col("CHARTTIME").str.to_datetime(format="%Y-%m-%d %H:%M:%S"),
        "ITEMID",
        "VALUENUM"
    ])
    .filter(pl.col("ITEMID").is_in(vital_ids))
)


# Wide pivot using group_by + max()
agg_exprs = [
    (
        pl.when(pl.col("ITEMID") == iid)
          .then(pl.col("VALUENUM"))
          .otherwise(None)
          .max()
          .alias(str(iid))
    )
    for iid in vital_ids
]

lf_wide = lf.group_by(["SUBJECT_ID", "HADM_ID", "CHARTTIME"]).agg(agg_exprs)

df_wide = lf_wide.collect()


# Rename for readability (same pattern)
df_final = df_wide.rename({

    "220050": "ABPs",
    "220051": "ABPd",
    "220052": "ABPm",

    "220179": "NBPs",
    "220180": "NBPd",
    "220181": "NBPm",
})


# Save
df_final.write_parquet("vital_panel_wide.parquet")
