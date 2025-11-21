import polars as pl


# 1 Load tables
segments = pl.read_csv("mimic_waveform_segments_filtered.csv", try_parse_dates=True)

patients = pl.read_csv("/labs/hulab/MIMICIII-v1.4/PATIENTS.csv", try_parse_dates=True)
admissions = pl.read_csv("/labs/hulab/MIMICIII-v1.4/ADMISSIONS.csv.gz", try_parse_dates=True)
diag = pl.read_csv(
    "/labs/hulab/MIMICIII-v1.4/DIAGNOSES_ICD.csv.gz",
    schema_overrides={"ICD9_CODE": pl.Utf8}
)

d_dict = pl.read_csv(
    "/labs/hulab/MIMICIII-v1.4/D_ICD_DIAGNOSES.csv.gz",
    dtypes={"ICD9_CODE": pl.Utf8},
    columns=["ICD9_CODE","SHORT_TITLE","LONG_TITLE"]
)



# 2 Build demographics (age, gender, ethnicity, etc.)
demo = (
    admissions.join(patients, on="SUBJECT_ID", how="inner")
    .with_columns([
        # Compute age in years at hospital admission
        ((pl.col("ADMITTIME") - pl.col("DOB")).dt.total_seconds() / 31557600.0)
          .alias("age_raw")
    ])
    .with_columns([
        pl.when(pl.col("age_raw") >= 89).then(90.0)   # cap at 90 for de-ID
          .when(pl.col("age_raw") < 0).then(None)     # invalid DOB
          .otherwise(pl.col("age_raw"))
          .alias("age_at_admit")
    ])
    .select([
        "SUBJECT_ID","HADM_ID",
        "GENDER","age_at_admit",
        "ETHNICITY","INSURANCE","LANGUAGE","MARITAL_STATUS"
    ])
)


# 3 Primary diagnosis per admission (SEQ_NUM = 1)
primary_diag = (
    diag.filter(pl.col("SEQ_NUM") == 1)
        .join(d_dict, on="ICD9_CODE", how="left")
        .select([
            "SUBJECT_ID","HADM_ID",
            "ICD9_CODE","SHORT_TITLE","LONG_TITLE"
        ])
)


# 4 Restrict to admissions that appear in waveform segments
seg_keys = segments.select(["SUBJECT_ID","HADM_ID"]).unique()


# 5 Merge everything
demo_with_diag = (
    seg_keys
    .join(demo, on=["SUBJECT_ID","HADM_ID"], how="left")
    .join(primary_diag, on=["SUBJECT_ID","HADM_ID"], how="left")
    .unique(subset=["SUBJECT_ID","HADM_ID"])
)


# 6 Save final output
demo_with_diag.write_csv("mimic_patient_admission_demo_with_diag.csv")

print("Rows:", demo_with_diag.shape[0])
print("Columns:", demo_with_diag.columns)
print(demo_with_diag.head(5))



# 7 Summary stats: unique counts
categorical_cols = ["GENDER","ETHNICITY","INSURANCE","LANGUAGE","MARITAL_STATUS","ICD9_CODE"]

for col in categorical_cols:
    n_unique = demo_with_diag[col].n_unique()
    print(f"{col}: {n_unique} unique values")


print("SHORT_TITLE unique:", demo_with_diag["SHORT_TITLE"].n_unique())
print("LONG_TITLE unique:", demo_with_diag["LONG_TITLE"].n_unique())