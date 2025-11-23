#!/usr/bin/env python3

import os
import csv
import wfdb
import polars as pl
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from stp2_wav_reader_extraction import *  # kept even if unused here

# -------------------------------------- #
RECORDS_PATH = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms"
BASE_DIR     = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
ADMISSIONS_PATH = "/labs/hulab/MIMICIII-v1.4/ADMISSIONS.csv.gz"
# -------------------------------------- #

# ============================================================
# PART 0 — Extract WFDB Metadata (per waveform file)
# ============================================================

def parse_records_file():
    with open(RECORDS_PATH, "r") as f:
        return [line.strip() for line in f]


def get_subject_id_from_header(hdr, rec):
    """
    Try to get SUBJECT_ID from header comments; if not available,
    fall back to directory name pXXXX -> XXXX.
    """
    if getattr(hdr, "comments", None):
        for c in hdr.comments:
            c = c.strip()
            if "Subject" in c:
                digits = "".join(ch for ch in c if ch.isdigit())
                if digits:
                    return int(digits)

    # Fallback: parse pXXXX from path (e.g., p0000001/p0000001-3000000/...)
    try:
        subject_token = rec.split("/")[1]   # pXXXX...
        return int(subject_token[1:])
    except Exception:
        return None


def extract_wave_header(rec):
    """
    Read WFDB header to get WAVE_START, WAVE_END, fs, and SUBJECT_ID.
    """
    full_path = os.path.join(BASE_DIR, rec)
    try:
        hdr = wfdb.rdheader(full_path)
        start = hdr.base_datetime
        fs = hdr.fs if hdr.fs else None

        if start is None or fs is None or hdr.sig_len is None:
            return None

        end = start + timedelta(seconds=hdr.sig_len / fs)
        subj = get_subject_id_from_header(hdr, rec)
        if subj is None:
            return None

        return {
            "SUBJECT_ID": subj,
            "record_path": rec,
            "WAVE_START": start,
            "WAVE_END": end,
            "fs": fs,
        }
    except Exception as e:
        print("Error loading header for", rec, ":", e)
        return None


records = parse_records_file()
wave_rows = []

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(extract_wave_header, rec): rec for rec in records}
    for fut in as_completed(futures):
        res = fut.result()
        if res is not None:
            wave_rows.append(res)

wave_df = pl.from_dicts(wave_rows)

# Write waveform_records_info.csv (raw header info, per WFDB record)
wave_records = (
    wave_df
)

OUTPUT_CSV = "waveform_records_info.csv"
wave_records.write_csv(OUTPUT_CSV)
print(f"Extracted WFDB metadata: {wave_records.height} records → {OUTPUT_CSV}")

# ============================================================
# PART 1 — Admissions join and waveform–admission valid windows
# ============================================================

# Admissions
adm = (
    pl.read_csv(ADMISSIONS_PATH)
    .with_columns([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8),
        pl.col("ADMITTIME").str.strptime(pl.Datetime, strict=False),
        pl.col("DISCHTIME").str.strptime(pl.Datetime, strict=False),
    ])
    .lazy()
)

# Waveform headers (from wave_df, not from CSV)
wave_lazy = (
    wave_df
    .select([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("record_path"),
        pl.col("WAVE_START"),
        pl.col("WAVE_END"),
        pl.col("fs").cast(pl.Float64),
    ])
    .lazy()
)

# Join admissions and waveforms on SUBJECT_ID, then compute overlap with admission window
candidate_join = adm.join(
    wave_lazy,
    on="SUBJECT_ID",
    how="full"
)

wf_verified = (
    candidate_join
    .with_columns([
        pl.max_horizontal("ADMITTIME", "WAVE_START").alias("VALID_START"),
        pl.min_horizontal("DISCHTIME", "WAVE_END").alias("VALID_END"),
    ])
    .with_columns([
        (pl.col("VALID_END") - pl.col("VALID_START"))
        .dt.total_seconds()
        .alias("OVERLAP_SEC")
    ])
    .filter(pl.col("OVERLAP_SEC") > 0)
    .select([
        "SUBJECT_ID",
        "HADM_ID",
        "ADMITTIME",
        "DISCHTIME",
        "record_path",
        "WAVE_START",
        "WAVE_END",
        "VALID_START",
        "VALID_END",
        "OVERLAP_SEC",
        "fs",
    ])
    .filter(pl.col("record_path").is_not_null())
    .unique()
    .collect()
    .sort(["SUBJECT_ID", "HADM_ID", "VALID_START"])
)

print("Verified waveform–admission rows:", wf_verified.height)

# Base waveform–admission mapping (keep ADMITTIME / DISCHTIME for history)
wave_base = (
    wf_verified
    .select([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8),
        pl.col("ADMITTIME"),
        pl.col("DISCHTIME"),
        pl.col("record_path"),
        pl.col("VALID_START"),
        pl.col("VALID_END"),
        pl.col("fs").cast(pl.Float64),
    ])
    .lazy()
)

# ============================================================
# PART 1A — Load labs and vitals
# ============================================================
lab_cols = ["Potassium", "Calcium", "Sodium", "Glucose",
        "Lactate", "Creatinine"]
labs = (
    pl.read_parquet("cmp_panel_wide.parquet")
    .select([
        "SUBJECT_ID", "HADM_ID", "CHARTTIME", *lab_cols ])
    .with_columns([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8),
        pl.col("CHARTTIME").cast(pl.Datetime),
        pl.col([
            "Potassium", "Calcium", "Sodium", "Glucose",
            "Lactate", "Creatinine",
        ]).cast(pl.Float64),
    ])
    .lazy()
)

vital_cols = ["NBPs", "NBPd", "NBPm"]

vitals = (
    pl.read_parquet("vital_panel_wide.parquet")
    .select(["SUBJECT_ID", "HADM_ID", "CHARTTIME", *vital_cols])
    .with_columns([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8),
        pl.col("CHARTTIME").cast(pl.Datetime),
        pl.col(vital_cols).cast(pl.Float64),
    ])
    .with_columns([
        # hard cap: values > 300 treated as invalid
        pl.when(pl.col(c) > 300).then(None).otherwise(pl.col(c)).alias(c)
        for c in vital_cols
    ]).filter(
        pl.col("NBPs").is_not_null()
        | pl.col("NBPd").is_not_null()
        | pl.col("NBPm").is_not_null()
    )
    .lazy()
)

# ============================================================
# PART 1B — Overlap (Labs/Vitals within waveform valid window)
# ============================================================

# Labs within each (SUBJECT_ID, HADM_ID, VALID_START–VALID_END)
joined_lab = labs.join(
    wave_base,
    on=["SUBJECT_ID", "HADM_ID"],
    how="full"
).filter(
    (pl.col("SUBJECT_ID") == pl.col("SUBJECT_ID_right")) &
    (pl.col("HADM_ID") == pl.col("HADM_ID_right")) &
    (pl.col("CHARTTIME") >= pl.col("VALID_START")) &
    (pl.col("CHARTTIME") <= pl.col("VALID_END"))
).drop(
    ["SUBJECT_ID_right", "HADM_ID_right"]
)


# VITALS within each waveform interval
joined_vital = vitals.join(
    wave_base,
    on=["SUBJECT_ID", "HADM_ID"],
    how="full"
).filter(
    (pl.col("SUBJECT_ID") == pl.col("SUBJECT_ID_right")) &
    (pl.col("HADM_ID") == pl.col("HADM_ID_right")) &
    (pl.col("CHARTTIME") >= pl.col("VALID_START")) &
    (pl.col("CHARTTIME") <= pl.col("VALID_END") )
).drop(
    ["SUBJECT_ID_right", "HADM_ID_right"]
)

#>>>>>>>>>>>>>>>>>>>>>>>>>
# valid lab encounters
valid_lab_hadm = (
    joined_lab
    .with_columns(
        pl.max_horizontal([pl.col(c).is_not_null() for c in lab_cols]).alias("lab_nonnull")
    )
    .group_by("HADM_ID")
    .agg(
        pl.col("lab_nonnull").max().alias("has_lab")   # group-level Boolean
    )
    .filter(pl.col("has_lab"))
    .select("HADM_ID")
)

# valid vital encounters
valid_vital_hadm = (
    joined_vital
    .with_columns(
        pl.max_horizontal([pl.col(c).is_not_null() for c in vital_cols]).alias("vital_nonnull")
    )
    .group_by("HADM_ID")
    .agg(
        pl.col("vital_nonnull").max().alias("has_vital")
    )
    .filter(pl.col("has_vital"))
    .select("HADM_ID")
)


# intersection (still LazyFrame)
overlap_hadm = valid_lab_hadm.join(valid_vital_hadm, on="HADM_ID", how="inner")

labs_filtered = joined_lab.join(overlap_hadm, on="HADM_ID", how="semi")
vitals_filtered = joined_vital.join(overlap_hadm, on="HADM_ID", how="semi")

# --------------------------------------------------------------- #

labs_tagged = labs_filtered.with_columns(pl.lit("lab").alias("source"))
vitals_tagged = vitals_filtered.with_columns(pl.lit("vital").alias("source"))

df_overlap = (
    pl.concat([labs_tagged, vitals_tagged], how="diagonal")
    .select([
        "SUBJECT_ID", "HADM_ID", "VALID_START", "VALID_END", "fs", "record_path", "CHARTTIME",
        "Potassium","Calcium","Sodium","Glucose","Lactate","Creatinine",
        "NBPs", "NBPd", "NBPm",
    ])
    .collect()
    .sort(["SUBJECT_ID", "HADM_ID", "CHARTTIME"])
)


print("Built df_overlap (lab+vital+wave, per waveform file). Rows:", df_overlap.height)

# ============================================================
# PART 1C — History (Labs/Vitals BEFORE waveform, within admission)
# ============================================================

# We want labs and vitals between ADMITTIME and VALID_START (history before waveform)
# First we need wave_base with ADMITTIME and VALID_START as a separate lazy table
wave_hist_base = (
    wf_verified
    .group_by(["SUBJECT_ID", "HADM_ID"])
    .agg([
        pl.col("VALID_START").min().alias("VALID_START"),
        pl.col("VALID_END").max().alias("VALID_END"),
    ])
    .with_columns([
        pl.col("SUBJECT_ID").cast(pl.Utf8),
        pl.col("HADM_ID").cast(pl.Utf8),
    ])
    .lazy()
)


joined_hist_lab = (
    labs.join(
        wave_hist_base,
        on=["SUBJECT_ID"],
        how="full",
    )
    .filter(
        (pl.col("SUBJECT_ID") == pl.col("SUBJECT_ID_right")) &
        (pl.col("CHARTTIME") <= pl.col("VALID_START"))
    ).drop(["SUBJECT_ID_right", "HADM_ID_right"])
)

joined_hist_vital = (
    vitals.join(
        wave_hist_base,
        on=["SUBJECT_ID"],
        how="full",
    )
    .filter(
        (pl.col("SUBJECT_ID") == pl.col("SUBJECT_ID_right")) &
        (pl.col("CHARTTIME") <= pl.col("VALID_START"))
    ).drop(["SUBJECT_ID_right", "HADM_ID_right"])
)

#>>>>>>>>>>>>>>>>>>>>>>>>>
df_history = (
    joined_hist_lab.join(
        joined_hist_vital,
        on = ["SUBJECT_ID", "HADM_ID", "VALID_START", "VALID_END", "CHARTTIME"], how="full"
    ).with_columns([
        pl.coalesce(["SUBJECT_ID", "SUBJECT_ID_right"]).alias("SUBJECT_ID"),
        pl.coalesce(["HADM_ID", "HADM_ID_right"]).alias("HADM_ID"),
        pl.coalesce(["CHARTTIME", "CHARTTIME_right"]).alias("CHARTTIME"),
        pl.coalesce(["VALID_START", "VALID_START_right"]).alias("VALID_START"),
        pl.coalesce(["VALID_END", "VALID_END_right"]).alias("VALID_END"),
    ])
).drop(
    ["SUBJECT_ID_right", "HADM_ID_right", "CHARTTIME_right",  "VALID_START_right", "VALID_END_right"]
).select(
    ["SUBJECT_ID", "HADM_ID", "VALID_START", "VALID_END", "CHARTTIME", #
     "Potassium","Calcium","Sodium","Glucose","Lactate","Creatinine",
     "NBPs", "NBPd", "NBPm"]
     ).collect().sort(["SUBJECT_ID","HADM_ID","CHARTTIME"])

print("Built df_history (pre-wave admission history). Rows:", df_history.height)

# ============================================================
# PART 1D — Apply lab outlier rules + negative-value cleaning
# ============================================================

# 1) Lab outlier rules (explicit)
lab_outlier_rules = {
    "Glucose": {"min": 0.1, "max": 2000.0},
    # Add more labs if desired, e.g.:
    # "Potassium": {"min": 1.0, "max": 8.0},
}

for col, rules in lab_outlier_rules.items():
    if col in df_overlap.columns:
        df_overlap = df_overlap.with_columns(
            pl.when(
                (pl.col(col) < rules["min"]) | (pl.col(col) > rules["max"])
            ).then(None).otherwise(pl.col(col)).alias(col)
        )
    if col in df_history.columns:
        df_history = df_history.with_columns(
            pl.when(
                (pl.col(col) < rules["min"]) | (pl.col(col) > rules["max"])
            ).then(None).otherwise(pl.col(col)).alias(col)
        )

# 2) Negative value cleaning for all labs and vitals
all_numeric_cols = [
    *lab_cols,
    *vital_cols,
]

for col in all_numeric_cols:
    if col in df_overlap.columns:
        df_overlap = df_overlap.with_columns(
            pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).alias(col)
        )
    if col in df_history.columns:
        df_history = df_history.with_columns(
            pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).alias(col)
        )

print("Applied lab outlier rules and negative-value cleaning.")

# ============================================================
# PART 1E — Save overlap + history tables (main requested outputs)
# ============================================================

df_overlap.write_csv("mimic_lab_vital_waveform_overlap.csv")
df_history.write_csv("mimic_lab_vital_waveform_history.csv")

print("Saved:")
print("  mimic_lab_vital_waveform_overlap.csv")
print("  mimic_lab_vital_waveform_history.csv")

# ============================================================
# PART 2 — Window extraction and quality filtering (uses df_overlap)
# ============================================================

# Compute per-subject windows for waveform extraction around lab/vital events
extract_windows = (
    df_overlap.lazy()
    .group_by("SUBJECT_ID")
    .agg([
        (pl.col("CHARTTIME").min() - pl.duration(hours=6)).alias("wave_extract_start"),
        (pl.col("CHARTTIME").max() + pl.duration(hours=6)).alias("wave_extract_end"),
    ])
    .collect()
)

df_with_windows = (
    df_overlap
    .join(extract_windows, on="SUBJECT_ID", how="left")
    .with_columns([
        pl.when(pl.col("wave_extract_start") < pl.col("VALID_START"))
          .then(pl.col("VALID_START"))
          .otherwise(pl.col("wave_extract_start"))
          .alias("extract_start"),

        pl.when(pl.col("wave_extract_end") > pl.col("VALID_END"))
          .then(pl.col("VALID_END"))
          .otherwise(pl.col("wave_extract_end"))
          .alias("extract_end"),
    ])
)

df_with_index = (
    df_with_windows
    .with_columns([
        (
            (pl.col("extract_start") - pl.col("VALID_START"))
            .dt.total_seconds()
            * pl.col("fs")
        )
        .cast(pl.Int64)
        .alias("index_start"),

        (
            (pl.col("extract_end") - pl.col("VALID_START"))
            .dt.total_seconds()
            * pl.col("fs")
        )
        .cast(pl.Int64)
        .alias("index_end"),
    ])
    .select([
        "SUBJECT_ID", "HADM_ID", "record_path", "VALID_START", "VALID_END",
        "extract_start", "extract_end", "index_start", "index_end",
    ])
    .unique()
)

df = (
    df_with_index
    .unique()
    .with_columns([
        pl.col("SUBJECT_ID").cast(pl.Int64),
        pl.col("HADM_ID").cast(pl.Int64),
        (
            (pl.col("VALID_END").cast(pl.Int64) - pl.col("VALID_START").cast(pl.Int64))
            / (1000000 * 60 * 60)
        )
        .round(3)
        .alias("valid_duration_hour"),
    ])
)

agg_df = (
    df.group_by(["SUBJECT_ID", "HADM_ID"])
    .agg([
        pl.col("VALID_START").min().alias("valid_start_min"),
        pl.col("VALID_END").max().alias("valid_end_max"),
        pl.col("valid_duration_hour").sum().round(3).alias("valid_duration_sum"),
    ])
)

df_filtered = (
    df.join(agg_df, on=["SUBJECT_ID", "HADM_ID"], how="left")
    .with_columns([
        (
            pl.col("valid_duration_sum")
            / (
                (pl.col("valid_end_max") - pl.col("valid_start_min")).cast(pl.Int64)
                / (1000000 * 60 * 60)
            )
        )
        .round(3)
        .alias("valid_ratio"),
    ])
)

df_filtered = (
    df_filtered
    .filter(
        (pl.col("valid_duration_sum") > 24)
        & (pl.col("valid_duration_sum") < 400)
        & (pl.col("valid_ratio") > 0.9)
    )
    .sort(["SUBJECT_ID", "HADM_ID", "VALID_START"])
)

df_filtered.write_csv("mimic_waveform_segments_filtered.csv")

print("Saved mimic_waveform_segments_filtered.csv")
print("Unique SUBJECT_IDs:", df_filtered.select("SUBJECT_ID").n_unique())
print("Unique HADM_IDs:", df_filtered.select("HADM_ID").n_unique())
print("Unique record_path:", df_filtered.select("record_path").n_unique())
