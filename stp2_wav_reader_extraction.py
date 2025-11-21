import os
import csv
import wfdb
import polars as pl
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from stp2_wav_reader_extraction import *

RECORDS_PATH = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms"
BASE_DIR     = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"


# Read waveform RECORDS list
def parse_records_file():
    with open(RECORDS_PATH, "r") as f:
        return [line.strip() for line in f]


# Extract waveform header metadata
def extract_record_info(rec):
    hdr = wfdb.rdheader(os.path.join(BASE_DIR, rec))
    start = hdr.base_datetime
    fs = hdr.fs if hdr.fs else None
    end = start + timedelta(seconds=(hdr.sig_len / fs)) if fs and hdr.sig_len else None
    subject_id = int(rec.split('/')[1][1:])
    return {
        "subject_id": subject_id,
        "record_path": rec,
        "start_time": start.isoformat(),
        "end_time": end.isoformat() if end else "",
        "fs": fs,
    }


# Load waveform metadata
records = parse_records_file()
rows = []

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(extract_record_info, rec): rec for rec in records}
    for future in as_completed(futures):
        try:
            rows.append(future.result())
        except Exception as e:
            print("Error loading", futures[future], e)


# Save waveform metadata
OUTPUT_CSV = "./waveform_records_info.csv"
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["subject_id", "record_path", "start_time", "end_time", "fs"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted info for {len(rows)} records → {OUTPUT_CSV}")


# Load labs, vitals, and waveform metadata
labs   = pl.read_parquet("cmp_panel_wide.parquet").lazy()
vitals = pl.read_parquet("vital_panel_wide.parquet").lazy()
wave   = pl.read_csv("waveform_records_info.csv", try_parse_dates=True).lazy()


# Match waves with labs
joined_lab = labs.join_where(
    wave,
    pl.col("SUBJECT_ID") == pl.col("subject_id"),
    pl.col("CHARTTIME") >= pl.col("start_time"),
    pl.col("CHARTTIME") <= pl.col("end_time")
)

df_lab = (
    joined_lab
    .sort(["SUBJECT_ID", "CHARTTIME"])
    .collect()
    .select([
        "SUBJECT_ID","HADM_ID","CHARTTIME",
        "start_time","end_time","record_path","fs",
        "Potassium","Calcium","Sodium","Glucose","Lactate","Creatinine"
    ])
)

df_lab.write_csv("mimic_wav_lab_overlap.csv")
print("Saved: mimic_wav_lab_overlap.csv")


# Match waves with vitals
joined_vital = vitals.join_where(
    wave,
    pl.col("SUBJECT_ID") == pl.col("subject_id"),
    pl.col("CHARTTIME") >= pl.col("start_time"),
    pl.col("CHARTTIME") <= pl.col("end_time")
)

df_vital = (
    joined_vital
    .sort(["SUBJECT_ID", "CHARTTIME"])
    .collect()
    .select([
        "SUBJECT_ID","HADM_ID","CHARTTIME",
        "start_time","end_time","record_path","fs",
        "GM",
        "ABPs","ABPd","ABPm",
        "NBPs","NBPd","NBPm"
    ])
)

df_vital.write_csv("mimic_wav_vital_overlap.csv")
print("Saved: mimic_wav_vital_overlap.csv")


# Load historical labs (before waveform start)
joined_history_lab = labs.join_where(
    wave,
    pl.col("SUBJECT_ID") == pl.col("subject_id"),
    pl.col("CHARTTIME") <= pl.col("start_time")
)

df_history_lab = (
    joined_history_lab
    .sort(["SUBJECT_ID", "CHARTTIME"])
    .collect()
    .select([
        "SUBJECT_ID","HADM_ID","CHARTTIME",
        "Potassium","Calcium","Sodium","Glucose","Lactate","Creatinine"
    ])
)

df_history_lab.write_csv("mimic_wav_lab_history.csv")
print("Saved: mimic_wav_lab_history.csv")


# Load historical vitals (before waveform start)
joined_history_vital = vitals.join_where(
    wave,
    pl.col("SUBJECT_ID") == pl.col("subject_id"),
    pl.col("CHARTTIME") <= pl.col("start_time")
)

df_history_vital = (
    joined_history_vital
    .sort(["SUBJECT_ID", "CHARTTIME"])
    .collect()
    .select([
        "SUBJECT_ID","HADM_ID","CHARTTIME",
        "GM","ABPs","ABPd","ABPm","NBPs","NBPd","NBPm"
    ])
)

df_history_vital.write_csv("mimic_wav_vital_history.csv")
print("Saved: mimic_wav_vital_history.csv")


# Build padded waveform extraction windows
df_overlap = df_lab

extract_windows = (
    df_overlap.group_by("SUBJECT_ID")
    .agg([
        (pl.col("CHARTTIME").min() - pl.duration(hours=6)).alias("wave_extract_start"),
        (pl.col("CHARTTIME").max() + pl.duration(hours=6)).alias("wave_extract_end")
    ])
)

df_with_windows = df_overlap.join(extract_windows, on="SUBJECT_ID", how="left")


# Clamp extraction windows inside waveform range
df_with_windows = df_with_windows.with_columns([
    pl.when(pl.col("wave_extract_start") < pl.col("start_time"))
      .then(pl.col("start_time"))
      .otherwise(pl.col("wave_extract_start"))
      .alias("extract_start"),

    pl.when(pl.col("wave_extract_end") > pl.col("end_time"))
      .then(pl.col("end_time"))
      .otherwise(pl.col("wave_extract_end"))
      .alias("extract_end"),
])


# Convert extract time to sample index
df_with_index = df_with_windows.with_columns([
    ((pl.col("extract_start") - pl.col("start_time")).dt.total_seconds() * pl.col("fs"))
        .cast(pl.Int64).alias("index_start"),
    ((pl.col("extract_end") - pl.col("start_time")).dt.total_seconds() * pl.col("fs"))
        .cast(pl.Int64).alias("index_end"),
]).select([
    "SUBJECT_ID","HADM_ID","record_path","start_time","end_time",
    "extract_start","extract_end","index_start","index_end"
]).unique()


# Compute valid durations
df = df_with_index

df = df.unique().with_columns([
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("HADM_ID").cast(pl.Int64),
    ((pl.col("end_time").cast(pl.Int64) - pl.col("start_time").cast(pl.Int64)) /
     (1000000 * 60 * 60)).round(3).alias("valid_duration_hour")
])

agg_df = df.group_by(["SUBJECT_ID", "HADM_ID"]).agg([
    pl.col("start_time").min().alias("valid_start_min"),
    pl.col("end_time").max().alias("valid_end_max"),
    pl.col("valid_duration_hour").sum().round(3).alias("valid_duration_sum")
])


# Compute ratio of valid waveform coverage
df_filtered = df.join(agg_df, on=["SUBJECT_ID", "HADM_ID"], how="left").with_columns([
    (pl.col("valid_duration_sum") /
     ((pl.col("valid_end_max") - pl.col("valid_start_min")).cast(pl.Int64) /
      (1000000 * 60 * 60))).round(3).alias("valid_ratio")
])


# Filter subjects by total usable waveform duration
df_filtered = df_filtered.filter(
    (pl.col("valid_duration_sum") > 24) &
    (pl.col("valid_duration_sum") < 400) &
    (pl.col("valid_ratio") > 0.9)
).sort(["SUBJECT_ID", "HADM_ID", "start_time"])

df_filtered.write_csv("mimic_waveform_segments_filtered.csv")
print("Saved: mimic_waveform_segments_filtered.csv")

print("Unique SUBJECT_IDs:", df_filtered.select("SUBJECT_ID").n_unique())
print("Unique HADM_IDs:", df_filtered.select("HADM_ID").n_unique())
print("Unique record_path:", df_filtered.select("record_path").n_unique())
