#!/usr/bin/env python3

import os
import json
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# OPTIONS
# ============================================================================
ONLY_COLLECT_TABLE = True   # If True, only create summary CSV, do NOT modify NPZ files.


# ============================================================================
# CONFIG
# ============================================================================
data_path = "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3"
to_path   = "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3_lab_vital/"

os.makedirs(to_path, exist_ok=True)

with open("./mimic_high_quality_info_list.json", "r") as f:
    ppg_meta = json.load(f)

file_list = [entry[0] for entry in ppg_meta]

# use the combined lab+vital overlap table
overlap_path = "./mimic_lab_vital_waveform_overlap.csv"

lab_cols = [
    "Potassium", "Calcium", "Sodium", "Glucose",
    "Lactate", "Creatinine",
]

# NBP vitals only (these are what exist in the combined overlap CSV)
vital_cols = [
    "NBPs", "NBPd", "NBPm",
]

robust_q_low  = 0.0
robust_q_high = 1.0

use_gpt_update = True

smp_rate = 40
seg_len  = 30
samples_per_seg = smp_rate * seg_len

batch_size_gpt = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_chennel = "PLETH40"
encoder_name = "GPT19M"
EMBED_NAME = f"emb_{current_chennel}_{encoder_name}"


# ============================================================================
# LOAD TABLES (combined lab+vital)
# ============================================================================
overlap_df = (
    pl.read_csv(overlap_path, try_parse_dates=True)
    .select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + lab_cols + vital_cols)
    .unique()
)

# cast IDs to Int64 for matching parsed filename IDs
overlap_df = overlap_df.with_columns([
    pl.col("SUBJECT_ID").cast(pl.Int64, strict=False),
    pl.col("HADM_ID").cast(pl.Int64, strict=False),
])

lab_df = overlap_df.select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + lab_cols)
vital_df = overlap_df.select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + vital_cols)


# ============================================================================
# SAFE CLEANING: STRING → FLOAT → NULL
# ============================================================================
def clean_numeric_columns(df, columns):
    # (1) string → float (non-strict)
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in columns
    ])

    # (2) NaN → null
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in columns
    ])

    # (3) negative values → null (labs and vitals cannot be negative)
    df = df.with_columns([
        pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
        for c in columns
    ])

    return df

lab_df   = clean_numeric_columns(lab_df,   lab_cols)
vital_df = clean_numeric_columns(vital_df, vital_cols)


# ============================================================================
# GLOBAL NORMALIZATION STATS
# ============================================================================
print("Computing robust normalization stats for labs...")
lab_norm_stats = {}
for col in lab_cols:
    s = lab_df[col].drop_nulls()
    if not s.is_empty():
        low  = s.quantile(robust_q_low)
        high = s.quantile(robust_q_high)
        high = max(high, low + 1e-6)
        lab_norm_stats[col] = (low, high)
        print(f"  {col}: {low:.3f} → {high:.3f}")

print("Computing robust normalization stats for vitals...")
vital_norm_stats = {}
for col in vital_cols:
    s = vital_df[col].drop_nulls()
    if not s.is_empty():
        low  = s.quantile(robust_q_low)
        high = s.quantile(robust_q_high)
        high = max(high, low + 1e-6)
        vital_norm_stats[col] = (low, high)
        print(f"  {col}: {low:.3f} → {high:.3f}")


# ============================================================================
# GPT ENCODER
# ============================================================================
if use_gpt_update and not ONLY_COLLECT_TABLE:
    from GPT.gpt import GPT, load_state_dict_gpt
    wav_encoder = GPT(
        patch_size=40,
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0,
        max_len=2400,
    )
    load_state_dict_gpt(
        wav_encoder,
        "/labs/hulab/mxwang/output_per_state_tbtt_branch/align/align_mimic/gpt_19M_align.pt",
    )
    wav_encoder.to(device)
    wav_encoder.eval()
else:
    wav_encoder = None


# ============================================================================
# HELPERS
# ============================================================================
def parse_filename(fname):
    core = fname[:-4] if fname.endswith(".npz") else fname
    parts = core.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except:
        return None


class SignalDataset(Dataset):
    def __init__(self, t):
        self.data = t.float()
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, i):
        return self.data[i]


def replace_clamped_vectorized(arr, gmin, gmax):
    mask = (arr != gmin) & (arr != gmax)
    local_min = np.where(mask, arr, np.inf).min(axis=1)
    local_max = np.where(mask, arr, -np.inf).max(axis=1)
    local_min = np.where(np.isfinite(local_min), local_min, gmin)
    local_max = np.where(np.isfinite(local_max), local_max, gmax)
    arr2 = np.where(arr == gmin, local_min[:, None], arr)
    arr2 = np.where(arr2 == gmax, local_max[:, None], arr2)
    return arr2


def recompute_gpt_embeddings(wave_arr):
    if wav_encoder is None:
        return None

    waveform = wave_arr.astype(np.float32)
    gmin, gmax = waveform.min(), waveform.max()
    waveform = replace_clamped_vectorized(waveform, gmin, gmax)

    seg_min = waveform.min(axis=1, keepdims=True)
    seg_max = waveform.max(axis=1, keepdims=True)
    seg_rng = np.where((seg_max - seg_min) == 0, 1.0, seg_max - seg_min)
    norm = (waveform - seg_min) / seg_rng

    loader = DataLoader(
        SignalDataset(torch.from_numpy(norm)),
        batch_size=batch_size_gpt, shuffle=False, num_workers=4, pin_memory=True,
    )
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device).view(-1, seg_len, 40)
            outs.append(wav_encoder.encoding(batch).cpu())
    return torch.cat(outs, dim=0).numpy()


def build_save_kwargs(dat, time_arr, ehr_mask, ehr_gt, ehr_trend, emb):
    d = {k: dat[k] for k in dat.files}
    d["time"]      = time_arr
    d["ehr_mask"]  = ehr_mask.astype(np.int32)
    d["ehr_gt"]    = ehr_gt.astype(np.float32)
    d["ehr_trend"] = ehr_trend.astype(np.float32)
    if emb is not None:
        d[EMBED_NAME] = emb
    return d


def normalize_column_inplace(gt, trend, mask, j, name, stats):
    if name not in stats:
        return
    low, high = stats[name]
    m = mask[:, j] > 0
    if not np.any(m):
        return
    vals_gt = np.clip(gt[m, j], low, high)
    vals_tr = np.clip(trend[m, j], low, high)
    sc = high - low
    gt[m, j]    = (vals_gt - low) / sc
    trend[m, j] = (vals_tr - low) / sc


def clean_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, np.datetime64):
        return str(v.astype("datetime64[ms]"))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return None


# ============================================================================
# MAIN LOOP
# ============================================================================
all_rows = []

print(f"Processing {len(file_list)} waveform files...")

for fname in tqdm(file_list):

    full_path = os.path.join(data_path, fname)
    if not os.path.exists(full_path):
        continue

    parsed = parse_filename(fname)
    if parsed is None:
        continue
    subject_id, hadm_id = parsed

    dat = None
    time_arr = None

    # ----------------------------------------------------------
    # NORMAL MODE → load NPZ
    # ----------------------------------------------------------
    if not ONLY_COLLECT_TABLE:
        try:
            dat = np.load(full_path, allow_pickle=True)
        except:
            continue

        if "time" not in dat:
            continue

        raw_t = dat["time"]
        if np.issubdtype(raw_t.dtype, np.datetime64):
            time_arr = raw_t.astype("datetime64[ms]")
        else:
            time_arr = raw_t.astype("int64").astype("datetime64[ms]")
        T = len(time_arr)

    # ----------------------------------------------------------
    # LOAD LAB + VITAL ROWS FOR THIS ENCOUNTER
    # ----------------------------------------------------------
    labs = lab_df.filter(
        (pl.col("SUBJECT_ID") == subject_id) & (pl.col("HADM_ID") == hadm_id)
    ).sort("CHARTTIME")
    labs_orig = labs.select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + lab_cols)

    vitals = vital_df.filter(
        (pl.col("SUBJECT_ID") == subject_id) & (pl.col("HADM_ID") == hadm_id)
    ).sort("CHARTTIME")
    vitals_orig = vitals.select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + vital_cols)

    if labs.is_empty() and vitals.is_empty():
        continue

    # ============================================================================
    # TABLE-ONLY MODE
    # ============================================================================
    if ONLY_COLLECT_TABLE:

        # labs
        for r in labs_orig.iter_rows(named=True):
            row = {
                "file": fname,
                "SUBJECT_ID": r["SUBJECT_ID"],
                "HADM_ID": r["HADM_ID"],
                "CHARTTIME": clean_value(r["CHARTTIME"]),
            }
            for c in lab_cols:
                row[c] = clean_value(r[c])
            for c in vital_cols:
                row[c] = None
            all_rows.append(row)

        # vitals
        for r in vitals_orig.iter_rows(named=True):
            row = {
                "file": fname,
                "SUBJECT_ID": r["SUBJECT_ID"],
                "HADM_ID": r["HADM_ID"],
                "CHARTTIME": clean_value(r["CHARTTIME"]),
            }
            for c in lab_cols:
                row[c] = None
            for c in vital_cols:
                row[c] = clean_value(r[c])
            all_rows.append(row)

        continue

    # ============================================================================
    # NORMAL MODE (waveform alignment)
    # ============================================================================
    # align labs
    if not labs.is_empty():
        lab_t = labs["CHARTTIME"].to_numpy().astype("datetime64[ms]")
        idx = np.searchsorted(time_arr, lab_t, side="left")
        idx = np.clip(idx, 1, T - 1)
        diff = lab_t - time_arr[idx - 1]
        ok = (diff > 0) & (diff <= np.timedelta64(60, "s"))
        labs = labs.filter(pl.Series(ok))
        labs_orig = labs_orig.filter(pl.Series(ok))
        idx_lab = idx[ok] if not labs.is_empty() else None
    else:
        idx_lab = None

    # align vitals
    if not vitals.is_empty():
        vit_t = vitals["CHARTTIME"].to_numpy().astype("datetime64[ms]")
        idx = np.searchsorted(time_arr, vit_t, side="left")
        idx = np.clip(idx, 1, T - 1)
        diff = vit_t - time_arr[idx - 1]
        ok = (diff > 0) & (diff <= np.timedelta64(60, "s"))
        vitals = vitals.filter(pl.Series(ok))
        vitals_orig = vitals_orig.filter(pl.Series(ok))
        idx_vit = idx[ok] if not vitals.is_empty() else None
    else:
        idx_vit = None

    if (idx_lab is None) and (idx_vit is None):
        continue

    # initialize EHR arrays
    num_labs = len(lab_cols)
    num_vits = len(vital_cols)
    total_dim = num_labs + num_vits

    ehr_mask  = np.zeros((T, total_dim), dtype=np.int32)
    ehr_gt    = np.zeros((T, total_dim), dtype=np.float32)
    ehr_trend = np.zeros((T, total_dim), dtype=np.float32)

    # GPT embeddings
    wave_arr = dat[current_chennel]
    emb_arr = recompute_gpt_embeddings(wave_arr) if use_gpt_update else dat.get(EMBED_NAME, None)

    # build LAB curves
    if idx_lab is not None:
        labs = labs.with_columns(pl.Series("interval_id", idx_lab))
        for j, col in enumerate(lab_cols):
            dfv = labs.filter(pl.col(col).is_not_null())
            if dfv.is_empty():
                continue

            idxv = dfv["interval_id"].to_numpy().astype(int) - 1
            valv = dfv[col].to_numpy()
            order = np.argsort(idxv)
            idxv, valv = idxv[order], valv[order]

            for t, v in zip(idxv, valv):
                if 0 <= t < T:
                    ehr_mask[t, j]  = 2
                    ehr_gt[t, j]    = v
                    ehr_trend[t, j] = v

            for k in range(len(idxv) - 1):
                s, e = idxv[k], idxv[k + 1]
                if e > s + 1:
                    interp = np.linspace(valv[k], valv[k + 1], e - s + 1)[1:-1]
                    ehr_mask[s+1:e, j]  = 1
                    ehr_gt[s+1:e, j]    = interp
                    ehr_trend[s+1:e, j] = interp

    # build VITAL curves
    if idx_vit is not None:
        vitals = vitals.with_columns(pl.Series("interval_id", idx_vit))
        for j, col in enumerate(vital_cols):
            col2 = num_labs + j
            dfv = vitals.filter(pl.col(col).is_not_null())
            if dfv.is_empty():
                continue

            idxv = dfv["interval_id"].to_numpy().astype(int) - 1
            valv = dfv[col].to_numpy()
            order = np.argsort(idxv)
            idxv, valv = idxv[order], valv[order]

            for t, v in zip(idxv, valv):
                if 0 <= t < T:
                    ehr_mask[t, col2]  = 2
                    ehr_gt[t, col2]    = v
                    ehr_trend[t, col2] = v

            for k in range(len(idxv) - 1):
                s, e = idxv[k], idxv[k + 1]
                if e > s + 1:
                    interp = np.linspace(valv[k], valv[k + 1], e - s + 1)[1:-1]
                    ehr_mask[s+1:e, col2]  = 1
                    ehr_gt[s+1:e, col2]    = 0.
                    ehr_trend[s+1:e, col2] = interp

    # normalization
    for j, col in enumerate(lab_cols):
        normalize_column_inplace(ehr_gt, ehr_trend, ehr_mask, j, col, lab_norm_stats)

    for j, col in enumerate(vital_cols):
        normalize_column_inplace(ehr_gt, ehr_trend, ehr_mask, num_labs + j, col, vital_norm_stats)

    # save NPZ
    np.savez(
        os.path.join(to_path, fname),
        **build_save_kwargs(dat, time_arr, ehr_mask, ehr_gt, ehr_trend, emb_arr)
    )

    # store raw lab+vital rows for summary
    for r in labs_orig.iter_rows(named=True):
        row = {
            "file": fname,
            "SUBJECT_ID": clean_value(r["SUBJECT_ID"]),
            "HADM_ID": clean_value(r["HADM_ID"]),
            "CHARTTIME": clean_value(r["CHARTTIME"]),
        }
        for c in lab_cols:
            row[c] = clean_value(r[c])
        for c in vital_cols:
            row[c] = None
        all_rows.append(row)

    for r in vitals_orig.iter_rows(named=True):
        row = {
            "file": fname,
            "SUBJECT_ID": clean_value(r["SUBJECT_ID"]),
            "HADM_ID": clean_value(r["HADM_ID"]),
            "CHARTTIME": clean_value(r["CHARTTIME"]),
        }
        for c in lab_cols:
            row[c] = None
        for c in vital_cols:
            row[c] = clean_value(r[c])
        all_rows.append(row)


# ============================================================================
# FINAL SUMMARY CSV
# ============================================================================
if all_rows:
    df = pl.DataFrame(all_rows, infer_schema_length=None)

    float_cols = ["SUBJECT_ID", "HADM_ID"] + lab_cols + vital_cols
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in float_cols
    ])

    df.write_csv("./mimic3_lab_vital_alignment_summary.csv")
    print("Summary CSV saved.")

    print("\n=== RELOADING CSV FOR DTYPE ANALYSIS ===")

    summary_df = pl.read_csv("./mimic3_lab_vital_alignment_summary.csv", infer_schema_length=None)

    print("\n=== COLUMN DTYPES ===")
    for col, dtype in zip(summary_df.columns, summary_df.dtypes):
        print(f"{col:20s} {str(dtype):10s}")

    print("\n=== UNIQUE COUNTS FOR EACH COLUMN ===")
    for col in summary_df.columns:
        n_unique = summary_df[col].n_unique()
        print(f"{col:20s} unique={n_unique}")

    print("\n=== FLOAT COLUMN SUMMARY (min, max, unique) ===")
    float_cols = [
        col for col, dtype in zip(summary_df.columns, summary_df.dtypes)
        if dtype in (pl.Float32, pl.Float64)
    ]
    for col in float_cols:
        s = summary_df[col].drop_nulls()
        if s.is_empty():
            print(f"{col:20s} (no numeric data)")
        else:
            print(
                f"{col:20s} unique={s.n_unique():6d} "
                f"min={s.min():10.4f} max={s.max():10.4f}"
            )

else:
    print("No aligned labs/vitals found.")

print("Done.")
