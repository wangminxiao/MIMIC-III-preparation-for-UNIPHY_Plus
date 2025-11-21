#!/usr/bin/env python3

import os
import json
import numpy as np
import polars as pl
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# CONFIG

data_path = "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3"
to_path   = "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3_lab_vital/"

os.makedirs(to_path, exist_ok=True)

with open("./mimic_high_quality_info_list.json", "r") as f:
    ppg_meta = json.load(f)

file_list = [entry[0] for entry in ppg_meta]

lab_overlap_path   = "./mimic_wav_lab_overlap.csv"
vital_overlap_path = "./mimic_wav_vital_overlap.csv"

lab_cols = [
    "Potassium", "Calcium", "Sodium", "Glucose",
    "Lactate", "Creatinine",
]

vital_cols = [
    "ABPs", "ABPd", "ABPm",
    "NBPs", "NBPd", "NBPm",
]

robust_q_low  = 0.005
robust_q_high = 0.98

use_gpt_update = True

smp_rate = 40
seg_len  = 30
samples_per_seg = smp_rate * seg_len

batch_size_gpt = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_chennel = "PLETH40"
encoder_name = "GPT19M"
EMBED_NAME = f"emb_{current_chennel}_{encoder_name}"


# LOAD LAB + VITAL TABLES

lab_df = (
    pl.read_csv(lab_overlap_path, try_parse_dates=True)
    .select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + lab_cols)
    .unique()
)

vital_df = (
    pl.read_csv(vital_overlap_path, try_parse_dates=True)
    .select(["SUBJECT_ID", "HADM_ID", "CHARTTIME"] + vital_cols)
    .unique()
)

# cast lab/vital to floats and clean NaNs -> nulls
lab_df = lab_df.with_columns([
    pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in lab_cols
])
lab_df = lab_df.with_columns([
    pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
    for c in lab_cols
])

vital_df = vital_df.with_columns([
    pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in vital_cols
])
vital_df = vital_df.with_columns([
    pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
    for c in vital_cols
])

# GLOBAL ROBUST NORMALIZATION STATS

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


# GPT ENCODER (optional)

if use_gpt_update:
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


# HELPERS

def parse_filename(fname):
    # SUBJECT_HADM_CLIPSTR_PLETH40_II120_II500_N.npz
    core = fname[:-4] if fname.endswith(".npz") else fname
    parts = core.split("_")
    if len(parts) < 3:
        return None
    try:
        subject = int(parts[0])
        hadm    = int(parts[1])
        return subject, hadm
    except Exception:
        return None

class SignalDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor.float()
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]

def replace_clamped_vectorized(arr, global_min, global_max):
    arr_out = arr.copy()
    mask_valid = (arr != global_min) & (arr != global_max)
    local_min = np.where(mask_valid, arr, np.inf).min(axis=1)
    local_max = np.where(mask_valid, arr, -np.inf).max(axis=1)
    local_min = np.where(np.isfinite(local_min), local_min, global_min)
    local_max = np.where(np.isfinite(local_max), local_max, global_max)
    local_min = local_min[:, None]
    local_max = local_max[:, None]
    arr_out = np.where(arr == global_min, local_min, arr_out)
    arr_out = np.where(arr == global_max, local_max, arr_out)
    return arr_out

def recompute_gpt_embeddings(wave_arr):
    if wav_encoder is None:
        return None
    if wave_arr.ndim != 2 or wave_arr.shape[1] != samples_per_seg:
        return None

    waveform = wave_arr.astype(np.float32)
    global_min = waveform.min()
    global_max = waveform.max()
    waveform = replace_clamped_vectorized(waveform, global_min, global_max)

    seg_min = waveform.min(axis=1, keepdims=True)
    seg_max = waveform.max(axis=1, keepdims=True)
    seg_range = np.where((seg_max - seg_min) == 0, 1.0, seg_max - seg_min)
    norm_waveform = (waveform - seg_min) / seg_range

    dataset = SignalDataset(torch.from_numpy(norm_waveform))
    loader = DataLoader(
        dataset,
        batch_size=batch_size_gpt,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device).view(-1, seg_len, 40)
            enc = wav_encoder.encoding(batch)
            outs.append(enc.cpu())

    return torch.cat(outs, dim=0).numpy()

def build_save_kwargs(dat, time_arr, ehr_mask, ehr_gt, ehr_trend, emb_arr):
    save_kwargs = {k: dat[k] for k in dat.files}
    save_kwargs["time"]      = time_arr
    save_kwargs["ehr_mask"]  = ehr_mask.astype(np.int32)
    save_kwargs["ehr_gt"]    = ehr_gt.astype(np.float32)
    save_kwargs["ehr_trend"] = ehr_trend.astype(np.float32)
    if emb_arr is not None:
        save_kwargs[EMBED_NAME] = emb_arr
    return save_kwargs

def normalize_column_inplace(ehr_gt, ehr_trend, ehr_mask, col_index, name, stats_dict):
    if name not in stats_dict:
        return
    low, high = stats_dict[name]
    mask_j = ehr_mask[:, col_index] > 0
    if not np.any(mask_j):
        return
    vals_gt = ehr_gt[mask_j, col_index]
    vals_tr = ehr_trend[mask_j, col_index]

    vals_gt = np.clip(vals_gt, low, high)
    vals_tr = np.clip(vals_tr, low, high)

    scale = high - low
    vals_gt = (vals_gt - low) / scale
    vals_tr = (vals_tr - low) / scale

    ehr_gt[mask_j, col_index]    = vals_gt
    ehr_trend[mask_j, col_index] = vals_tr


# MAIN LOOP

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

    try:
        dat = np.load(full_path, allow_pickle=True)
    except Exception as e:
        print("Failed to load:", fname, e)
        continue

    if "time" not in dat or current_chennel not in dat.files:
        continue

    raw_time = dat["time"]
    if np.issubdtype(raw_time.dtype, np.datetime64):
        time_arr = raw_time.astype("datetime64[ms]")
    else:
        time_arr = raw_time.astype("int64").astype("datetime64[ms]")

    T = len(time_arr)

    # waveform for GPT embedding (optional)
    wave_arr = dat[current_chennel]
    emb_arr = None
    if use_gpt_update:
        emb_arr = recompute_gpt_embeddings(wave_arr)
    else:
        if EMBED_NAME in dat.files:
            emb_arr = dat[EMBED_NAME]

    # labs for this encounter (raw)
    labs = (
        lab_df
        .filter((pl.col("SUBJECT_ID") == subject_id) & (pl.col("HADM_ID") == hadm_id))
        .sort("CHARTTIME")
    )
    labs_original = labs.select(["SUBJECT_ID","HADM_ID","CHARTTIME"] + lab_cols)

    # vitals for this encounter (raw)
    vitals = (
        vital_df
        .filter((pl.col("SUBJECT_ID") == subject_id) & (pl.col("HADM_ID") == hadm_id))
        .sort("CHARTTIME")
    )
    vitals_original = vitals.select(["SUBJECT_ID","HADM_ID","CHARTTIME"] + vital_cols)

    # if no labs and no vitals, skip
    if labs.is_empty() and vitals.is_empty():
        continue

    # align lab times to nearest segment
    if not labs.is_empty():
        lab_time = labs["CHARTTIME"].to_numpy().astype("datetime64[ms]")
        idx_lab = np.searchsorted(time_arr, lab_time, side="left")
        idx_lab = np.clip(idx_lab, 1, T - 1)
        left_diff = lab_time - time_arr[idx_lab - 1]
        ok_lab = (left_diff > np.timedelta64(0, "s")) & (left_diff <= np.timedelta64(60, "s"))
        labs = labs.filter(pl.Series(ok_lab))
        labs_original = labs_original.filter(pl.Series(ok_lab))
        idx_lab = idx_lab[ok_lab]
        if labs.is_empty():
            idx_lab = None
    else:
        idx_lab = None

    # align vital times to nearest segment
    if not vitals.is_empty():
        vital_time = vitals["CHARTTIME"].to_numpy().astype("datetime64[ms]")
        idx_vit = np.searchsorted(time_arr, vital_time, side="left")
        idx_vit = np.clip(idx_vit, 1, T - 1)
        left_diff_v = vital_time - time_arr[idx_vit - 1]
        ok_v = (left_diff_v > np.timedelta64(0, "s")) & (left_diff_v <= np.timedelta64(60, "s"))
        vitals = vitals.filter(pl.Series(ok_v))
        vitals_original = vitals_original.filter(pl.Series(ok_v))
        idx_vit = idx_vit[ok_v]
        if vitals.is_empty():
            idx_vit = None
    else:
        idx_vit = None

    if (idx_lab is None or labs.is_empty()) and (idx_vit is None or vitals.is_empty()):
        # nothing aligned within 60s window
        continue

    num_labs = len(lab_cols)
    num_vits = len(vital_cols)
    total_dim = num_labs + num_vits

    ehr_mask  = np.zeros((T, total_dim), dtype=np.int32)
    ehr_gt    = np.zeros((T, total_dim), dtype=np.float32)
    ehr_trend = np.zeros((T, total_dim), dtype=np.float32)

    # build labs (raw) into first part of ehr_*
    if idx_lab is not None and not labs.is_empty():
        labs = labs.with_columns(pl.Series("interval_id", idx_lab))
        for j, col in enumerate(lab_cols):
            lab_valid = labs.filter(pl.col(col).is_not_null())
            if lab_valid.is_empty():
                continue

            idxv = lab_valid["interval_id"].to_numpy().astype(int) - 1
            valv = lab_valid[col].to_numpy()
            order = np.argsort(idxv)
            idxv, valv = idxv[order], valv[order]

            # real points
            for t_idx, v in zip(idxv, valv):
                if t_idx < 0 or t_idx >= T:
                    continue
                ehr_mask[t_idx, j]  = 2
                ehr_gt[t_idx, j]    = v
                ehr_trend[t_idx, j] = v

            # interpolation between real labs
            for k in range(len(idxv) - 1):
                s_idx, e_idx = idxv[k], idxv[k + 1]
                if e_idx <= s_idx:
                    continue
                v0, v1 = valv[k], valv[k + 1]
                length = e_idx - s_idx
                if length > 1:
                    interp_vals = np.linspace(v0, v1, length + 1)[1:-1]
                    s2 = s_idx + 1
                    e2 = e_idx
                    ehr_mask[s2:e2, j]   = 1
                    ehr_gt[s2:e2, j]     = interp_vals
                    ehr_trend[s2:e2, j]  = interp_vals

    # build vitals (raw) into remaining dimensions
    if idx_vit is not None and not vitals.is_empty():
        vitals = vitals.with_columns(pl.Series("interval_id", idx_vit))
        for j, col in enumerate(vital_cols):
            col_index = num_labs + j
            vit_valid = vitals.filter(pl.col(col).is_not_null())
            if vit_valid.is_empty():
                continue

            idxv = vit_valid["interval_id"].to_numpy().astype(int) - 1
            valv = vit_valid[col].to_numpy()
            order = np.argsort(idxv)
            idxv, valv = idxv[order], valv[order]

            # real vital points
            for t_idx, v in zip(idxv, valv):
                if t_idx < 0 or t_idx >= T:
                    continue
                ehr_mask[t_idx, col_index]  = 2
                ehr_gt[t_idx, col_index]    = v
                ehr_trend[t_idx, col_index] = v

            # interpolate vitals between points (same logic as labs)
            for k in range(len(idxv) - 1):
                s_idx, e_idx = idxv[k], idxv[k + 1]
                if e_idx <= s_idx:
                    continue
                v0, v1 = valv[k], valv[k + 1]
                length = e_idx - s_idx
                if length > 1:
                    interp_vals = np.linspace(v0, v1, length + 1)[1:-1]
                    s2 = s_idx + 1
                    e2 = e_idx
                    ehr_mask[s2:e2, col_index]   = 1
                    ehr_gt[s2:e2, col_index]     = interp_vals
                    ehr_trend[s2:e2, col_index]  = interp_vals

    # normalize all lab+vital columns in-place, using global stats
    for j, col in enumerate(lab_cols):
        normalize_column_inplace(ehr_gt, ehr_trend, ehr_mask, j, col, lab_norm_stats)

    for j, col in enumerate(vital_cols):
        col_index = num_labs + j
        normalize_column_inplace(ehr_gt, ehr_trend, ehr_mask, col_index, col, vital_norm_stats)

    # save normalized EHR arrays (no raw values in NPZ)
    save_kwargs = build_save_kwargs(
        dat,
        time_arr,
        ehr_mask,
        ehr_gt,
        ehr_trend,
        emb_arr,
    )
    np.savez(os.path.join(to_path, fname), **save_kwargs)

    # append raw lab + vital rows into summary list
    for r in labs_original.iter_rows(named=True):
        row = {
            "file": fname,
            "SUBJECT_ID": r["SUBJECT_ID"],
            "HADM_ID": r["HADM_ID"],
            "CHARTTIME": r["CHARTTIME"],
        }
        for c in lab_cols:
            row[c] = r[c]
        for c in vital_cols:
            row[c] = None

        for c in lab_cols:
            if row[c] is not None:
                row[c] = float(row[c])
        for c in vital_cols:
            if row[c] is not None:
                row[c] = float(row[c])

        all_rows.append(row)

    for r in vitals_original.iter_rows(named=True):
        row = {
            "file": fname,
            "SUBJECT_ID": r["SUBJECT_ID"],
            "HADM_ID": r["HADM_ID"],
            "CHARTTIME": r["CHARTTIME"],
        }
        for c in lab_cols:
            row[c] = None
        for c in vital_cols:
            row[c] = r[c]

        for c in lab_cols:
            if row[c] is not None:
                row[c] = float(row[c])
        for c in vital_cols:
            if row[c] is not None:
                row[c] = float(row[c])

        all_rows.append(row)



# SAVE SUMMARY CSV WITH RAW VALUES

if all_rows:
    pl.DataFrame(all_rows).write_csv("./mimic3_lab_vital_alignment_summary.csv")
    print("Summary CSV saved with raw labs and vitals.")
else:
    print("No aligned labs/vitals found for any waveform file.")

print("Done.")
