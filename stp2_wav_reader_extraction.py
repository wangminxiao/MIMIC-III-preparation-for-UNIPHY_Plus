#!/usr/bin/env python3
"""
Build a 40 Hz MIMIC-III PPG segment dataset.

Concatenate WFDB waveform clips back-to-back (no padding),
keep true timestamps for each sample, and generate 30s windows
with resampled PPG and ECG.

Filename encodes the start indices (in 30s-window units) of each WFDB clip
in the concatenated 30s-window sequence.
"""

import os
import numpy as np
import polars as pl
import wfdb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import resample_poly
import threading

# --------------------------------------------------------------------- #
SAVE_PATH = "/opt/localdata100tb/UNIPHY_Plus/dataset/EST/MIMIC3_SPO2_I_40hz_v3"
BASE_DIR  = "/labs/hulab/MIMIC_waveform_matched_subset/physionet.org/files/mimic3wdb-matched/1.0"
CSV_PATH  = "./mimic_waveform_segments_filtered.csv"

SELECT_CH = ["PLETH", "II"]
TARGET_FS = {"PLETH": 40, "II120": 120, "II500": 500}
WIN_SEC = 30
# --------------------------------------------------------------------- #

FAILED_READS = []
FAILED_READS_LOCK = threading.Lock()


def log_failed(encounter_id, path, reason):
    with FAILED_READS_LOCK:
        FAILED_READS.append((encounter_id, path, reason))


def flush_failed_reads(log_path="failed_records_partial.log"):
    with FAILED_READS_LOCK:
        if not FAILED_READS:
            return
        with open(log_path, "a") as f:
            for eid, path, reason in FAILED_READS:
                f.write(f"{eid},{path},{reason}\n")
        FAILED_READS.clear()


# --------------------------------------------------------------------- #
def process_one_encounter(encounter_id: int, indices: list[int], df: pl.DataFrame):

    # sort rows by extract_start
    sub_df = df[indices].sort("extract_start")
    rows = sub_df.iter_rows(named=True)

    segment_list = []          # (time_seg, pleth_seg, ecg_seg)
    clip_sample_offsets = []   # WFDB clip boundaries in *sample* units
    current_total_samples = 0  # in samples, over concatenated waveform

    src_fs = None
    global_start_dt = None
    global_end_dt = None

    # -------------------------------
    # Load and accumulate WFDB clips
    # -------------------------------
    for row in rows:
        record_path = os.path.join(BASE_DIR, row["record_path"])
        valid_start = row["extract_start"]
        valid_end   = row["extract_end"]

        try:
            header = wfdb.rdheader(record_path)
            if header.base_datetime is None:
                raise ValueError("Missing base_datetime")

            fs = int(header.fs)
            if src_fs is None:
                src_fs = fs
            elif src_fs != fs:
                raise ValueError("Inconsistent sampling rate across clips")

            # convert timestamps
            base_ms  = int(np.datetime64(header.base_datetime, "ms").astype(np.int64))
            start_ms = int(np.datetime64(valid_start,        "ms").astype(np.int64))
            end_ms   = int(np.datetime64(valid_end,          "ms").astype(np.int64))

            idx0 = ((start_ms - base_ms) // 1000) * fs
            idx1 = min(((end_ms - base_ms) // 1000) * fs, header.sig_len)
            if idx0 >= idx1:
                continue

            rec = wfdb.rdrecord(record_path, sampfrom=idx0, sampto=idx1, return_res=16)
            names = rec.sig_name

            if not all(ch in names for ch in SELECT_CH):
                raise ValueError(f"Missing required channels: {SELECT_CH}")

            pleth_seg = rec.p_signal[:, names.index("PLETH")]
            ecg_seg   = rec.p_signal[:, names.index("II")]
            nsamp = len(pleth_seg)

            # --------------------------------------------
            # Record WFDB clip boundary in *sample* units
            # --------------------------------------------
            clip_sample_offsets.append(current_total_samples)
            current_total_samples += nsamp

            # --------------------------------------------
            # Per-sample timestamps
            # --------------------------------------------
            time_seg = start_ms + np.arange(nsamp) * (1000 // fs)

            segment_list.append((time_seg, pleth_seg, ecg_seg))

            # track global start/end (for debugging if needed)
            if global_start_dt is None or valid_start < global_start_dt:
                global_start_dt = valid_start
            if global_end_dt is None or valid_end > global_end_dt:
                global_end_dt = valid_end

        except Exception as e:
            log_failed(encounter_id, record_path, str(e))

    # nothing valid
    if not segment_list:
        return

    # ----------------------------
    # Concatenate back-to-back
    # ----------------------------
    time_all  = np.concatenate([s[0] for s in segment_list])
    pleth_all = np.concatenate([s[1] for s in segment_list])
    ecg_all   = np.concatenate([s[2] for s in segment_list])

    SRC_FS = src_fs
    seg_len = WIN_SEC * SRC_FS  # samples per 30s window
    nseg = len(pleth_all) // seg_len

    pleth40_list = []
    ii120_list   = []
    ii500_list   = []
    time_windows = []

    # ----------------------------
    # Window extraction (30s)
    # ----------------------------
    for j in range(nseg):
        start_idx = j * seg_len

        pseg = pleth_all[start_idx:start_idx + seg_len]
        eseg = ecg_all[start_idx:start_idx + seg_len]
        tseg = time_all[start_idx:start_idx + seg_len]

        if np.isnan(pseg).any() or np.isnan(eseg).any():
            continue

        pleth40 = resample_poly(pseg, TARGET_FS["PLETH"], SRC_FS)
        e120    = resample_poly(eseg, TARGET_FS["II120"], SRC_FS)
        e500    = resample_poly(eseg, TARGET_FS["II500"], SRC_FS)

        pleth40_list.append(pleth40.astype(np.float16))
        ii120_list.append(e120.astype(np.float16))
        ii500_list.append(e500.astype(np.float16))

        time_windows.append(tseg[0])

    if not pleth40_list:
        return

    pleth40 = np.stack(pleth40_list)
    ii120   = np.stack(ii120_list)
    ii500   = np.stack(ii500_list)
    timearr = np.array(time_windows, dtype=np.int64)

    # --------------------------------------------
    # Convert clip sample offsets → segment indices
    # --------------------------------------------
    # "How many 30s segments have passed when this WFDB clip starts?"
    clip_raw = np.array(
        [off // seg_len for off in clip_sample_offsets],
        dtype=np.int64
    )

    # ----------------------------
    # Filename uses segment indices
    # ----------------------------
    clip_str = "-".join(str(v) for v in clip_raw) if len(clip_raw) > 0 else "0"

    out_name = (
        f"{sub_df[0,'SUBJECT_ID']}_{encounter_id}_"
        f"{clip_str}_PLETH40_II120_II500_{len(timearr)}.npz"
    )

    os.makedirs(SAVE_PATH, exist_ok=True)

    # ----------------------------
    # Save dataset
    # ----------------------------
    np.savez(
        os.path.join(SAVE_PATH, out_name),
        time=timearr,
        clip_raw_index=clip_raw,
        PLETH40=pleth40,
        II120=ii120,
        II500=ii500
    )

    flush_failed_reads()


# --------------------------------------------------------------------- #
def run_parallel(df: pl.DataFrame, max_workers=8):
    groups = df.group_by("HADM_ID").agg(pl.col("row_index").alias("indices"))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_encounter, row["HADM_ID"], row["indices"], df)
            for row in groups.iter_rows(named=True)
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing encounters"):
            _.result()


def main(test_mode=True, max_files=48, max_workers=30):
    df = pl.read_csv(CSV_PATH, try_parse_dates=True)
    df = df.sort(["SUBJECT_ID", "HADM_ID", "extract_start"]).with_row_index(name="row_index")

    if test_mode:
        print(f"[Test Mode] Limiting to {max_files} waveform records...")
        df = df.head(max_files)

    run_parallel(df, max_workers=max_workers)

    if FAILED_READS:
        logname = "failed_records_test.log" if test_mode else "failed_records.log"
        with open(logname, "w") as f:
            for eid, path, reason in FAILED_READS:
                f.write(f"{eid},{path},{reason}\n")
        print(f"[Warning] {len(FAILED_READS)} failed records logged.")

    print("Done.")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main(test_mode=False)
