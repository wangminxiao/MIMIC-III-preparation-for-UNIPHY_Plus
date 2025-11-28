#!/usr/bin/env python3
"""
Build a 40 Hz MIMIC-III PPG segment dataset (Multiprocessing version).

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import resample_poly
import multiprocessing

# --------------------------------------------------------------------- #
SAVE_PATH = "MIMIC3_SPO2_I_40hz_v3"
BASE_DIR  = "mimic3wdb_matched_1.0/"
CSV_PATH  = "./mimic_waveform_segments_filtered.csv"

SELECT_CH = ["PLETH", "II"]
TARGET_FS = {"PLETH": 40, "II120": 120, "II500": 500}
WIN_SEC = 30
# --------------------------------------------------------------------- #


def process_one_encounter(encounter_id: int, rows_data: list[dict]):
    sub_df = pl.DataFrame(rows_data).sort("extract_start")
    rows = sub_df.iter_rows(named=True)

    segment_list = []          # (time_seg, pleth_seg, ecg_seg)
    clip_sample_offsets = []   # WFDB clip boundaries in *sample* units
    current_total_samples = 0  # in samples, over concatenated waveform

    src_fs = None
    global_start_dt = None
    global_end_dt = None
    failed_info = None

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
            failed_info = (encounter_id, record_path, str(e))
            # Continue processing other clips, but remember the failure

    # nothing valid
    if not segment_list:
        return failed_info if failed_info else None

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
        return failed_info if failed_info else None

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

    return failed_info if failed_info else None


# --------------------------------------------------------------------- #
def run_parallel(df: pl.DataFrame, max_workers=8):
    groups = df.group_by("HADM_ID").agg(pl.col("row_index").alias("indices"))
    
    tasks = []
    for row in groups.iter_rows(named=True):
        encounter_id = row["HADM_ID"]
        indices = row["indices"]
        rows_data = df[indices].to_dicts()
        tasks.append((encounter_id, rows_data))
    
    failed_reads = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_encounter, encounter_id, rows_data)
            for encounter_id, rows_data in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing encounters"):
            try:
                result = future.result()
                if result is not None:
                    failed_reads.append(result)
            except Exception as e:
                failed_reads.append((None, None, str(e)))
    
    return failed_reads


def main(test_mode=True, max_files=48, max_workers=60) -> None:
    cpu_count = multiprocessing.cpu_count()
    if max_workers > cpu_count:
        raise ValueError(f"max_workers ({max_workers}) > CPU cores ({cpu_count})")
    
    df = pl.read_csv(CSV_PATH, try_parse_dates=True)
    df = df.sort(["SUBJECT_ID", "HADM_ID", "extract_start"]).with_row_index(name="row_index")

    if test_mode:
        print(f"[Test Mode] Limiting to {max_files} waveform records...")
        df = df.head(max_files)

    failed_reads = run_parallel(df, max_workers=max_workers)

    if failed_reads:
        logname = "failed_records_test.log" if test_mode else "failed_records.log"
        with open(logname, "w") as f:
            for eid, path, reason in failed_reads:
                f.write(f"{eid},{path},{reason}\n")
        print(f"[Warning] {len(failed_reads)} failed records logged to {logname}.")

    print("Done.")


# --------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main(test_mode=False)

