# MIMIC-III Waveform + Lab + Vital Data Preparation Pipeline

This repository implements a complete multi-stage pipeline for transforming raw **MIMIC-III waveform data (WFDB)**, **EHR tables**, **laboratory measurements**, **NBP vitals**, and **demographics** into **synchronized, model-ready NPZ files** suitable for UNIPHY+/UNIPHY-Lab sequence modeling tasks.

Each stage is an independent Python script designed for modular re-execution and debugging.

---

# Software & Package Requirements

Below is a list of the core Python packages used across all stages.

**Core Libraries**
```
python==3.10.13
polars==1.20.0
numpy==2.2.4
pandas==2.2.3
scipy==1.15.2
wfdb==3.4.0
pyarrow=19.0.0
```

**Deep Learning (optional in Stage 3)**
```
torch==2.5.1
torchvision==0.20.1
torchaudio==2.1.0
```
**UNIPHY+ Embedding Dependencies**

* Custom GPT-19M modules (`GPT.gpt`, `load_state_dict_gpt`, provide later)
* CUDA-compatible PyTorch build

---

# Raw Dataset Download

### Access Requirements

To download MIMIC-III datasets you must:

* Register for a PhysioNet account
* Complete human-subjects training (e.g., CITI)
* Sign the MIMIC-III Data Use Agreement
* Request access to both MIMIC-III Clinical and Waveform databases

### Dataset Links & Example Download Commands

**1. MIMIC-III Clinical Database (v1.4)**
[https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)

```bash
aws s3 sync --no-sign-request \
    s3://mimic-iii-physionet-1.4/ ./mimiciii_1.4/
```

**2. MIMIC-III Waveform Database Matched Subset (v1.0)**
[https://physionet.org/content/mimic3wdb-matched/1.0/](https://physionet.org/content/mimic3wdb-matched/1.0/)

```bash
aws s3 sync --no-sign-request \
    s3://mimic-iii-wdb-matched-1.0/ ./mimic3wdb_matched_1.0/
```

Notes:

* AWS CLI is strongly recommended due to dataset size.
* PhysioNet `wget` scripts may be used as an alternative.

---

# Pipeline Structure

Stage 0 → Build lab + vital wide tables
Stage 1 → Map waveform headers with labs/vitals and compute extraction windows
Stage 1B → Build demographics + ICD-9 table
Stage 2 → Extract and resample waveforms (40/120/500 Hz) into NPZ
Stage 3 → Align labs/vitals → waveform, normalize, embed, output final NPZ

---

# Stage 0 — Lab Table and NBP Vital Table

**Source files:**

* `stp0_1_lab_filter.py`
* `stp0_0_vital_filter.py`

### Purpose

These scripts extract:

* CMP lab values (Potassium, Calcium, Sodium, Glucose, Lactate, Creatinine)
* NBP blood pressure measurements (NBPs, NBPd, NBPm)

Both scripts:

* Convert long-format rows into **wide-format** tables
* Group by `SUBJECT_ID`, `HADM_ID`, `CHARTTIME`
* Clean negative values
* Save to efficient Parquet files

**Outputs**

* `cmp_panel_wide.parquet`
* `vital_panel_wide.parquet`

---

# Stage 1 — Waveform × Lab × Vital Temporal Mapping

**Source file:**
`stp1_0_mapping_wav_lab_vital.py`

### Purpose

This is the core matching logic between:

* WFDB waveform directories
* Lab timestamps
* Vital timestamps
* Admission intervals

### Main Steps

#### 1. Extract WFDB header metadata

* `record_path`
* `WAVE_START` / `WAVE_END`
* sampling frequency `fs`
* `SUBJECT_ID`

→ Saved into `waveform_records_info.csv`

#### 2. Join waveform windows with lab/vital timestamps

* Combine lab + vital tables
* Match events inside waveform coverage windows

#### 3. Construct extraction windows

For each subject:

* Pad around events (+/− 6 hours)
* Clip to actual waveform availability
* Convert to sample indices (`index_start`, `index_end`)

#### 4. Remove problematic segments

* Overlapping directories
* Multi-directory conflicts
* Low coverage windows
* Unreasonably short or long waveform windows

**Outputs**

* `mimic_lab_vital_waveform_overlap.csv`
* `mimic_lab_vital_waveform_history.csv` (admission history pre-waveform)
* `mimic_waveform_segments_filtered.csv`

---

# Stage 1B — Demographics + ICD-9 Extraction

**Source file:**
`stp1_1_demographics.py`

### Purpose

Builds a demographic table for every admission that passed Stage 1 filters.

Extracted features include:

* Age at admission (capped at 90 for de-ID)
* Gender, ethnicity, insurance, language, marital status
* Primary ICD-9 diagnosis (from `SEQ_NUM = 1`)
* ICD-9 short/long titles

**Output**

* `mimic_patient_admission_demo_with_diag.csv`

---

# Stage 2 — Waveform Extraction and Resampling

**Source file:**
`stp2_wav_reader_extraction.py`

### Purpose

Convert raw WFDB waveform directories into aligned NPZ waveform files.

### Main Steps

1. **Load WFDB channels** using sample index windows
2. **Concatenate multi-record segments** into continuous time series
3. **Create absolute millisecond timestamps** (`time`)
4. **Resample waveforms**

   * PPG → **40 Hz (PLETH40)**
   * ECG → **120 Hz (II120)**
   * ECG → **500 Hz (II500)**
5. **Generate 30-second windows** (fully synchronized)
6. **Save NPZ** with:

   * `time`
   * `clip_raw_index`
   * `PLETH40`, `II120`, `II500`

**Typical output directory:**
`MIMIC3_SPO2_I_40hz_v3/`

---

# Stage 3 — Lab & Vital Alignment, Normalization, & Embedding

**Source file:**
`stp3_1_npz_prepare.py`

### Purpose

Augment waveform NPZ files with aligned and normalized laboratory and vital trajectories, and optionally add GPT-derived embeddings.

### Main Steps

#### 1. Load waveform NPZ + overlap tables

* Parses filenames (`SUBJECT_ID`, `HADM_ID`)
* Loads full waveform arrays

#### 2. Compute global normalization statistics

* Robust quantile-based normalization
* Applied to:

  * Potassium
  * Calcium
  * Sodium
  * Glucose
  * Lactate
  * Creatinine
  * NBPs / NBPd / NBPm

#### 3. Align labs/vitals to waveform timeline

* Convert waveform time → datetime64
* Match lab/vital `CHARTTIME` to nearest waveform index (≤ 60 seconds)
* Remove out-of-range or invalid matches

#### 4. Build EHR supervision arrays

`ehr_mask` : 0 = none, 1 = interpolated, 2 = observed
`ehr_gt` : normalized value
`ehr_trend` : linearly interpolated trend

Interpolation filled for both labs and vitals.

#### 5. GPT-19M Embedding Generation (optional)

When enabled:

* Clean clamped min/max
* Normalize per segment
* Reshape to `(30, 40)`
* Run through GPT-19M encoder
* Save under:
  `emb_PLETH40_GPT19M`

#### 6. Save final NPZ files

Each final NPZ includes:

* original waveforms
* normalized EHR targets
* optional embeddings
* timestamps and metadata

#### 7. Write raw summary table

* `mimic3_lab_vital_alignment_summary.csv`

**Output directory:**
`MIMIC3_SPO2_I_40hz_v3_lab_vital/`

---

