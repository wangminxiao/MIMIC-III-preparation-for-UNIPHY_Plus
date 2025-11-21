# MIMIC-III Waveform + Lab + Vital Data Preparation Pipeline

This repository contains a complete multi-stage pipeline for converting raw MIMIC-III waveform (WFDB) data, EHR tables, laboratory values, and vital signs into synchronized, model-ready NPZ files for UNIPHY+/UNIPHY-Lab applications. Each stage is implemented as an independent Python script.

---

## Raw Dataset Download

### Access & Training Requirements
Before downloading the datasets you must:
- Create a free account on PhysioNet and request access to the credentialed dataset.
- Complete an approved human-subjects research course (e.g., CITI “Data or Specimens Only Research”).
- Sign the required Data Use Agreement (DUA).

### Dataset Links & Download Commands

1. **MIMIC-III Clinical Database v1.4**  
   Link: https://physionet.org/content/mimiciii/1.4/

   Example download (AWS CLI strongly recommended):
   ```bash
   aws s3 sync --no-sign-request \
       s3://mimic-iii-physionet-1.4/ ./mimiciii_1.4/
   ```

2. **MIMIC-III Waveform Database Matched Subset v1.0**  
   Link: https://physionet.org/content/mimic3wdb-matched/1.0/

   Example download:
   ```bash
   aws s3 sync --no-sign-request \
       s3://mimic-iii-wdb-matched-1.0/ ./mimic3wdb_matched_1.0/
   ```

### Notes
- AWS CLI download is highly recommended due to the dataset size.
- Alternatively, PhysioNet's `wget` download script can be used if AWS CLI is not available.

---

## Pipeline Overview

Stage 0  → Build lab and blood-pressure vital wide tables  
Stage 1  → Map waveforms with labs and vitals and extract valid windows  
Stage 1B → Generate demographics and ICD-9 metadata  
Stage 2  → Extract waveform samples and resample them (40 / 120 / 500 Hz)  
Stage 3  → Align labs and vitals to waveform timeline, normalize, embed, and save final NPZ files  

---

# Stage 0 — Lab Table and Blood-Pressure Vital Table

Source files:  
./stp0_1_lab_filter.py  
./stp0_0_vital_filter.py

Purpose:  
These scripts extract lab values (Potassium, Calcium, Sodium, Glucose, Lactate, Creatinine) and arterial/non-invasive blood-pressure vitals from MIMIC-III.  
Both scripts convert long-format events into compact wide-format tables indexed by SUBJECT_ID, HADM_ID, and CHARTTIME.

Outputs:  
cmp_panel_wide.parquet  
vital_panel_wide.parquet

---

# Stage 1 — Waveform × Lab × Vital Temporal Mapping

Source file:  
./stp1_0_mapping_wav_lab_vital.py

Purpose:  
This script links waveform segments to labs and vitals through temporal matching. It performs the following steps:

1. Parse WFDB headers to obtain record_path, start_time, end_time, and sampling rate fs.  
2. Join waveform windows with lab and vital timestamps using Polars join_where conditions.  
3. Build padded extraction windows around all lab/vital timestamps (±6 hours).  
4. Convert these windows into WFDB sample indices.  
5. Apply quality filtering to keep only high-coverage waveform segments and remove conflicting or overlapping directories.

Outputs:  
waveform_records_info.csv  
mimic_lab_vital_waveform_overlap.csv  
mimic_lab_vital_waveform_history.csv  
mimic_waveform_segments_filtered.csv

---

# Stage 1B — Demographics and ICD-9 Metadata

Source file:  
./stp1_1_demographics.py

Purpose:  
This script builds admission-level demographic metadata (age, gender, ethnicity, insurance, language, marital status) and merges the primary ICD-9 diagnosis for each admission.  
It filters to only the SUBJECT_ID and HADM_ID values identified in Stage 1.

Output:  
mimic_patient_admission_demo_with_diag.csv

---

# Stage 2 — Waveform Extraction and Resampling

Source file:  
./stp2_wav_reader_extraction.py

Purpose:  
This script loads raw waveform samples from WFDB directories using the sample index ranges produced in Stage 1.  
It then performs:

1. Multi-record waveform concatenation into continuous time series  
2. Construction of per-sample millisecond timestamps  
3. Resampling  
   - PPG → 40 Hz  
   - ECG → 120 Hz  
   - ECG → 500 Hz  
4. 30-second segment slicing  
5. Saving encounter-level waveform NPZ files

Example output directory:  
MIMIC3_SPO2_I_40hz_v3

---

# Stage 3 — Lab and Vital Alignment, Normalization, and Embedding

Source file:  
./stp3_1_npz_prepare.py

Purpose:  
This script merges waveform NPZ files with normalized lab and vital targets, producing final UNIPHY-ready NPZ files.  
Key steps include:

1. Load waveform NPZ files and metadata  
2. Compute global quantile-based normalization statistics for labs and vitals  
3. Align lab and vital timestamps to waveform timeline using a 60-second matching window  
4. Build supervised EHR arrays  
   ehr_mask  (0 = none, 1 = interpolated, 2 = real)  
   ehr_gt  
   ehr_trend  
5. <span style="color:red;">Generation of GPT-19M waveform embeddings (might need waiting for the model can checkpoints share)</span>
6. Save final multi-modal NPZ files containing waveform, EHR targets, embeddings, and timestamps  
7. Save a raw lab/vital summary table

Output directory:  
MIMIC3_SPO2_I_40hz_v3_lab_vital  
