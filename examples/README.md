# Example ECG inputs

These CSVs are bundled with the repository so the inference CLI works out of
the box. **No patient PII** — both files are derived from a single public
MIT-BIH record under PhysioNet's open-data terms.

## `sample_ecg.csv`

- 20 seconds of MIT-BIH record `100`, channel 0 (MLII), 360 Hz.
- Used by `make smoke` and the README quick-start.
- Regenerate with:

  ```python
  import os, csv, wfdb
  os.chdir("data/mitdb"); rec = wfdb.rdrecord("100")
  sig = rec.p_signal[:int(20*rec.fs), 0]
  ```

## `wearable_lead_i_synthetic.csv`

- 30 seconds derived from MIT-BIH record `100`, channel 1 (V5), low-pass
  filtered to 40 Hz, downsampled to 250 Hz, mild Gaussian noise added — a
  rough simulation of a consumer Lead-I wearable trace.
- Regenerate with `python scripts/make_wearable_demo.py`.

## Citation

If you republish these files, cite both this repository and the MIT-BIH
Arrhythmia Database (PhysioNet). See `CITATION.cff`.
