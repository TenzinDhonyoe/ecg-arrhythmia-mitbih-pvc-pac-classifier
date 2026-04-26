# Data Documentation

## Training dataset

This project trains on the **MIT-BIH Arrhythmia Database** (PhysioNet, v1.0.0).

- Source: <https://physionet.org/content/mitdb/1.0.0/>
- Citation: Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database."
  *IEEE Engineering in Medicine and Biology Magazine* 20(3):45–50, 2001.

## Composition (after our preprocessing pipeline)

- 48 records × 30 minutes, sampled at 360 Hz.
- Each record has two simultaneous channels; the shipped models use **MLII**
  (channel 0 in most records). The second channel is V1 / V5 depending on
  the record.
- Beat-level annotations in `*.atr` files. We retain three target symbols
  (`N`, `V`, `A`/`a`) and drop the rest. After 180-sample window extraction:

| Split   | Records | Beats   | N       | V (PVC) | a (PAC) |
|---------|---------|---------|---------|---------|---------|
| Train   | 32      | 56 507  | 48 389  | 5 710   | 2 408   |
| Val     |  6      | 11 716  | 11 230  | 309     | 177     |
| Test    |  8      | 16 367  | 15 152  | 1 104   | 111     |

PAC is 2.4 % of training beats and 0.7 % of test beats. This imbalance is
the dominant challenge for both LR and ResNet. The exact record-level split
is saved to `artifacts/baseline/record_split.json` (seed 42).

## How to fetch

```bash
python scripts/fetch_mitbih.py --dest data/mitdb
```

This wraps `wfdb.dl_database("mitdb", dl_dir=...)` and reconstructs the
`RECORDS` index from the downloaded `.hea` files when needed (recent PhysioNet
mirrors don't always pull `RECORDS` itself). One-time, ~100 MB,
~5–15 minutes.

## Label mapping

| MIT-BIH symbol | Class id | Class name |
| -------------- | -------- | ---------- |
| `N`            | 0        | Normal     |
| `V`            | 1        | PVC        |
| `A`            | 2        | PAC        |
| `a`            | 2        | PAC (merged with `A`) |

All other MIT-BIH symbols (e.g. `L`, `R`, `e`, `j`, `/`, `f`, `Q`) are
dropped during segmentation. We do **not** target AAMI EC57 superclasses
(N / S / V / F / Q) — see `docs/MODEL_CARD.md` § "Future work".

## Data use rules

- **Do not commit raw MIT-BIH files to git.** They are ignored by
  `.gitignore`.
- Users must comply with the [PhysioNet license](https://physionet.org/about/licenses/)
  and the dataset's own citation requirements.
- Do not commit personal ECG data without explicit provenance and
  consent. The example wearable CSV (`examples/wearable_lead_i_synthetic.csv`)
  is generated from MIT-BIH and contains no patient PII.

## Expected directory layout

```
data/mitdb/
  ├─ RECORDS              # one record stem per line (e.g. 100, 101, ...)
  ├─ 100.hea, 100.dat, 100.atr
  ├─ 101.hea, 101.dat, 101.atr
  ...
```

`ecg-train --data-dir data/mitdb` finds records via the `RECORDS` file. If
you bring your own WFDB dataset, just emit a `RECORDS` index and the same
WFDB sidecar files (`*.hea`, `*.dat`, `*.atr`) and it will work — at your
own scientific risk, since the model card numbers are MIT-BIH-specific.
