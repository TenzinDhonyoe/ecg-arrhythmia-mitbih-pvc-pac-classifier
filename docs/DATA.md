# Data Documentation

## Training Dataset

This project trains on the MIT-BIH Arrhythmia Database (PhysioNet).

- Source: https://physionet.org/content/mitdb/1.0.0/
- Citation: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.

## Data Use Rules

- Do not commit raw MIT-BIH files to git.
- Users must download data directly from PhysioNet and comply with the dataset license/terms.

## Expected Directory

Training CLI expects a path containing `RECORDS` and WFDB record files:

```
/path/to/mit-bih-arrhythmia-database-1.0.0
```

## Labels

- `N` -> class `0`
- `V` -> class `1`
- `A` and `a` -> merged class `2` (PAC)
