import numpy as np

from ecg_arrhythmia.preprocessing import compute_rr_features, split_by_record


def test_compute_rr_features_shape():
    peaks = np.array([100, 200, 310, 420], dtype=int)
    rr = compute_rr_features(peaks)
    assert rr.shape == (4, 2)
    assert np.all(np.isfinite(rr))


def test_split_has_no_overlap():
    record_ids = np.array(["100", "100", "101", "102", "103", "104"])
    split = split_by_record(record_ids, train_ratio=0.5, val_ratio=0.25, seed=1)
    train = set(split.train_records)
    val = set(split.val_records)
    test = set(split.test_records)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
