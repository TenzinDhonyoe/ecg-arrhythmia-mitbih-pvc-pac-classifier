"""AAMI EC57 mapping + de Chazal DS1/DS2 split tests.

These tests pin the AAMI EC57 mapping against published references
(de Chazal et al. 2004 + AAMI EC57:2012) so a typo or stray edit can't
quietly drift the mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg_arrhythmia.labels import AAMI5
from ecg_arrhythmia.preprocessing import (
    DE_CHAZAL_DS1,
    DE_CHAZAL_DS2,
    PACED_RECORDS,
    DatasetSplit,
    split_de_chazal,
)

# (symbol, expected_aami_class_name)
EXPECTED_AAMI_MAPPING = [
    # N — normal + bundle-branch blocks + escape beats
    ("N", "N"),
    ("L", "N"),
    ("R", "N"),
    ("e", "N"),
    ("j", "N"),
    # S — supraventricular ectopic
    ("A", "S"),
    ("a", "S"),
    ("J", "S"),  # junctional/nodal premature
    ("S", "S"),  # supraventricular ectopic literal
    # V — ventricular
    ("V", "V"),
    ("E", "V"),  # ventricular escape
    # F — fusion of V and N
    ("F", "F"),
    # Q — unclassified / paced
    ("Q", "Q"),
    ("/", "Q"),  # paced
    ("f", "Q"),  # fusion of paced and N
    ("?", "Q"),
]


@pytest.mark.parametrize("symbol,expected_label", EXPECTED_AAMI_MAPPING)
def test_aami_symbol_to_label(symbol, expected_label):
    cls_id = AAMI5.label_to_id[symbol]
    assert AAMI5.id_to_label[cls_id] == expected_label


def test_aami_excludes_artifact_marker_pipe():
    """The '|' marker is intentionally excluded from AAMI5.

    Some annotation tools use '|' as an artifact marker rather than a beat
    label, so including it would pollute the Q class with non-beat events.
    See docs/MODEL_CARD.md.
    """
    assert "|" not in AAMI5.label_to_id


def test_aami_does_not_silently_promote_unknown_symbols():
    """Random non-MIT-BIH symbols don't accidentally land in some class."""
    for s in ("X", "Z", "!", "[", "]", "(", ")", "~", "+", "*"):
        assert s not in AAMI5.label_to_id


def test_ds1_ds2_disjoint_and_total_44():
    ds1 = set(DE_CHAZAL_DS1)
    ds2 = set(DE_CHAZAL_DS2)
    assert len(ds1) == 22
    assert len(ds2) == 22
    assert ds1.isdisjoint(ds2)
    assert len(ds1 | ds2) == 44


def test_ds1_ds2_excludes_paced_records():
    """The canonical de Chazal split intentionally drops the 4 paced records."""
    paced = set(PACED_RECORDS)
    assert paced.isdisjoint(set(DE_CHAZAL_DS1))
    assert paced.isdisjoint(set(DE_CHAZAL_DS2))


def test_split_de_chazal_basic():
    record_ids = np.array(list(DE_CHAZAL_DS1) + list(DE_CHAZAL_DS2))
    split = split_de_chazal(record_ids, val_records=4, seed=42)
    assert isinstance(split, DatasetSplit)
    train = set(split.train_records)
    val = set(split.val_records)
    test = set(split.test_records)
    # No leakage between any pair.
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Test set is DS2 unchanged.
    assert test == set(DE_CHAZAL_DS2)
    # Validation comes from DS1; train is DS1 minus val.
    assert val.issubset(set(DE_CHAZAL_DS1))
    assert train == set(DE_CHAZAL_DS1) - val
    assert len(val) == 4


def test_split_de_chazal_excludes_paced_when_requested():
    record_ids = np.array(
        list(DE_CHAZAL_DS1) + list(DE_CHAZAL_DS2) + list(PACED_RECORDS)
    )
    split = split_de_chazal(record_ids, val_records=4, seed=42, exclude_paced=True)
    paced = set(PACED_RECORDS)
    for bucket in (split.train_records, split.val_records, split.test_records):
        assert paced.isdisjoint(set(bucket))


def test_split_de_chazal_errors_when_records_missing():
    record_ids = np.array(["100"])  # only DS2 record present
    with pytest.raises(ValueError, match="DS1"):
        split_de_chazal(record_ids)


def test_split_de_chazal_deterministic():
    record_ids = np.array(list(DE_CHAZAL_DS1) + list(DE_CHAZAL_DS2))
    a = split_de_chazal(record_ids, val_records=3, seed=7)
    b = split_de_chazal(record_ids, val_records=3, seed=7)
    assert a.train_records == b.train_records
    assert a.val_records == b.val_records
    assert a.test_records == b.test_records
