"""Stage 1 lockdown: label scheme refactor must be back-compat for 3-class users.

Tests in this module pin two things:

1. The legacy module-level aliases ``LABEL_TO_ID``, ``ID_TO_LABEL``,
   ``TARGET_SYMBOLS`` still exist and still describe MITBIH3.
2. ``BeatPrediction.to_dict`` for a 3-class result still emits the literal
   keys ``prob_N``/``prob_V``/``prob_a`` so existing CSV/JSON consumers don't
   break.
"""

from __future__ import annotations

import numpy as np

from ecg_arrhythmia import ECGClassifier
from ecg_arrhythmia.api import BeatPrediction, PredictionResult
from ecg_arrhythmia.labels import (
    AAMI5,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MITBIH3,
    TARGET_SYMBOLS,
    LabelScheme,
    get_scheme,
)
from ecg_arrhythmia.training import train_baseline_quick_smoke


def test_module_aliases_point_at_mitbih3():
    assert MITBIH3.label_to_id == LABEL_TO_ID
    assert MITBIH3.id_to_label == ID_TO_LABEL
    assert MITBIH3.target_symbols == TARGET_SYMBOLS
    assert MITBIH3.n_classes == 3
    assert MITBIH3.labels == ["N", "V", "a"]


def test_aami5_basic_shape():
    assert AAMI5.n_classes == 5
    assert AAMI5.labels == ["N", "S", "V", "F", "Q"]
    # Every MIT-BIH WFDB symbol that maps under AAMI5 must point to a valid id
    for symbol, id_ in AAMI5.label_to_id.items():
        assert 0 <= id_ < 5
        assert AAMI5.id_to_label[id_] in AAMI5.labels


def test_get_scheme_lookup():
    assert get_scheme("mitbih3") is MITBIH3
    assert get_scheme("aami5") is AAMI5
    try:
        get_scheme("nope")
    except ValueError as e:
        assert "Unknown" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown scheme")


def test_beat_prediction_to_dict_back_compat_3class():
    bp = BeatPrediction(
        beat_index=1,
        peak_sample=100,
        peak_time_s=0.27,
        label="N",
        confidence=0.9,
        probabilities={"N": 0.9, "V": 0.07, "a": 0.03},
    )
    d = bp.to_dict()
    # Legacy keys must still be present.
    assert d["prob_N"] == 0.9
    assert d["prob_V"] == 0.07
    assert d["prob_a"] == 0.03
    assert d["label"] == "N"


def test_beat_prediction_to_dict_5class():
    bp = BeatPrediction(
        beat_index=2,
        peak_sample=200,
        peak_time_s=0.55,
        label="S",
        confidence=0.6,
        probabilities={"N": 0.2, "S": 0.6, "V": 0.1, "F": 0.05, "Q": 0.05},
    )
    d = bp.to_dict()
    assert d["prob_N"] == 0.2
    assert d["prob_S"] == 0.6
    assert d["prob_V"] == 0.1
    assert d["prob_F"] == 0.05
    assert d["prob_Q"] == 0.05


def test_prediction_result_dynamic_class_counts():
    beats = [
        BeatPrediction(1, 0, 0.0, "N", 0.9, {"N": 0.9, "S": 0.05, "V": 0.05, "F": 0.0, "Q": 0.0}),
        BeatPrediction(2, 100, 0.3, "V", 0.7, {"N": 0.2, "S": 0.05, "V": 0.7, "F": 0.05, "Q": 0.0}),
    ]
    res = PredictionResult(beats=beats)
    counts = res.class_counts()
    # All 5 classes must appear, with correct counts.
    assert counts == {"N": 1, "S": 0, "V": 1, "F": 0, "Q": 0}
    # probabilities matrix has 5 columns, summing to ~1 per row.
    probs = res.probabilities
    assert probs.shape == (2, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_smoke_classifier_carries_mitbih3_scheme(tmp_path):
    """A baseline trained on the (3-class) smoke dataset reports MITBIH3."""
    artifacts = train_baseline_quick_smoke(tmp_path, seed=0)
    clf = ECGClassifier.load_baseline(artifacts.model_path, artifacts.scaler_path)
    assert clf.scheme.name == "mitbih3"
    # And its loaded scheme is the same singleton.
    assert clf.scheme is MITBIH3


def test_label_scheme_dataclass_is_frozen():
    """LabelScheme is a frozen dataclass; mutating it should raise."""
    try:
        MITBIH3.name = "evil"  # type: ignore[misc]
    except (AttributeError, Exception) as exc:
        # FrozenInstanceError is a subclass of AttributeError on 3.11+
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)
    else:
        raise AssertionError("expected FrozenInstanceError")


def test_class_weights_unhardcoded():
    from ecg_arrhythmia.preprocessing import class_weights

    y = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4])
    cw = class_weights(y, n_classes=5)
    # Every class id must have an entry, even those with zero or one sample.
    assert set(cw.keys()) == {0, 1, 2, 3, 4}
    # Larger class should have smaller weight.
    assert cw[0] < cw[2]
    # Two classes with the same support get the same weight.
    assert cw[1] == cw[3]


__all__: list[str] = []  # tests don't export anything

# Keep the LabelScheme symbol referenced so unused-import linters stay quiet.
_ = LabelScheme
