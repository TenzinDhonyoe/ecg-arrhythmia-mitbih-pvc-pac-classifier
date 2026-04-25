import numpy as np

from ecg_arrhythmia.inference import resample_signal, windows_from_peaks


def test_resample_signal_changes_length():
    sig = np.sin(np.linspace(0, 10, 100))
    out = resample_signal(sig, input_fs=250, output_fs=360)
    assert len(out) != len(sig)
    assert np.all(np.isfinite(out))


def test_windows_from_peaks_extracts_fixed_length():
    sig = np.random.randn(1000)
    peaks = np.array([100, 300, 500, 900])
    windows, valid = windows_from_peaks(sig, peaks)
    assert windows.shape[1] == 180
    assert len(windows) == len(valid)
