# cic/utils.py
# Shared utility helpers for CIC-Lite.
# These do NOT change your core pipeline logic.

import numpy as np
import time


# -----------------------------
# Timing helper (optional)
# -----------------------------
class Timer:
    """
    Simple timing context manager.

    Usage:
        with Timer() as t:
            ... code ...
        print(t.ms)
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.ms = (self.end - self.start) * 1000.0


# -----------------------------
# Normalization helpers
# -----------------------------
def safe_norm(x: np.ndarray) -> np.ndarray:
    """
    Normalize a complex waveform safely.
    """
    mag = np.linalg.norm(x)
    if mag < 1e-12:
        return x
    return x / mag


def max_normalize(x: np.ndarray) -> np.ndarray:
    """
    Max-normalize a magnitude vector safely.
    """
    m = x.max()
    if m < 1e-12:
        return x
    return x / m


# -----------------------------
# FFT helpers
# -----------------------------
def fft_wave(x: np.ndarray) -> np.ndarray:
    """
    Wrapper for FFT. Keeps the codebase future-flexible.
    """
    return np.fft.fft(x)


def ifft_wave(X: np.ndarray) -> np.ndarray:
    """
    Inverse FFT wrapper.
    """
    return np.fft.ifft(X)


# -----------------------------
# Misc utility
# -----------------------------
def cosine_phase(a: float, b: float) -> float:
    """
    Compute cos(a - b) safely.
    """
    return float(np.cos(a - b))
