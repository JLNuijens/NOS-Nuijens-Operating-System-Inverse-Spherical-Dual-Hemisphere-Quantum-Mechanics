# cic/encoder.py
# Thin wrapper around your existing encoders.
# Does NOT modify any of your original code.

import numpy as np

# Import your working encoders exactly as-is
from char_wave import char_to_wave
from embed_wave import embed_to_wave


def encode_text(text: str, N: int = 512, mode: str = "embed"):
    """
    Convert text into a complex waveform.

    Parameters:
        text : str
            Input text.
        N : int
            Waveform length.
        mode : "char" or "embed"
            Selects which encoder to use.

    Returns:
        np.ndarray : complex waveform for CIC resonance scoring.
    """
    if mode == "char":
        return char_to_wave(text, N=N)
    elif mode == "embed":
        return embed_to_wave(text, N=N)
    else:
        raise ValueError(f"Unknown encoder mode '{mode}'. Use 'char' or 'embed'.")


def make_encoder(mode: str = "embed", N: int = 512):
    """
    Factory that returns a callable encoder function with preset parameters.

    Usage:
        encoder = make_encoder("embed", 512)
        wave = encoder("hello world")
    """
    def _encode(text: str):
        return encode_text(text, N=N, mode=mode)

    return _encode

