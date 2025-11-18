# cic/store.py
# Minimal in-memory waveform store for CIC-Lite.

import numpy as np


class MemoryStore:
    """
    MemoryStore: holds complex waveforms for CICIndex.

    - Stores waveforms in a Python list
    - Ensures dtype = complex64 / complex128 as needed
    - Provides simple add() and indexing
    """

    def __init__(self):
        self.memory = []  # list of np.ndarray waveforms

    def add(self, wave: np.ndarray):
        """
        Add a complex waveform to memory.

        Parameters:
            wave : np.ndarray
                Complex-typed waveform from encoder.
        """
        # Ensure it's stored as complex for FFT scoring
        self.memory.append(wave.astype(np.complex64))

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]
