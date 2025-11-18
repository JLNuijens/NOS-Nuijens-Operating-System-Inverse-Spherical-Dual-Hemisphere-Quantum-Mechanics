# cic/index.py
# Minimal CIC-Lite index wrapper.
# Uses your encoder, your resonance scorer, and your memory store.

from typing import List, Tuple
import numpy as np

from encoder import encode_text
from resonance import resonance_score
from store import MemoryStore


class CICIndex:
    """
    CICIndex: A minimal DH1-native resonance index.

    Responsibilities:
        - Encode text into complex waveforms
        - Store waveforms
        - Search by resonance scoring
    """

    def __init__(self, N: int = 512, K: int = 16, lam: float = 0.5, mode: str = "embed"):
        self.N = N
        self.K = K
        self.lam = lam
        self.mode = mode   # "char" or "embed"
        self.store = MemoryStore()

    # -----------------------------
    # Adding data
    # -----------------------------
    def add_text(self, text: str):
        """
        Encode text → complex waveform and store it.
        """
        wave = encode_text(text, N=self.N, mode=self.mode)
        self.store.add(wave)

    def add_texts(self, texts: List[str]):
        """
        Add multiple documents.
        """
        for t in texts:
            self.add_text(t)

    # -----------------------------
    # Searching
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for the top-k matching items.

        Returns list of (index, score).
        """
        qwave = encode_text(query, N=self.N, mode=self.mode)

        scores = []
        for idx, mwave in enumerate(self.store.memory):
            score = resonance_score(qwave, mwave, K=self.K, lam=self.lam)
            scores.append((idx, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    # -----------------------------
    # Export / Import
    # -----------------------------
    def save(self, path: str):
        """
        Save memory store to a .npy file.
        """
        np.save(path, np.array(self.store.memory, dtype=complex))

    def load(self, path: str):
        """
        Load memory store from a .npy file.
        """
        arr = np.load(path, allow_pickle=True)
        for wave in arr:
            self.store.add(wave)
