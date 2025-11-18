CIC-Lite (Cromelin Information Compiler – Lite)

CIC-Lite is a minimal implementation of the DH1 complex-wave retrieval substrate. It converts text into complex waveforms and ranks stored entries by measuring resonance in the Fourier domain. Retrieval is completely deterministic and requires no training or gradient-based optimization. The package is intended as a lightweight, transparent replacement for vector-search libraries such as FAISS in small or medium-scale retrieval pipelines.

CIC-Lite exposes a simple encoder, a resonance scoring function, a memory store, and a high-level index interface. All operations rely only on NumPy and standard Python.

Installation

Clone the repository and ensure NumPy is installed:

git clone https://github.com/JLNuijens/NOS-Nuijens-Operating-System-Inverse-Spherical-Dual-Hemisphere-Quantum-Mechanics
pip install numpy


CIC-Lite can be imported once the package folder is on your Python path.

Example
from cic import CICIndex

index = CICIndex(N=128, K=16)

index.add_text("hello world")
index.add_text("complex wave resonance retrieval")

results = index.search("hello")
print(results)


This example creates an index, stores two entries, and performs a query. The ranking is based on resonance between the query waveform and stored waveforms in the frequency domain.

Package Structure

The CIC-Lite package includes the following modules:

__init__.py
Defines the public API for external use and provides clean imports for the index, encoder, and resonance functions.

encoder.py
Implements the character-wave encoder and the factory for constructing encoders. Converts raw text into complex-valued time-domain waveforms.

resonance.py
Implements the FFT-based resonance scoring function. This computes magnitude agreement and phase alignment over the top-K frequency bins of the query.

store.py
Defines a simple in-memory structure for storing waveforms, retrieving them by index, and associating them with original text entries.

index.py
Provides a high-level interface that ties together encoding, storage, and resonance scoring. This is the recommended entry point for most usage.

utils.py
Contains small mathematical utilities such as waveform normalization and phase-difference evaluation.
