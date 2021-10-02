"""
Constants for testing

Constants
---------
REL : float
    almost equal resolution
"""
from pathlib import Path


"tolerance for floating point comparisons"
REL = 1e-4

"database path"
DATA_PATH = Path(__file__).parent.parent / "data"
