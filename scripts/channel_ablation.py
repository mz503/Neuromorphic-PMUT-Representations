"""Run the paired three-fold channel comparison alongside the original pipeline."""
import os
from pathlib import Path
import sys

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pmut_models.channel_ablation import main

if __name__ == "__main__":
    main()
