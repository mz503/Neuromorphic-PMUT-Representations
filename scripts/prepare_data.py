from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pmut_models.reproduction import data_stage
if __name__=='__main__': data_stage()
