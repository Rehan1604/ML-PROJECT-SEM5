from pathlib import Path

from src.datasets.statistics import print_dataset_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "GTSRB"

TRAIN_CSV = DATA_DIR / "Train.csv"


print_dataset_statistics(TRAIN_CSV)