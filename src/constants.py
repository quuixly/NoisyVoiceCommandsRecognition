from pathlib import Path

N_MFCC = 40
MAX_PAD_LEN = 128
PRE_EMPHASIS = 0.97
TOP_DB = 30

DATA_DIR = Path("data")
MFCC_DIR = Path("mfcc")
AUG_DIR = Path("augmented")
FEATURES_DIR = Path("features")
MANIFEST_PATH = Path("manifest.csv")
LABELS_PATH = Path("labels.json")
CHECKPOINTS_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")

COMMANDS_TO_PROCESS = ["up", "down", "left", "right", "stop", "go"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42
