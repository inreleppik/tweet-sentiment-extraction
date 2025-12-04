SEED = 42

MODEL_NAME = "roberta-base"
MAX_LEN = 96

BATCH_SIZE = 128
NUM_EPOCHS = 3
N_SPLITS = 10
LR = 3e-5

TRAIN_CSV = "data/train_data.csv"
TEST_CSV = "data/test_data.csv"  # поправь путь под себя

OUTPUT_DIR = "checkpoints"