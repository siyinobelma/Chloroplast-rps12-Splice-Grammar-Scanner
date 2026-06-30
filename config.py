import torch

class Config:
    # data
    RAW_DATA = "data/raw_pairs.csv"
    PROCESSED_DIR = "data/processed"
    TRAIN_FILE = "data/processed/train.pt"
    VAL_FILE = "data/processed/val.pt"
    TEST_FILE = "data/processed/test.pt"
    MAX_SEQ_LEN = 500
    NEGATIVE_METHODS = ["swap_exon1", "swap_exon2", "reverse_intron", "shuffle_intron"]

    # model
    EMBED_DIM = 64
    NUM_FILTERS = 128
    KERNEL_SIZE = 7
    DROPOUT = 0.3

    # training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

import os
os.makedirs(Config.PROCESSED_DIR, exist_ok=True)
