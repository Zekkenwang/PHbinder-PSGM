import torch

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42 

    TRAIN_DATA_PATH = "data/raw/hebing.CSV"
    HLA_DB_PATH = "data/raw/pseudo sequence.CSV"
    PROCESSED_TRAIN_DATA_PATH = "data/processed/train.csv"
    PROCESSED_VAL_DATA_PATH = "data/processed/val.csv"
    PROCESSED_TEST_DATA_PATH = "data/processed/test.csv"
    MODEL_SAVE_PATH = "models/psgm/hla_generator.pth"
    RESULTS_SAVE_PATH = "models/psgm/results/generation_results_GAN.csv" 
    EXTERNAL_DATA_PATH_PSGM = "data/external/psgm/external_data.CSV"
    PREDICTION_RESULTS_PATH_PSGM = "data/external/psgm/output.CSV"
    PEPTIDE_MAX_LEN = 14
    HLA_SEQ_LEN = 34

    ESM2_DIM = 640
    EMBED_DIM = 256
    GENERATOR_DEC_LAYERS = 6 
    GENERATOR_NHEAD = 8
    DISCRIMINATOR_EMBED_DIM = 256 
    DISCRIMINATOR_NUM_LAYERS = 3 
    DISCRIMINATOR_NHEAD = 8 

    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-7
    LAMBDA_ADV = 0.3
    PATIENCE = 3
    MIN_DELTA = 0.001
    VAL_BATCH_SIZE_MULTIPLIER = 2
    GENERATION_TOP_P = 0.9
    GENERATION_TEMPERATURE = 1.0
    NN_NUM_NEIGHBORS = 50
    TOP_MATCHES_TO_RETURN = 50
    PAD_TOKEN = '[PAD]'
    MASK_TOKEN = '[MASK]'
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
