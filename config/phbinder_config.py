# config/phbinder_config.py

import torch

# --- General Configuration ---
# Device to run the models on. Automatically detects CUDA if available.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Random seed for reproducibility.
SEED = 42

# --- Path Configurations ---
# Local path to the ESM-2 (esm2_t30_150M_UR50D) model.
# Users should download this model and place it in the 'models/' directory.
# Example: models/esm2_t30_150M_UR50D/
LOCAL_ESM_MODEL_PATH = "models/esm2_t30_150M_UR50D"

# Path to save the LoRA fine-tuned weights.
SAVE_PATH_LORA_WEIGHTS = "models/phbinder_lora_weights"

# Path to save the main PHbinder model checkpoints.
SAVE_PATH_MAIN_MODEL_CHECKPOINTS = "models/phbinder_checkpoints"

# Paths to raw data files.
TRAIN_DATA_PATH = "data/raw/HLA_I_epitope_train_shuffle.csv"
VALIDATION_DATA_PATH = "data/raw/HLA_I_epitope_validation.csv"
TEST_DATA_PATH = "data/raw/HLA_I_epitope_test.csv"

# --- ESM Tokenizer and Data Preprocessing Parameters ---
# Target sequence length for padding epitope sequences.
EPITOPE_MAX_LEN = 16

# --- LoRA Fine-tuning Parameters ---
# Target modules for LoRA application in the ESM model.
# As per your `setup_lora_model` function.
LORA_TARGET_MODULES = ['query', 'key', 'value', 'out_proj']
# LoRA attention dimension.
LORA_R = 16
# LoRA alpha parameter.
LORA_ALPHA = 32
# Dropout probability for LoRA layers.
LORA_DROPOUT = 0.1
# Task type for PEFT configuration.
LORA_TASK_TYPE = "FEATURE_EXTRACTION"

# Learning rate for LoRA fine-tuning.
LORA_LEARNING_RATE = 1e-5
# Number of training epochs for LoRA fine-tuning.
LORA_NUM_EPOCHS = 10
# Patience for early stopping during LoRA fine-tuning.
LORA_PATIENCE = 5
# Batch size for LoRA fine-tuning.
LORA_BATCH_SIZE = 32

# --- Main PHbinder Model Parameters (This_work class) ---
# Number of Transformer encoder layers.
TRANSFORMER_N_LAYERS = 6
# Number of attention heads in Transformer.
TRANSFORMER_N_HEAD = 16
# Dimension of the model embeddings in Transformer.
TRANSFORMER_D_MODEL = 640
# Dimension of the feed-forward network in Transformer.
TRANSFORMER_D_FF = 64
# Number of channels for CNN layers.
CNN_NUM_CHANNEL = 256
# Kernel size for region embedding CNN.
CNN_REGION_EMBEDDING_SIZE = 3
# Kernel size for main CNN blocks.
CNN_KERNEL_SIZE = 3
# Padding size for main CNN blocks.
CNN_PADDING_SIZE = 1
# Stride for main CNN blocks.
CNN_STRIDE = 1
# Pooling size for max pooling layers in CNN blocks.
CNN_POOLING_SIZE = 2
# Dropout rate for Transformer layers.
TRANSFORMER_DROPOUT = 0.2
# Hidden size for the first layer of the final classification FC task.
FC_TASK_HIDDEN_SIZE_1 = TRANSFORMER_D_MODEL // 4 # Derived from d_model / 4
# Hidden size for the second layer of the final classification FC task.
FC_TASK_HIDDEN_SIZE_2 = 64
# Dropout rate for the final classification FC task.
FC_TASK_DROPOUT = 0.3
# Number of output classes for the classifier (e.g., 2 for bind/non-bind).
NUM_CLASSES = 2

# --- Main PHbinder Training Parameters ---
# Learning rate for the main model training.
MAIN_MODEL_LEARNING_RATE = 5e-6
# Weight decay for the main model optimizer.
MAIN_MODEL_WEIGHT_DECAY = 0.0025
# Number of training epochs for the main model. Set high, early stopping will manage.
MAIN_MODEL_NUM_EPOCHS = 1000
# Patience for early stopping during main model training.
MAIN_MODEL_PATIENCE = 5
# Batch size for main model training and evaluation.
MAIN_MODEL_BATCH_SIZE = 64

# --- Loss Function Parameter ---
# Constant offset used in the custom loss calculation (loss - 0.04).abs() + 0.04
LOSS_OFFSET = 0.04
