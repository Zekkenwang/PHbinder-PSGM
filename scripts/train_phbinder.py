# scripts/train_phbinder.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import configurations
from config import phbinder_config as config

# Import modules from src/phbinder
from src.phbinder.model import This_work
from src.phbinder.dataset import pad_inner_lists_to_length, addbatch
from src.phbinder.utils import set_seed, training, test_loader_eval
from src.phbinder.lora_finetune import setup_lora_model, finetune_lora_model

# Import AutoTokenizer directly here for global access during data loading
from transformers import AutoTokenizer

def main():
    """
    Main function to run the PHbinder model training and evaluation.
    """
    # --- Configuration and Initialization ---
    device = config.DEVICE
    set_seed(config.SEED)

    print(f"Using device: {device}")
    print(f"Setting random seed to: {config.SEED}")

    # Initialize tokenizer once for all data processing
    # Note: ensure config.LOCAL_ESM_MODEL_PATH points to your downloaded ESM-2 model directory
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.LOCAL_ESM_MODEL_PATH,
            local_files_only=True,
            clean_up_tokenization_spaces=False
        )
        print(f"Tokenizer loaded from: {config.LOCAL_ESM_MODEL_PATH}")
    except OSError as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure the ESM-2 model is downloaded and located at: {config.LOCAL_ESM_MODEL_PATH}")
        print("You might need to download it manually or adjust the path in config/phbinder_config.py.")
        return

    # --- Data Loading and Preprocessing ---
    print("\nLoading and preprocessing data...")
    try:
        train_df = pd.read_csv(config.TRAIN_DATA_PATH, header=0, sep=',')
        val_df = pd.read_csv(config.VALIDATION_DATA_PATH, header=0, sep=',')
        test_df = pd.read_csv(config.TEST_DATA_PATH, header=0, sep=',')
        print("Data CSVs loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please check if the data paths in config/phbinder_config.py are correct and files exist.")
        return

    # Extract epitopes and labels
    x_train_epitope = train_df['Epitope'].str.upper()
    x_val_epitope = val_df['Epitope'].str.upper()
    x_test_epitope = test_df['Epitope'].str.upper()

    train_label = torch.tensor(np.array(train_df['Label'], dtype='int64'))
    val_label = torch.tensor(np.array(val_df['Label'], dtype='int64'))
    test_label = torch.tensor(np.array(test_df['Label'], dtype='int64')) # For final evaluation

    # Tokenize and pad epitope sequences
    print("Tokenizing and padding epitope sequences...")
    x_train_encoding = tokenizer(x_train_epitope.tolist())['input_ids']
    x_train_encoded_padded = torch.tensor(
        pad_inner_lists_to_length(x_train_encoding, config.EPITOPE_MAX_LEN),
        dtype=torch.long
    )

    x_val_encoding = tokenizer(x_val_epitope.tolist())['input_ids']
    x_val_encoded_padded = torch.tensor(
        pad_inner_lists_to_length(x_val_encoding, config.EPITOPE_MAX_LEN),
        dtype=torch.long
    )

    x_test_encoding = tokenizer(x_test_epitope.tolist())['input_ids']
    x_test_encoded_padded = torch.tensor(
        pad_inner_lists_to_length(x_test_encoding, config.EPITOPE_MAX_LEN),
        dtype=torch.long
    )
    print("Data preprocessing complete.")

    # --- LoRA Fine-tuning ---
    print("\n--- Starting LoRA Fine-tuning ---")
    lora_model = setup_lora_model()
    lora_model.to(device)

    # Create DataLoaders for LoRA fine-tuning
    lora_train_dataloader = addbatch(x_train_encoded_padded, train_label, batchsize=config.LORA_BATCH_SIZE)
    lora_val_dataloader = addbatch(x_val_encoded_padded, val_label, batchsize=config.LORA_BATCH_SIZE)

    # Initialize the separate classifier for LoRA fine-tuning phase
    # This classifier takes the CLS token embedding from ESM and outputs class logits
    hidden_size = lora_model.config.hidden_size # ESM hidden size
    lora_classifier = nn.Linear(hidden_size, config.NUM_CLASSES).to(device)

    # Perform LoRA fine-tuning
    # The `save_path` for LoRA weights is handled internally by finetune_lora_model
    lora_weights_saved_path = finetune_lora_model(
        lora_model,
        lora_classifier, # Pass the classifier to be optimized
        lora_train_dataloader,
        lora_val_dataloader,
        device
    )
    print(f"LoRA fine-tuning finished. Weights saved to: {lora_weights_saved_path}")

    # --- Main PHbinder Model Training ---
    print("\n--- Starting Main PHbinder Model Training ---")
    
    # Initialize This_work model with the path to the saved LoRA weights
    # This_work will load the LoRA weights internally.
    model = This_work(lora_weights_saved_path)
    model.to(device)

    # Define optimizer and loss criterion for the main model
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.MAIN_MODEL_LEARNING_RATE,
        weight_decay=config.MAIN_MODEL_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    # Create DataLoader for main training
    main_train_dataloader = addbatch(x_train_encoded_padded, train_label, batchsize=config.MAIN_MODEL_BATCH_SIZE)

    # Start main model training
    trained_model = training(
        model,
        device,
        config.MAIN_MODEL_NUM_EPOCHS,
        criterion,
        optimizer,
        main_train_dataloader,
        x_val_encoded_padded,
        val_label,
        x_test_encoded_padded,
        test_label,
        patience=config.MAIN_MODEL_PATIENCE
    )
    print("Main PHbinder model training finished.")

    # --- Final Evaluation ---
    print("\n--- Performing Final Evaluation on Test Set ---")
    
    # Load the best saved model state for final evaluation
    # Ensure the path matches where `training` function saved the best model
    best_model_path = f"{config.SAVE_PATH_MAIN_MODEL_CHECKPOINTS}/best_model_I.pt"
    try:
        trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")
    except FileNotFoundError as e:
        print(f"Could not load best model for final evaluation: {e}")
        print(f"Ensure that '{best_model_path}' exists after training.")
        return

    trained_model.eval() # Set model to evaluation mode

    # Evaluate on the test set
    acc, mcc, f1, recall, precision, auc_score, _, _ = test_loader_eval(
        x_test_encoded_padded,
        test_label,
        config.MAIN_MODEL_BATCH_SIZE,
        device,
        trained_model
    )

    print("\n--- Final Test Set Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"AUC:       {auc_score:.4f}")
    print("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main()
