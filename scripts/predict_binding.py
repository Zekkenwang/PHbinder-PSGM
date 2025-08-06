# scripts/predict_binding.py

import torch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef

# Import configurations
from config import phbinder_config as config

# Import PHbinder modules
from src.phbinder.model import This_work
from src.phbinder.dataset import pad_inner_lists_to_length, addbatch
from src.phbinder.utils import set_seed # Re-use set_seed from phbinder.utils
# test_loader_eval is for metrics, we'll implement a prediction-only loop
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Predict peptide-HLA binding using the PHbinder model.")
    parser.add_argument("--input_data_file", type=str, required=True,
                        help="Path to a CSV file containing epitopes. Expected column: 'Epitope'. "
                             "Optional column: 'Label' for evaluation.")
    parser.add_argument("--output_predictions_file", type=str, required=True,
                        help="Path to save the predictions and probabilities to a CSV file.")
    parser.add_argument("--main_model_path", type=str, 
                        default=os.path.join(config.SAVE_PATH_MAIN_MODEL_CHECKPOINTS, "best_model_I.pt"),
                        help="Path to the trained PHbinder main model checkpoint.")
    parser.add_argument("--lora_weights_path", type=str, 
                        default=config.SAVE_PATH_LORA_WEIGHTS,
                        help="Path to the LoRA fine-tuning weights for the ESM backbone.")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Device to use for prediction (e.g., 'cuda' or 'cpu').")
    
    args = parser.parse_args()

    set_seed(config.SEED) # Ensure reproducibility for any random operations (though less critical for inference)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Initialize Tokenizer ---
    print(f"\nInitializing tokenizer from: {config.LOCAL_ESM_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.LOCAL_ESM_MODEL_PATH,
            local_files_only=True,
            clean_up_tokenization_spaces=False
        )
        print("Tokenizer initialized successfully.")
    except OSError as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure the ESM-2 model is downloaded and located at: {config.LOCAL_ESM_MODEL_PATH}")
        return

    # --- Load Input Data ---
    print(f"Loading input data from: {args.input_data_file}")
    try:
        input_df = pd.read_csv(args.input_data_file, header=0, sep=',')
        if 'Epitope' not in input_df.columns:
            raise ValueError("Input CSV must contain an 'Epitope' column.")
        
        epitopes = input_df['Epitope'].str.upper().tolist()
        has_labels = 'Label' in input_df.columns
        if has_labels:
            true_labels = torch.tensor(np.array(input_df['Label'], dtype='int64'))
            print("Labels found in input data for evaluation.")
        else:
            true_labels = None
            print("No 'Label' column found. Performing prediction only.")

        print(f"Loaded {len(epitopes)} epitopes.")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {args.input_data_file}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading input data: {e}")
        return

    # --- Tokenize and Pad Epitopes ---
    print("Tokenizing and padding epitope sequences...")
    epitope_encodings = tokenizer(epitopes)['input_ids']
    epitope_encoded_padded = torch.tensor(
        pad_inner_lists_to_length(epitope_encodings, config.EPITOPE_MAX_LEN),
        dtype=torch.long
    )
    print("Epitope preprocessing complete.")

    # --- Prepare DataLoader ---
    # Create a dummy label tensor if no labels are provided, as addbatch expects two tensors
    dummy_labels = torch.zeros(epitope_encoded_padded.size(0), dtype=torch.long)
    predict_dataloader = addbatch(epitope_encoded_padded, dummy_labels, batchsize=config.MAIN_MODEL_BATCH_SIZE)
    
    # --- Load Model ---
    print(f"\nLoading PHbinder model from: {args.main_model_path}")
    print(f"Using LoRA weights from: {args.lora_weights_path}")
    
    model = This_work(args.lora_weights_path) # This_work expects lora_weights_path
    model.to(device)
    try:
        model.load_state_dict(torch.load(args.main_model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        print("PHbinder model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Main model checkpoint not found at {args.main_model_path}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Perform Prediction ---
    print("\nStarting prediction...")
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for batch_epitopes, _ in tqdm(predict_dataloader, desc="Predicting binding"):
            batch_epitopes = batch_epitopes.to(device)
            
            # Forward pass
            outputs = model(batch_epitopes) # [batch_size, num_classes]
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy() # [batch_size, num_classes]
            
            # Get predicted class (0 or 1)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy() # [batch_size]

            all_probabilities.extend(probabilities[:, 1].tolist()) # Probability of class 1 (binding)
            all_predictions.extend(predictions.tolist())

    # --- Save Results ---
    output_df = input_df.copy()
    output_df['Predicted_Label'] = all_predictions
    output_df['Binding_Probability'] = all_probabilities

    output_dir = os.path.dirname(args.output_predictions_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_df.to_csv(args.output_predictions_file, index=False)
    print(f"\nPredictions saved to: {args.output_predictions_file}")

    # --- Evaluate Metrics (if labels provided) ---
    if has_labels:
        print("\n--- Evaluation Metrics ---")
        y_true = true_labels.numpy()
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"MCC:       {mcc:.4f}")
        print(f"AUC:       {auc:.4f}")
        print("Evaluation complete.")

if __name__ == "__main__":
    main()

