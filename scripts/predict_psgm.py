import torch
import pandas as pd
import os
import sys
from config.psgm_config import Config
from src.psgm.vocab import vocab
from src.psgm.model import HLAGenerator
from src.psgm.mapper import HLAMapper
from src.psgm.utils import get_position_masks, evaluate_results


def predict_and_evaluate():
    print("--- Starting PSGM Prediction and Evaluation on External Data ---")
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    print("Initializing model architecture...")
    model = HLAGenerator(len(vocab)).to(device)

    print(f"Loading trained generator weights from: {Config.MODEL_SAVE_PATH}")
    try:
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{Config.MODEL_SAVE_PATH}'.")
        print("Please ensure you have trained the model and the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return
        
    model.eval() 

    print("Loading evaluation utilities (HLAMapper, Position Masks)...")
    try:
        mapper = HLAMapper(Config.HLA_DB_PATH)
    except FileNotFoundError:
        print(f"ERROR: HLA Database file not found at '{Config.HLA_DB_PATH}'.")
        print("Please check the path in your config file.")
        return
        
    position_masks = get_position_masks()
    
    print(f"Loading external data from: {Config.EXTERNAL_DATA_PATH_PSGM}")
    try:
        external_df = pd.read_csv(Config.EXTERNAL_DATA_PATH_PSGM)
        if 'peptide' not in external_df.columns:
            print("ERROR: External data CSV must contain a column named 'peptide'.")
            return
        external_peptides = external_df['peptide'].unique().tolist()
        print(f"Found {len(external_peptides)} unique peptides for evaluation.")
    except FileNotFoundError:
        print(f"ERROR: External data file not found at '{Config.EXTERNAL_DATA_PATH_PSGM}'.")
        print("Please update the path in your config/psgm_config.py file.")
        return

    print("\nStarting generation and evaluation process... This may take a while.")
    results_df = evaluate_results(
    model,
    mapper,
    device,
    external_peptides, 
    position_masks
)
    
    print("\n--- Evaluation Complete ---")
    try:
        output_dir = os.path.dirname(Config.PREDICTION_RESULTS_PATH_PSGM)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        results_df.to_csv(Config.PREDICTION_RESULTS_PATH_PSGM, index=False)
        print(f"Results saved to: {Config.PREDICTION_RESULTS_PATH_PSGM}")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\n--- Sample of Evaluation Results ---")
    print(results_df.head())


if __name__ == "__main__":
    predict_and_evaluate()
