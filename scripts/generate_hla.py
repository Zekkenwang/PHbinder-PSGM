# scripts/generate_hla.py

import torch
import pandas as pd
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

# Import configurations
from config.psgm_config import Config

# Import PSGM modules
from src.psgm.vocab import vocab
from src.psgm.model import HLAGenerator
from src.psgm.mapper import HLAMapper
from src.psgm.utils import get_position_masks, generate

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Generate HLA pseudo-sequences using the PSGM model.")
    parser.add_argument("--input_peptides_file", type=str, required=True,
                        help="Path to a CSV file containing peptides. Expected column: 'Peptide'.")
    parser.add_argument("--output_results_file", type=str, default=Config.RESULTS_SAVE_PATH,
                        help="Path to save the generated HLA sequences and mapping results.")
    parser.add_argument("--model_path", type=str, default=Config.MODEL_SAVE_PATH,
                        help="Path to the trained HLAGenerator model (.pth file).")
    parser.add_argument("--hla_db_path", type=str, default=Config.HLA_DB_PATH,
                        help="Path to the HLA pseudo-sequence database CSV for mapping.")
    parser.add_argument("--device", type=str, default=Config.DEVICE,
                        help="Device to use for generation (e.g., 'cuda' or 'cpu').")
    
    args = parser.parse_args()

    set_seed(Config.SEED)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"\nLoading HLAGenerator model from: {args.model_path}")
    model = HLAGenerator(len(vocab)).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("HLAGenerator model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Initialize Mapper ---
    print(f"Initializing HLAMapper with database: {args.hla_db_path}")
    try:
        mapper = HLAMapper(args.hla_db_path)
        print("HLAMapper initialized.")
    except FileNotFoundError:
        print(f"Error: HLA database file not found at {args.hla_db_path}. Please check the path.")
        return
    except Exception as e:
        print(f"Error initializing mapper: {e}")
        return

    # --- Get Position Masks ---
    position_masks = get_position_masks()
    print("Position masks loaded.")

    # --- Load Input Peptides ---
    print(f"Loading input peptides from: {args.input_peptides_file}")
    try:
        input_df = pd.read_csv(args.input_peptides_file)
        if 'Peptide' not in input_df.columns:
            raise ValueError("Input CSV must contain a 'Peptide' column.")
        peptides_to_generate = input_df['Peptide'].astype(str).unique().tolist()
        print(f"Loaded {len(peptides_to_generate)} unique peptides for generation.")
    except FileNotFoundError:
        print(f"Error: Input peptides file not found at {args.input_peptides_file}. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading input peptides: {e}")
        return

    # --- Generate HLAs and Map ---
    all_results = []
    # Load the original training data to find possible matching HLAs for context
    try:
        df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_train_full = df_train_full.drop_duplicates(['peptide', 'allele', 'pseudo_seq'])
    except FileNotFoundError:
        print(f"Warning: Training data not found at {Config.TRAIN_DATA_PATH}. Possible Matches column will be empty.")
        df_train_full = pd.DataFrame(columns=['peptide', 'allele', 'pseudo_seq'])

    print("\nStarting HLA generation and mapping...")
    for pep in tqdm(peptides_to_generate, desc="Processing peptides"):
        # Query training data for matching HLA types and pseudo-sequences (if available)
        matched_hla_entries = df_train_full[df_train_full['peptide'] == pep][['allele', 'pseudo_seq']].values.tolist()
        matched_hla_info = [f"{hla[0]}, {hla[1]}" for hla in matched_hla_entries]

        # Generate HLA pseudo-sequence
        generated_seq = generate(model, pep, device, position_masks)

        # Query HLA database for similarity using the mapper
        matches = mapper.query(generated_seq) 
        
        # Extract top 50 HLA types (alleles) from matches
        top_50_matches_alleles = [match[0] for match in matches] 

        # Prepare a single row for the results DataFrame
        current_row = [pep, generated_seq] 
        
        # Pad Top 50 Matches to ensure 50 columns
        current_row.extend(top_50_matches_alleles + [""] * (50 - len(top_50_matches_alleles)))
        
        # Pad Possible Matches to ensure a consistent number of columns (e.g., 51 for up to 50 matches + 1 for header)
        current_row.extend(matched_hla_info + [""] * (51 - len(matched_hla_info))) 
        
        all_results.append(current_row)

    # Define CSV column names
    columns = ["Peptide", "Generated Sequence"] + \
              [f"Top 50 Matches {i+1}" for i in range(50)] + \
              [f"Possible Matches {i+1}" for i in range(51)]

    # Save results to CSV
    results_df = pd.DataFrame(all_results, columns=columns)
    output_dir = os.path.dirname(args.output_results_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_df.to_csv(args.output_results_file, index=False)
    print(f"\nGeneration complete. Results saved to: {args.output_results_file}")

if __name__ == "__main__":
    main()

