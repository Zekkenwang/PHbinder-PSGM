# psgm/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
from tqdm import tqdm

from config.psgm_config import Config
from psgm.vocab import vocab # Import the global vocab instance
from psgm.mapper import HLAMapper # Import HLAMapper for evaluate_results
from psgm.model import HLAGenerator # Import HLAGenerator for type hinting/clarity in generate/evaluate_results

# Define amino acid masks for specific positions based on your provided data
# This is a static mapping, so it doesn't need to depend on 'df'.
def get_position_masks():
    position_masks = {}
    position_masks[0] = [vocab.token_to_idx['Y']]
    position_masks[1] = [vocab.token_to_idx[aa] for aa in ['Y', 'F', 'H', 'D', 'S', 'T']]
    position_masks[2] = [vocab.token_to_idx[aa] for aa in ['A', 'S', 'T', 'D', 'F']]
    position_masks[3] = [vocab.token_to_idx[aa] for aa in ['M', 'S', 'T', 'G', 'E']]
    position_masks[4] = [vocab.token_to_idx[aa] for aa in ['Y', 'E']]
    position_masks[5] = [vocab.token_to_idx[aa] for aa in ['Q', 'G', 'R', 'L']]
    position_masks[6] = [vocab.token_to_idx[aa] for aa in ['E', 'N', 'K']]
    position_masks[7] = [vocab.token_to_idx[aa] for aa in ['N', 'K', 'I', 'E']]
    position_masks[8] = [vocab.token_to_idx[aa] for aa in ['M', 'V', 'Y', 'C', 'R']]
    position_masks[9] = [vocab.token_to_idx[aa] for aa in ['A', 'S', 'R', 'Q']]
    position_masks[10] = [vocab.token_to_idx[aa] for aa in ['H', 'Q', 'A', 'T']]
    position_masks[11] = [vocab.token_to_idx[aa] for aa in ['T', 'K']]
    position_masks[12] = [vocab.token_to_idx[aa] for aa in ['D', 'H', 'T', 'N', 'Y', 'V']]
    position_masks[13] = [vocab.token_to_idx[aa] for aa in ['A', 'V', 'D', 'E', 'Y']]
    position_masks[14] = [vocab.token_to_idx[aa] for aa in ['N', 'D', 'S', 'V', 'E']]
    position_masks[15] = [vocab.token_to_idx[aa] for aa in ['T', 'N', 'S', 'K']]
    position_masks[16] = [vocab.token_to_idx[aa] for aa in ['L', 'A', 'I', 'V']]
    position_masks[17] = [vocab.token_to_idx[aa] for aa in ['Y', 'T']]
    position_masks[18] = [vocab.token_to_idx[aa] for aa in ['I', 'V', 'L', 'W', 'F']]
    position_masks[19] = [vocab.token_to_idx[aa] for aa in ['I', 'R', 'M', 'S', 'W', 'L', 'F']]
    position_masks[20] = [vocab.token_to_idx[aa] for aa in ['Y', 'C', 'F', 'S', 'N']]
    position_masks[21] = [vocab.token_to_idx[aa] for aa in ['R', 'H', 'Q', 'D', 'N']]
    position_masks[22] = [vocab.token_to_idx[aa] for aa in ['D', 'H', 'N', 'F']]
    position_masks[23] = [vocab.token_to_idx[aa] for aa in ['Y', 'L']]
    position_masks[24] = [vocab.token_to_idx[aa] for aa in ['T', 'S', 'N', 'D']]
    position_masks[25] = [vocab.token_to_idx[aa] for aa in ['W', 'L', 'S']]
    position_masks[26] = [vocab.token_to_idx[aa] for aa in ['A', 'W']]
    position_masks[27] = [vocab.token_to_idx[aa] for aa in ['V', 'E', 'A', 'R', 'L']]
    position_masks[28] = [vocab.token_to_idx[aa] for aa in ['L', 'R', 'D', 'Q', 'T']]
    position_masks[29] = [vocab.token_to_idx[aa] for aa in ['A', 'W', 'L', 'T']]
    position_masks[30] = [vocab.token_to_idx[aa] for aa in ['Y', 'T', 'E']]
    position_masks[31] = [vocab.token_to_idx[aa] for aa in ['T', 'R', 'L', 'G', 'E']]
    position_masks[32] = [vocab.token_to_idx[aa] for aa in ['W', 'G', 'H', 'S']]
    position_masks[33] = [vocab.token_to_idx[aa] for aa in ['Y', 'W', 'H']]
    return position_masks


def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """
    Applies Top-P (nucleus) sampling to logits.
    logits: [batch_size, vocab_size] (usually batch_size=1 for generation)
    """
    logits = logits / temperature 
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # The mask ensures that at least one token is selected
    sorted_mask = cumulative_probs < top_p
    # Add the token that exceeded the threshold
    sorted_mask = torch.cat([sorted_mask, torch.ones_like(sorted_mask[:, :1], dtype=torch.bool)], dim=-1)

    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, 
                             src=sorted_logits.masked_fill(~sorted_mask, float('-inf'))
                            )

    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


def generate(model: HLAGenerator, peptide: str, device: torch.device, position_masks: dict, 
             max_len: int = Config.HLA_SEQ_LEN, top_p: float = 0.9, temperature: float = 1.0):
    """
    Generates an HLA pseudo-sequence conditioned on a peptide.
    
    Args:
        model (HLAGenerator): The trained HLA generator model.
        peptide (str): The input peptide sequence.
        device (torch.device): The device to run generation on.
        position_masks (dict): Dictionary of allowed tokens for each position.
        max_len (int): Maximum length of the generated HLA sequence.
        top_p (float): Top-P sampling parameter.
        temperature (float): Sampling temperature.
    
    Returns:
        str: The generated HLA pseudo-sequence.
    """
    model.eval()
    pep_enc = vocab.encode(peptide)

    with torch.no_grad():
        peptide_tensor = torch.tensor(pep_enc, device=device).unsqueeze(0) # Add batch dimension
        
        # Encode peptide to get memory for decoder
        memory = model.encode_peptide(peptide_tensor) # [1, peptide_max_len, embed_dim]

        # Start generation with the first position's constrained token
        # The first token is hardcoded to 'Y' at position 0 in get_position_masks.
        # This assumes position_masks[0] only contains 'Y'.
        # If it can contain multiple, it should be sampled from position_masks[0].
        # Original code used `generated = [vocab.token_to_idx['Y']]`, so we keep that.
        generated_ids = [vocab.token_to_idx['Y']]

        for pos in range(1, max_len): # Iterate from position 1 up to max_len-1
            # Current generated sequence as decoder input
            hla_input = torch.tensor(generated_ids, device=device).unsqueeze(0) # Add batch dimension
            
            # Decode HLA based on current sequence and peptide memory
            # The decoder expects the full sequence as input, and it will predict the next token.
            # We take only the last prediction.
            output = model.decode_hla(hla_input, memory) # [1, current_hla_len, embed_dim]
            logits = model.fc_out(output[:, -1, :])  # Logits for the next token [1, vocab_size]

            # Apply position-specific mask
            mask_indices = position_masks.get(pos, []) # Get allowed tokens for current position
            
            # If no specific mask, allow all amino acids (or handle as error)
            if not mask_indices:
                # Default to allowing all amino acids if no specific mask is defined for a position
                mask_indices = [vocab.token_to_idx[aa] for aa in vocab.amino_acids]

            masked_logits = torch.full_like(logits, float('-inf'))
            if mask_indices: # Ensure mask_indices is not empty
                masked_logits[:, mask_indices] = logits[:, mask_indices] # Only keep logits for allowed tokens
            
            # Sample next token
            next_token = top_p_sampling(masked_logits, top_p=top_p, temperature=temperature)

            # Fallback if sampled token is not valid (e.g., if masked_logits was all -inf)
            # This check (`next_token >= len(vocab)`) might be redundant if sampling is robust.
            # The more important check is `next_token not in mask_indices` if `mask_indices` is strict.
            if next_token not in mask_indices:
                # If the sampled token is somehow not in the allowed mask, pick a random one from the mask
                # This can happen if top_p sampling causes issues with heavily masked logits,
                # or if the initial sample was somehow outside the masked region (unlikely with scatter_).
                if mask_indices: # Ensure there's something to choose from
                    next_token = random.choice(mask_indices)
                else: # Should not happen if default mask logic above is robust
                    next_token = vocab.token_idx['[MASK]'] # Fallback to mask token or specific default
            
            generated_ids.append(next_token)

    return vocab.decode(generated_ids)[:Config.HLA_SEQ_LEN]


def evaluate_results(model: HLAGenerator, mapper: HLAMapper, device: torch.device, 
                     test_peptides: list, position_masks: dict):
    """
    Generates HLA sequences for a list of test peptides, maps them to known HLA types,
    and saves the results to a CSV file.
    
    Args:
        model (HLAGenerator): The trained HLA generator model.
        mapper (HLAMapper): The HLA mapper instance.
        device (torch.device): The device to run generation on.
        test_peptides (list): List of peptide strings to generate HLAs for.
        position_masks (dict): Dictionary of allowed tokens for each position.
        
    Returns:
        pd.DataFrame: DataFrame containing the generated results.
    """
    results = []
    
    # Load the original training data to find possible matching HLAs
    try:
        df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_train_full = df_train_full.drop_duplicates(['peptide', 'allele', 'pseudo_seq'])
    except FileNotFoundError:
        print(f"Error: Training data not found at {Config.TRAIN_DATA_PATH}. Cannot fetch possible matches.")
        df_train_full = pd.DataFrame(columns=['peptide', 'allele', 'pseudo_seq'])


    for pep in tqdm(test_peptides, desc="Generating and Evaluating HLAs"):
        # Query training data for matching HLA types and pseudo-sequences
        matched_hla_entries = df_train_full[df_train_full['peptide'] == pep][['allele', 'pseudo_seq']].values.tolist()
        # Format as "allele, pseudo_seq"
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
        
        # Pad Possible Matches to ensure a consistent number of columns (e.g., 50 or a max if varying)
        # Using 51 columns as in original code, implying max 51 possible matches
        current_row.extend(matched_hla_info + [""] * (51 - len(matched_hla_info))) 
        
        results.append(current_row)

    # Define CSV column names
    columns = ["Peptide", "Generated Sequence"] + \
              [f"Top 50 Matches {i+1}" for i in range(50)] + \
              [f"Possible Matches {i+1}" for i in range(51)]

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(Config.RESULTS_SAVE_PATH, index=False)
    print(f"Generation results saved to: {Config.RESULTS_SAVE_PATH}")

    return results_df

