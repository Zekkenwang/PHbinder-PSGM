# psgm/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from config.psgm_config import Config
from psgm.vocab import vocab # Import the global vocab instance

class PeptideHLADataset(Dataset):
    def __init__(self, df):
        self.df = df
        self._strict_filtering()
    
    def _strict_filtering(self):
        # Filter peptides that match the amino acid alphabet and max length
        valid_pep_pattern = f'^[ACDEFGHIKLMNPQRSTVWY]{{1,{Config.peptide_max_len}}}$'
        valid_pep = self.df['peptide'].astype(str).str.match(valid_pep_pattern)
        
        # Filter HLA pseudo-sequences to fixed length
        valid_hla = self.df['pseudo_seq'].astype(str).str.len() == Config.hla_seq_len
        
        # Apply filters and drop duplicates
        self.df = self.df[valid_pep & valid_hla].copy()
        self.df = self.df.drop_duplicates(['peptide', 'pseudo_seq'])
        print(f"Dataset after filtering: {len(self.df)} samples remaining.")

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep_enc = vocab.encode(str(row['peptide'])) # Ensure string type
        hla_enc = vocab.encode(str(row['pseudo_seq'])) # Ensure string type
        return {
            'peptide': torch.tensor(pep_enc, dtype=torch.long),
            'hla': torch.tensor(hla_enc, dtype=torch.long)
        }

def condition_collate(batch):
    """
    Custom collate function for DataLoader to handle variable length sequences.
    Pads sequences to the longest sequence in the batch.
    """
    peptides = [item['peptide'] for item in batch]
    hlas = [item['hla'] for item in batch]
    
    # Pad peptides and HLAs to the max length in the current batch
    peptide_padded = torch.nn.utils.rnn.pad_sequence(peptides, batch_first=True, padding_value=vocab.pad_idx)
    hla_padded = torch.nn.utils.rnn.pad_sequence(hlas, batch_first=True, padding_value=vocab.pad_idx)
    
    return {'peptide': peptide_padded, 'hla': hla_padded}
