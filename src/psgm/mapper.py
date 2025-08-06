# psgm/mapper.py

import pandas as pd
import numpy as np
from Bio import pairwise2 # For sequence alignment score
from sklearn.neighbors import NearestNeighbors # For nearest neighbor search

from config.psgm_config import Config
from psgm.vocab import vocab # Import the global vocab instance

class HLAMapper:
    def __init__(self, db_path):
        self.hla_seq_len = Config.HLA_SEQ_LEN
        self.db = pd.read_csv(db_path)
        self._validate_columns(['allele', 'pseudo_seq'])
        self._preprocess()
        self._build_index()

    def _validate_columns(self, required_cols):
        missing_cols = [col for col in required_cols if col not in self.db.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要列: {missing_cols}")
        print(f"列验证通过: 找到所有必要列 {required_cols}")    
    
    def _preprocess(self):
        # Ensure pseudo_seq is string and remove leading/trailing spaces
        self.db['pseudo_seq'] = self.db['pseudo_seq'].astype(str).str.strip().str[:self.hla_seq_len]
        
        # Filter out sequences that don't match the required length after truncation
        invalid = self.db['pseudo_seq'].str.len() != self.hla_seq_len
        if invalid.any():
            print(f"警告: 发现{invalid.sum()}条长度不等于{self.hla_seq_len}的序列，已过滤")
            self.db = self.db[~invalid]
        
    def _sequence_to_vector(self, seq):
        # Pad sequence with 'X' (or a specific token) if shorter than hla_seq_len
        seq_padded = seq[:self.hla_seq_len].ljust(self.hla_seq_len, 'X')
        return [vocab.token_to_idx.get(c, vocab.pad_idx) for c in seq_padded]
    
    def _build_index(self):
        sequences = self.db['pseudo_seq'].tolist()
        self.seq_matrix = np.array([self._sequence_to_vector(s) for s in sequences])
        assert self.seq_matrix.shape[1] == self.hla_seq_len, \
            f"特征维度错误: 期望{self.hla_seq_len}, 实际{self.seq_matrix.shape[1]}"
        
        # Use Hamming distance for sequence similarity in nearest neighbors
        self.nn = NearestNeighbors(n_neighbors=50, metric='hamming')
        self.nn.fit(self.seq_matrix)
        print(f"Built NearestNeighbors index for {len(self.db)} HLA pseudo-sequences.")

    def query(self, generated_seq):
        """
        Queries the HLA database for the most similar pseudo-sequences.
        Returns top 50 matches based on normalized alignment score.
        """
        clean_seq = self._clean_sequence(generated_seq)
        
        # Ensure query sequence is of fixed length for NearestNeighbors
        query_vec = np.array([self._sequence_to_vector(clean_seq)])
        
        distances, indices = self.nn.kneighbors(query_vec)
        results = []
        for i, idx in enumerate(indices[0]):
            db_entry = self.db.iloc[idx]
            db_seq = str(db_entry['pseudo_seq']) # Ensure string type
            
            # Use pairwise2 for a more accurate alignment score
            # A score of 0 indicates no alignment.
            # Using globalxx for general sequence similarity.
            alignments = pairwise2.align.globalxx(clean_seq, db_seq)
            score = alignments[0].score if alignments else 0.0
            
            # Normalize score by the length of the longer sequence for a ratio
            norm_score = score / max(len(clean_seq), len(db_seq), 1) # Avoid division by zero
            
            results.append((
                db_entry['allele'],
                norm_score,
                db_seq
            ))
        
        # Sort by normalized score in descending order and return top 50
        return sorted(results, key=lambda x: -x[1])[:50]
    
    def _clean_sequence(self, seq):
        """
        Removes any special tokens from a sequence and truncates/pads to HLA_SEQ_LEN.
        """
        # Ensure seq is a string
        seq = str(seq) 
        
        # Remove any placeholder or special tokens that might be in the generated sequence
        # Note: Original code had PEP/HLA_START/END tokens which are not explicitly used in vocab
        # If your generation process introduces them, add them to special_tokens.
        # For now, just filter for amino acids.
        
        cleaned_seq = ''.join([c for c in seq if c in vocab.amino_acids])
        
        # Truncate if longer than Config.hla_seq_len, pad with 'X' if shorter
        return cleaned_seq[:Config.hla_seq_len].ljust(Config.hla_seq_len, 'X')
