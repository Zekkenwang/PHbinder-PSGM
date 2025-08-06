# psgm/vocab.py

class HLAVocab:
    def __init__(self):
        self.special_tokens = ['[PAD]', '[MASK]']
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.vocab = self.special_tokens + self.amino_acids
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.pad_idx = self.token_to_idx['[PAD]']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.vocab_size
    
    def encode(self, seq, add_special=False):
        tokens = list(seq)
        # Use .get() with a default value for unknown characters
        return [self.token_to_idx.get(c, self.token_to_idx['[MASK]']) for c in tokens]
    
    def decode(self, ids):
        # Decode only known amino acids, filter out special tokens or unknown IDs
        return ''.join([self.idx_to_token.get(i, '') for i in ids if self.idx_to_token.get(i, '') in self.amino_acids])

# Global instance of the vocabulary
vocab = HLAVocab()
