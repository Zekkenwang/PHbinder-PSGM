# psgm/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm # For ESM-2 model
from transformers import EsmModel # Although esm.pretrained is used, EsmModel might be here for consistency if needed later

from config.psgm_config import Config
from psgm.vocab import vocab # Import the global vocab instance

class HLAGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_dim = Config.EMBED_DIM
        self.esm2_dim = Config.ESM2_DIM
        self.device = torch.device(Config.DEVICE)
        
        # Load ESM-2 model
        # Note: esm.pretrained returns the model and its alphabet
        self.esm2, self.esm2_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.esm2 = self.esm2.to(self.device)
        self.esm2.eval() # Keep ESM2 in evaluation mode, typically frozen during generation
        
        self.pep_embed = nn.Linear(self.esm2_dim, self.embed_dim)
        self.hla_embed = nn.Embedding(vocab_size, self.embed_dim, padding_idx=vocab.pad_idx) # Add padding_idx
        # Position encoder: +2 for potential [CLS] and [EOS] tokens if they were explicitly used
        # For a fixed length, Config.hla_seq_len is sufficient if we just use 0 to max_len-1 positions.
        # Original code uses Config.hla_seq_len + 2, maintaining that.
        self.pos_encoder = nn.Embedding(Config.hla_seq_len + 2, self.embed_dim) 
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim, 
            nhead=Config.NHEAD, 
            dim_feedforward=1024, # Standard Transformer default
            batch_first=True,
            norm_first=True # Original code uses norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=Config.DEC_LAYERS)
        self.fc_out = nn.Linear(self.embed_dim, vocab_size)
        
    def encode_peptide(self, src):
        # src: [batch_size, peptide_len] (token IDs)
        batch_converter = self.esm2_alphabet.get_batch_converter()
        
        # Convert token IDs back to sequences for ESM batch converter
        data = []
        for i in range(src.size(0)):
            token_ids = src[i].tolist()
            # Filter out padding tokens for sequence reconstruction
            seq = "".join([vocab.idx_to_token[t] for t in token_ids if t != vocab.pad_idx])
            data.append((f"peptide_{i}", seq))

        # ESM batch converter expects a list of tuples (sequence_name, sequence_string)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad(): # ESM embeddings are usually frozen during training of the main model
            # Get representations from the last layer (layer 30 for esm2_t30_150M_UR50D)
            results = self.esm2(batch_tokens, repr_layers=[30]) 
            token_representations = results["representations"][30]  # [batch_size, seq_len + 2 (CLS+EOS), 640]
        
        # Extract embeddings corresponding to the original peptide sequence (excluding CLS and EOS)
        # Assuming CLS is at index 0 and EOS is at the end.
        src_emb = token_representations[:, 1:-1, :]  # [batch_size, actual_peptide_len, 640]
        
        # Pad src_emb back to Config.peptide_max_len if peptides were of varying lengths
        # after removing CLS/EOS. This ensures consistent memory shape for decoder.
        # Original code assumes src_emb is already padded correctly. If peptides
        # were already padded with PAD tokens before ESM, this would be fine.
        # However, ESM's `batch_converter` also handles padding. We need to be careful
        # if the input `src` comes from `pad_sequence` which pads to batch max length,
        # then ESM adds CLS/EOS.
        # The `src` here (peptide_ids) is `peptide_padded` from `condition_collate`.
        # So it's already padded with `vocab.pad_idx`. `batch_converter` will then remove padding
        # and add CLS/EOS.
        # The correct way to handle `src_emb` after ESM:
        # If `src` had varying lengths, `src_emb` also has varying lengths for its second dim.
        # We need to pad `src_emb` to `Config.peptide_max_len` again for Transformer input.
        
        # A simple approach for fixed-size memory if all peptides are padded to max_len beforehand:
        # For variable length peptides, pad src_emb to Config.peptide_max_len using a manual pad.
        # For simplicity, if input peptides are already fixed size or memory does not require fixed size,
        # we can proceed. The problem states `peptide_max_len`, implying a fixed size.
        
        # If input peptides are already padded to Config.peptide_max_len using vocab.pad_idx,
        # then `batch_tokens` will also be padded by ESM's internal logic.
        # The original code just takes `token_representations[:, 1:-1, :]`. This implies that
        # `token_representations` for sequences shorter than `Config.peptide_max_len` + 2
        # will implicitly be shorter.
        # For a fixed `memory` size required by Transformer, we must ensure padding.
        
        # Let's assume input `src` (peptide_ids) are already padded to Config.peptide_max_len
        # The `esm.pretrained` models process variable lengths but return padded representations
        # up to the longest sequence in the batch. Here, we want fixed length.
        # The easiest is to ensure `src_emb` is always `[batch_size, Config.peptide_max_len, 640]`
        
        # Current logic for `src_emb` from ESM:
        # ESM's batch_converter pads to the longest sequence in `data`.
        # `token_representations` includes special tokens for ESM.
        # `[:, 1:-1, :]` removes CLS/EOS. If original `seq` was padded, those pads are preserved.
        
        # The `pad_sequence` in `condition_collate` pads to batch's max length, not `Config.peptide_max_len`.
        # So `src` has variable length in its second dimension.
        # The `src_emb` obtained from ESM will also have varying sequence length dimension.
        # TransformerDecoder's `memory` argument can handle variable length, but it must be passed
        # with a `memory_key_padding_mask` if needed. Original code doesn't use it for generator.
        # The simplest way to achieve a fixed `memory` dimension is to slice/pad `src_emb` explicitly.
        
        # To make src_emb fixed length:
        # 1. Truncate if longer than Config.peptide_max_len
        # 2. Pad if shorter than Config.peptide_max_len
        actual_pep_len = src_emb.size(1)
        if actual_pep_len > Config.peptide_max_len:
            src_emb = src_emb[:, :Config.peptide_max_len, :]
        elif actual_pep_len < Config.peptide_max_len:
            padding = torch.zeros(src_emb.size(0), Config.peptide_max_len - actual_pep_len, src_emb.size(2), device=self.device)
            src_emb = torch.cat([src_emb, padding], dim=1)
            
        src_emb = self.pep_embed(src_emb)  # [batch_size, Config.peptide_max_len, Config.EMBED_DIM]
        return src_emb

    def decode_hla(self, tgt, memory, tgt_mask=None):
        # tgt: [batch_size, hla_len] (token IDs for target HLA sequence)
        # memory: [batch_size, peptide_len, embed_dim] (encoded peptide)
        
        # Add positional embeddings to HLA embeddings
        # Assuming tgt sequence starts from index 0 for positional encoding
        pos_emb = self.pos_encoder(torch.arange(tgt.size(1), device=tgt.device))
        tgt_emb = self.hla_embed(tgt) + pos_emb
        
        # Transformer Decoder expects tgt_mask for causal masking
        # It's usually a square mask for auto-regressive generation
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        return self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

    def forward(self, peptide_ids, hla_ids):
        # peptide_ids: [batch_size, peptide_len] (input peptide sequence)
        # hla_ids: [batch_size, hla_len] (target HLA sequence, shifted right for decoder)
        
        memory = self.encode_peptide(peptide_ids) # [batch_size, peptide_max_len, embed_dim]
        
        # `hla_ids[:, :-1]` is used as input for decoder (teacher forcing), as output predicts `hla_ids[:, 1:]`
        decoder_output = self.decode_hla(hla_ids[:, :-1], memory) 
        # decoder_output: [batch_size, hla_len-1, embed_dim]
        
        return self.fc_out(decoder_output) # [batch_size, hla_len-1, vocab_size]


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim=Config.EMBED_DIM, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab.pad_idx) # Add padding_idx
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=Config.NHEAD, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, seq):
        # seq: [batch_size, seq_len] (token IDs)
        embedded = self.embed(seq) # [batch_size, seq_len, embed_dim]
        
        # Create a padding mask for the Transformer Encoder
        # pad_mask: True where padding token, False otherwise
        pad_mask = (seq == vocab.pad_idx) # [batch_size, seq_len]
        
        encoded = self.encoder(embedded, src_key_padding_mask=pad_mask) # [batch_size, seq_len, embed_dim]
        
        # Global average pooling on sequence dimension
        pooled = encoded.mean(dim=1) # [batch_size, embed_dim]
        
        return torch.sigmoid(self.fc(pooled)) # [batch_size, 1]
