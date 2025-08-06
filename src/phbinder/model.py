# src/phbinder/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import PeftModel
from transformers import EsmModel

# Import configuration
from config import phbinder_config as config

class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        b, c, h, w = x.shape
        batch_size = b

        x = self.proj_in(x) 
        x = rearrange(x, 'b c h w -> b (h w) c')  

        Q = self.Wq(x)  
        K = self.Wk(context)  
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K) * self.scale

        if pad_mask is not None:
            # Expand pad_mask to match attention weights shape
            # Assuming pad_mask is [batch_size, 1, seq_len] or [batch_size, seq_len]
            # It needs to be [batch_size, num_heads, query_len, key_len]
            # Here, context is [batch_size, 1, emb_dim] so key_len is 1. x is (h*w), so query_len is (h*w).
            # The original code's pad_mask usage is for self-attention,
            # for cross-attention, the mask needs to align with K (context).
            # Given context is a single vector, pad_mask is unlikely to be used here as it's not sequence based.
            # If your context ever becomes sequence-based, adjust this.
            # For now, if pad_mask is passed, ensure its shape. Let's assume it aligns with context's 'seq_len'
            # (which is 1 here, so mask will be applied to a single element if at all).
            # Re-evaluating based on x.shape (b c h w) and context (b 1 c):
            # Q is (b, num_heads, h*w, depth), K is (b, num_heads, 1, depth)
            # att_weights is (b, num_heads, h*w, 1)
            # So pad_mask should be (b, num_heads, h*w, 1) to mask columns.
            # If pad_mask is [B, context_len], it should apply to K. The original code applies it to att_weights.
            # I will remove the pad_mask logic for Cross-Attention here as its usage is ambiguous for your current context setup.
            # If it's truly needed, you'll need to define how pad_mask should be structured for cross-attention.
            pass # Removed pad_mask application as it's not typically used this way for context as single vector

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)  
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  
        out = self.proj_out(out)  

        return out, att_weights


class This_work(nn.Module):
    def __init__(self, lora_weights_path):
        super(This_work, self).__init__()

        # Model hyperparameters from config
        n_layers = config.TRANSFORMER_N_LAYERS
        n_head = config.TRANSFORMER_N_HEAD
        d_model = config.TRANSFORMER_D_MODEL
        d_ff = config.TRANSFORMER_D_FF
        cnn_num_channel = config.CNN_NUM_CHANNEL
        region_embedding_size = config.CNN_REGION_EMBEDDING_SIZE
        cnn_kernel_size = config.CNN_KERNEL_SIZE
        cnn_padding_size = config.CNN_PADDING_SIZE
        cnn_stride = config.CNN_STRIDE
        pooling_size = config.CNN_POOLING_SIZE

        self.cross_attention = Cross_MultiAttention(
            in_channels = cnn_num_channel,  
            emb_dim = d_model,              
            num_heads = n_head            
        )

        # Load base ESM model and then the LoRA weights
        self.esm = EsmModel.from_pretrained(config.LOCAL_ESM_MODEL_PATH)
        for param in self.esm.parameters():
            param.requires_grad = False # Freeze base ESM parameters
        self.lora_esm = PeftModel.from_pretrained(self.esm, lora_weights_path)
        # Note: The `self.peft_config = LoraConfig(...)` line from original code is removed here,
        # as it was not used for loading a PeftModel, and the config is stored within the loaded model.

        self.region_cnn1 = nn.Conv1d(d_model, cnn_num_channel, region_embedding_size)
        self.padding1 = nn.ConstantPad1d((1, 1), 0)
        self.padding2 = nn.ConstantPad1d((0, 1), 0) # For cnn_block2, this looks like right padding by 1
        self.relu = nn.LeakyReLU()
        self.cnn1 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size,
                              padding=cnn_padding_size, stride=cnn_stride)
        self.cnn2 = nn.Conv1d(cnn_num_channel, cnn_num_channel, kernel_size=cnn_kernel_size, # This CNN2 is not used in cnn_block1/2
                              padding=cnn_padding_size, stride=cnn_stride) # Only cnn1 is used repeatedly
        self.maxpooling = nn.MaxPool1d(kernel_size=pooling_size)
        
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_ff, 
            dropout=config.TRANSFORMER_DROPOUT, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=n_layers)

        self.bn1 = nn.BatchNorm1d(d_model) # This bn1 is not used in forward pass
        self.bn2 = nn.BatchNorm1d(cnn_num_channel)
        
        self.fc_task = nn.Sequential(
            nn.Linear(2 * cnn_num_channel, config.FC_TASK_HIDDEN_SIZE_1),
            nn.Dropout(config.FC_TASK_DROPOUT),
            nn.ReLU(),
            nn.Linear(config.FC_TASK_HIDDEN_SIZE_1, config.FC_TASK_HIDDEN_SIZE_2),
        )
        self.classifier = nn.Linear(config.FC_TASK_HIDDEN_SIZE_2, config.NUM_CLASSES)

    def cnn_block1(self, x):
        return self.cnn1(self.relu(x))

    def cnn_block2(self, x):
        x = self.padding2(x) # Apply padding before pooling
        px = self.maxpooling(x)
        x = self.relu(px)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn1(x) # Double application of cnn1
        x = px + x # Residual connection
        return x

    def forward(self, x_in):
        # x_in: [batch_size, sequence_length] (token IDs)
        outputs = self.lora_esm(x_in, output_hidden_states=True)
        # Assuming hidden_states[30] is the desired layer for ESM-2 t30_150M_UR50D
        emb = outputs.hidden_states[30] # [batch_size, sequence_length, d_model]
        
        # Transformer Branch
        output = self.transformer_encoder(emb) # [batch_size, sequence_length, d_model]
        representation = output.mean(dim=1) # [batch_size, d_model] (mean pooling over sequence)

        # CNN Branch
        cnn_emb = self.region_cnn1(emb.transpose(1, 2)) # [batch_size, d_model, sequence_length] -> [batch_size, cnn_num_channel, sequence_length - region_embedding_size + 1]
        cnn_emb = self.padding1(cnn_emb) # Adjust padding based on kernel size
        conv = cnn_emb + self.cnn_block1(self.cnn_block1(cnn_emb)) # Initial CNN layers with residual
        
        # Apply cnn_block2 repeatedly until sequence length is 1
        while conv.size(-1) >= 2: # Keep applying block until sequence dimension is reduced to 1
            conv = self.cnn_block2(conv)
            # This loop needs careful consideration if padding/pooling doesn't guarantee reduction to 1.
            # If maxpooling kernel=2, stride=2, padding=0, it halves the dim.
            # If dim is 1, `padding2` makes it 2, `maxpooling` makes it 1.
            # If dim is 2, `padding2` makes it 3, `maxpooling` makes it 1.
            # This loop should eventually terminate if maxpooling works as expected.
        
        cnn_out = torch.squeeze(conv, dim=-1) # Remove the sequence dimension, [batch_size, cnn_num_channel]
        cnn_out = self.bn2(cnn_out) # [batch_size, cnn_num_channel]

        # Cross Multi-Head Attention
        # x_in for cross_attention is cnn_out_expanded [b, cnn_num_channel, 1, 1]
        # context for cross_attention is representation [b, 1, d_model]
        cnn_out_expanded = cnn_out.unsqueeze(-1).unsqueeze(-1)  # [batch_size, cnn_num_channel, 1, 1]
        context = representation.unsqueeze(1)  # [batch_size, 1, d_model]
        
        att_output, att_weights = self.cross_attention(x=cnn_out_expanded, context=context)
        
        att_output = att_output.view(att_output.size(0), -1)  # [batch_size, cnn_num_channel * 1 * 1] -> [batch_size, cnn_num_channel]
        
        # Concatenate outputs from attention and CNN branch
        final_representation = torch.concat((att_output, cnn_out), dim=1) # [batch_size, 2 * cnn_num_channel]
        
        # Fully connected task layers
        reduction_feature = self.fc_task(final_representation) # [batch_size, FC_TASK_HIDDEN_SIZE_2]
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1) # Ensure flat

        # Classifier
        logits_clsf = self.classifier(reduction_feature)
        logits_clsf = torch.nn.functional.softmax(logits_clsf, dim=1)
        return logits_clsf, reduction_feature

