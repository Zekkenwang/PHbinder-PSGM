# src/phbinder/dataset.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# Import configuration
from config import phbinder_config as config
from transformers import AutoTokenizer

# Initialize tokenizer globally for data processing
# This is usually done once and passed around, or managed by a singleton if complex.
# For simplicity here, it's initialized once.
tokenizer = AutoTokenizer.from_pretrained(
    config.LOCAL_ESM_MODEL_PATH, 
    local_files_only=True, 
    clean_up_tokenization_spaces=False
)

class TCRDataset(TensorDataset):
    def __init__(self, epitope_data, labels):
        self.epitope_data = epitope_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        epitope = self.epitope_data[idx]
        label = self.labels[idx]
        return epitope, label

def pad_inner_lists_to_length(outer_list, target_length=config.EPITOPE_MAX_LEN):
    """
    Pads inner lists (tokenized sequences) to a target length.
    Padding value is 1 (ESM's default padding token ID).
    """
    padded_list = []
    for inner_list in outer_list:
        current_length = len(inner_list)
        if current_length < target_length:
            padding_length = target_length - current_length
            padded_list.append(inner_list + [1] * padding_length)
        else:
            # Truncate if longer than target_length (though typical for padding is just pad)
            padded_list.append(inner_list[:target_length])
    return padded_list

def addbatch(epitope_data, labels, batchsize):
    """
    Creates a DataLoader from epitope data and labels.
    """
    dataset = TCRDataset(epitope_data.long(), labels)
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return data_loader

def remove_samples_with_length_greater_than(series, length):
    """
    Filters a pandas Series of strings, keeping only those with length <= specified length.
    """
    series_filtered = series[series.str.len() <= length]
    return series_filtered

