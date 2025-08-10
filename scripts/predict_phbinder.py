import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from config import phbinder_config as config
from src.phbinder.model import This_work
from src.phbinder.dataset import pad_inner_lists_to_length
import os

def predict_on_external_data(model, tokenizer, sequences, device):
    sequences_upper = [s.upper() for s in sequences]
    
    sequences_encoding = tokenizer(sequences_upper)['input_ids']
    padded_encodings = pad_inner_lists_to_length(sequences_encoding)
    
    data_tensor = torch.tensor(padded_encodings, dtype=torch.long)
    model.eval() 
    model.to(device)
    data_tensor = data_tensor.to(device)

    with torch.no_grad():
        probabilities, _ = model(data_tensor)
        _, predicted_classes = torch.max(probabilities, 1)

    return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()


def main():
    device = config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(
        config.LOCAL_ESM_MODEL_PATH,
        local_files_only=True,
        clean_up_tokenization_spaces=False
    )

    lora_adapter_path = config.SAVE_PATH_LORA_WEIGHTS
    model = This_work(lora_adapter_path)
    
    best_model_path = os.path.join(config.SAVE_PATH_MAIN_MODEL_CHECKPOINTS, "best_model_I.pt")
    
    print(f"Loading trained weights from: {best_model_path}")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{best_model_path}'.")
        return

    print(f"Loading external data from: {config.EXTERNAL_DATA_PATH}")
    try:
        external_df = pd.read_csv(config.EXTERNAL_DATA_PATH)
        if 'Epitope' not in external_df.columns:
            print("ERROR: External data CSV must contain a column named 'Epitope'.")
            return
        external_sequences = external_df['Epitope'].tolist()
    except FileNotFoundError:
        print(f"ERROR: External data file not found at '{config.EXTERNAL_DATA_PATH}'.")
        return

    predicted_labels, class_probabilities = predict_on_external_data(
        model,
        tokenizer,
        external_sequences,
        device
    )

    external_df['Predicted_Label'] = predicted_labels
    external_df['Probability_Class_0'] = class_probabilities[:, 0]
    external_df['Probability_Class_1'] = class_probabilities[:, 1]

    try:
        external_df.to_csv(config.PREDICTION_OUTPUT_PATH, index=False)
        print(f"Results saved to: {config.PREDICTION_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving results: {e}")

    print(external_df.head())


if __name__ == "__main__":
    main()

