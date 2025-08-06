# src/phbinder/lora_finetune.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModel

# Import configuration
from config import phbinder_config as config
# Import utility functions for loss calculation
from src.phbinder.utils import get_val_loss # This function is used in finetune_lora_model
from tqdm import tqdm # For progress bar

# Initialize tokenizer once for use in this module or pass it in
# For LoRA finetuning, the tokenizer is used to prepare inputs.
# If the tokenizer is only used in data preparation (dataset.py),
# you might not need it here directly. But if ESMForMaskedLM is used directly
# (which it's not in your setup_lora_model), it might need it.
# It's better to make it available via config or pass it.
# As your current code doesn't use the tokenizer here, I'll remove direct init.

def setup_lora_model():
    """
    Sets up the base ESM model with LoRA configuration.
    """
    peft_config = LoraConfig(
        target_modules=config.LORA_TARGET_MODULES,  
        task_type=config.LORA_TASK_TYPE,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT
    )
    # Load base ESM model for LoRA
    model = AutoModel.from_pretrained(config.LOCAL_ESM_MODEL_PATH)
    lora_model = get_peft_model(model, peft_config)
    
    # Print trainable parameters to confirm LoRA setup
    lora_model.print_trainable_parameters()
    return lora_model

def finetune_lora_model(lora_model, classifier, dataloader, val_dataloader, device):
    """
    Fine-tunes the LoRA-enabled ESM model.
    
    Args:
        lora_model: The LoRA-enabled ESM model.
        classifier: The linear layer classifier that takes ESM hidden states and outputs class logits.
                    This is treated as a separate trainable component as in the original code.
        dataloader: Training DataLoader.
        val_dataloader: Validation DataLoader.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
    """
    optimizer = torch.optim.AdamW(
        # Optimize both LoRA parameters and classifier parameters
        list(filter(lambda p: p.requires_grad, lora_model.parameters())) + 
        list(filter(lambda p: p.requires_grad, classifier.parameters())), 
        lr=config.LORA_LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss() # Standard CE loss for classification
    
    lora_model.to(device)
    classifier.to(device) # Ensure classifier is also on the device
    
    lora_model.train() # Set LoRA model to training mode
    classifier.train() # Set classifier to training mode

    best_val_loss = float('inf')
    patience_counter = 0
    best_lora_model_state = None # To save best LoRA weights
    best_classifier_state = None # To save best classifier weights

    for epoch in range(config.LORA_NUM_EPOCHS):
        running_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for step, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"LoRA Epoch {epoch + 1} Training")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through LoRA model to get embeddings
            outputs = lora_model(inputs)
            logits_from_esm = outputs.last_hidden_state[:, 0, :] # Get CLS token embedding
            
            # Forward pass through classifier
            logits = classifier(logits_from_esm)
            
            # Calculate loss (using custom loss function from utils)
            loss = get_val_loss(logits, labels, criterion) # Original code uses get_val_loss for training here
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy for monitoring
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(dataloader)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        # Validation loop
        lora_model.eval() # Set to evaluation mode
        classifier.eval() # Set classifier to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc=f"LoRA Epoch {epoch + 1} Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = lora_model(inputs)
                logits_from_esm = outputs.last_hidden_state[:, 0, :]
                logits = classifier(logits_from_esm)
                loss = get_val_loss(logits, labels, criterion)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the state dicts
            best_lora_model_state = lora_model.state_dict()
            best_classifier_state = classifier.state_dict()
            
            # Save the LoRA weights using PEFT's save_pretrained
            lora_model.save_pretrained(config.SAVE_PATH_LORA_WEIGHTS)
            # You might want to save the classifier separately too
            torch.save(classifier.state_dict(), f"{config.SAVE_PATH_LORA_WEIGHTS}/classifier.pt")
            
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")

        if patience_counter >= config.LORA_PATIENCE:
            print("Early stopping triggered for LoRA fine-tuning.")
            break

        lora_model.train() # Set back to training mode
        classifier.train() # Set back to training mode

    # After training, load the best state dictionaries if they exist
    if best_lora_model_state is not None:
        lora_model.load_state_dict(best_lora_model_state)
        classifier.load_state_dict(best_classifier_state)
        # Final save of the best model (redundant if already saved, but ensures latest best)
        lora_model.save_pretrained(config.SAVE_PATH_LORA_WEIGHTS)
        torch.save(classifier.state_dict(), f"{config.SAVE_PATH_LORA_WEIGHTS}/classifier.pt")
        print(f"Best LoRA model and classifier saved to {config.SAVE_PATH_LORA_WEIGHTS}")
    
    return config.SAVE_PATH_LORA_WEIGHTS # Return path to saved LoRA weights

