# src/phbinder/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, matthews_corrcoef, f1_score, recall_score, precision_score
)

# Import configuration
from config import phbinder_config as config
# Import dataset functions for DataLoader creation
from src.phbinder.dataset import TCRDataset, addbatch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True # Enables cuDNN autotuner for faster convolutions

def set_seed(seed=config.SEED):
    """
    Sets random seeds for reproducibility across torch, numpy, and python random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed) # Optional: set PYTHONHASHSEED for hash randomness

def get_entropy(probs):
    """
    Calculates entropy from probabilities.
    Input probs: [batch_size, num_classes]
    """
    # Mean over batch, then compute entropy.
    # Add a small epsilon to log to prevent log(0)
    ent = -(probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent

def get_cond_entropy(probs):
    """
    Calculates conditional entropy from probabilities.
    Input probs: [batch_size, num_classes]
    """
    # Entropy for each sample, then mean over batch.
    # Add a small epsilon to log to prevent log(0)
    cond_ent = -(probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent

def get_val_loss(logits, label, criterion):
    """
    Calculates the combined validation loss, incorporating custom terms.
    """
    # Standard CrossEntropyLoss
    loss = criterion(logits.view(-1, config.NUM_CLASSES), label.view(-1))
    loss = (loss.float()).mean()
    
    # Custom loss modification from original code
    loss = (loss - config.LOSS_OFFSET).abs() + config.LOSS_OFFSET
    
    # Apply softmax for entropy calculations
    probs = F.softmax(logits, dim=1)

    # Combined loss includes entropy and conditional entropy terms
    sum_loss = loss + get_entropy(probs) - get_cond_entropy(probs)
    return sum_loss[0] # Assuming sum_loss is a 1-element tensor

def get_loss(logits, label, criterion):
    """
    Calculates the training loss (standard CrossEntropyLoss with custom offset).
    """
    loss = criterion(logits.view(-1, config.NUM_CLASSES), label.view(-1))
    loss = (loss.float()).mean()
    loss = (loss - config.LOSS_OFFSET).abs() + config.LOSS_OFFSET
    return loss


def save_model(model_dict, best_metric, save_dir, save_prefix):
    """
    Saves the model state dictionary to the specified directory.
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = '{}.pt'.format(save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    print(f'Saving model to: {save_path_pt}')
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)
    print(f'Model saved: {save_prefix}')


def test_loader_eval(test_epitope, test_labels, batchsize, device, model):
    """
    Evaluates the model on a test set and returns various metrics.
    """
    model.eval() # Set model to evaluation mode
    correct = 0
    length = 0
    Result_list = [] # Stores raw model outputs (probabilities)
    labels_list = [] # Stores true labels
    predicted_list = [] # Stores predicted class labels

    test_loader = addbatch(test_epitope, test_labels, batchsize)

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for step, (epitope_inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            epitope_inputs = epitope_inputs.to(device)
            labels = labels.to(device)
            
            labels_list.append(labels.cpu())
            
            # Get model predictions
            Result, representation = model(epitope_inputs) # Result is probabilities after softmax
            Result_list.append(Result.detach().cpu().numpy())
            
            # Get predicted class
            _, predicted = torch.max(Result, 1)
            predicted_list.append(predicted.cpu().numpy())
            
            correct += (predicted.to(device) == labels.to(device)).sum().item()
            length += len(labels)
    
    Result_list = np.concatenate(Result_list)
    labels_list = np.concatenate(labels_list)
    predicted_list = np.concatenate(predicted_list)

    # Calculate metrics
    accuracy = 100 * correct / length
    
    # ROC AUC
    # Use Result_list[:, 1] for positive class probabilities
    fpr, tpr, _ = roc_curve(labels_list, Result_list[:, 1])
    auc_score = auc(fpr, tpr)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels_list, predicted_list)
    
    # F1 Score
    f1 = f1_score(labels_list, predicted_list)
    
    # Recall (Sensitivity)
    recall = recall_score(labels_list, predicted_list)
    
    # Precision
    precision = precision_score(labels_list, predicted_list)
    
    model.train() # Set model back to training mode
    
    return accuracy, mcc, f1, recall, precision, auc_score, Result_list, labels_list


def training(model, device, epochs, criterion, optimizer, traindata, val_epitope, val_labels, test_epitope, test_labels, patience=config.MAIN_MODEL_PATIENCE):
    """
    Main training loop for the PHbinder model.
    """
    running_loss = 0
    max_performance = 0 # Track best validation metric (accuracy in this case)
    early_stop_counter = 0
    best_model_state = None
    stop_training = False

    for epoch in range(epochs):
        if stop_training:
            break

        epoch_progress = tqdm(total=len(traindata), desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

        for step, (epitope_inputs, labels) in enumerate(traindata):
            model.train() # Ensure model is in training mode
            epitope_inputs = epitope_inputs.to(device)
            labels = labels.to(device)
            model = model.to(device) # Ensure model is on the correct device (redundant if already moved)
            
            optimizer.zero_grad() # Clear previous gradients
            
            outputs, _ = model(epitope_inputs) # Forward pass
            loss = get_val_loss(outputs, labels, criterion) # Calculate loss
            
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            
            running_loss += loss.item() # Accumulate loss for the epoch
            
            epoch_progress.update(1)
            epoch_progress.set_postfix(loss=running_loss / (step + 1))

        epoch_progress.close()

        # Evaluate on validation set
        val_acc, val_mcc, val_f1, val_recall, val_precision, val_auc, _, _ = test_loader_eval(
            val_epitope, val_labels, config.MAIN_MODEL_BATCH_SIZE, device, model
        )

        # Evaluate on test set
        test_acc, test_mcc, test_f1, test_recall, test_precision, test_auc, _, _ = test_loader_eval(
            test_epitope, test_labels, config.MAIN_MODEL_BATCH_SIZE, device, model
        )

        # Print validation and test set results
        print(f"Epoch {epoch + 1}:")
        print(f"Validation - Acc: {val_acc:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, AUC: {val_auc:.4f}")
        print(f"Test - Acc: {test_acc:.4f}, MCC: {test_mcc:.4f}, F1: {test_f1:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, AUC: {test_auc:.4f}")

        # Early stopping logic based on validation accuracy
        if val_acc > max_performance:
            print("Validation accuracy improved. Saving best model.")
            save_model(model.state_dict(), val_acc, config.SAVE_PATH_MAIN_MODEL_CHECKPOINTS, 'best_model_I')
            max_performance = val_acc
            early_stop_counter = 0
            best_model_state = model.state_dict() # Store the state dict of the best performing model
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered. No improvement in validation accuracy for {patience} epochs.")
                stop_training = True

        running_loss = 0 # Reset running loss for the next epoch

        if stop_training:
            break

    # Load the best model state if early stopping occurred and a better model was saved
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from training.")

    return model

