import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle


def create_data_loaders(train, val, vars):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train_df = train[vars].copy()
    x_val_df = val[vars].copy()
   
    # Extract and remove the label and weight from the copied dataframes
    label_train_df = x_train_df.pop('signal_label')
    label_val_df = x_val_df.pop('signal_label')
    weights_train = x_train_df.pop('weightNorm')
    weights_val = x_val_df.pop('weightNorm')

    # Now x_train_df and x_val_df only contain the input features
    x_train = x_train_df.to_numpy()
    x_val = x_val_df.to_numpy()

    label_train = label_train_df.to_numpy().astype(float)
    label_val = label_val_df.to_numpy().astype(float)
    weights_train = weights_train.to_numpy().astype(float)
    weights_val = weights_val.to_numpy().astype(float)


    # Apply StandardScaler to the data
    scaler = StandardScaler()

    # We fit on the training data and transform on train, val and test data
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    # x_test = scaler.transform(x_test)

    # Convert numpy arrays to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    label_train = torch.tensor(label_train, dtype=torch.float32).unsqueeze(1)
    label_val = torch.tensor(label_val, dtype=torch.float32).unsqueeze(1)
    weights_train = torch.tensor(weights_train, dtype=torch.float32)
    weights_val = torch.tensor(weights_val, dtype=torch.float32)

    


    # Create datasets
    train_dataset = TensorDataset(x_train, label_train, weights_train)
    val_dataset = TensorDataset(x_val, label_val, weights_val)

    # Create DataLoaders
    batch_size = 640  # You can adjust the batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, scaler

def create_test_loader(test, vars, scaler, batch_size=640):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_test_df = test[vars].copy()
   
    # Extract and remove the label and weight from the copied dataframes
    label_test_df = x_test_df.pop('signal_label')
    weights_test = x_test_df.pop('weightNorm')

    # Now x_test_df only contains the input features
    x_test = x_test_df.to_numpy()

    x_test = scaler.transform(x_test)

    label_test = label_test_df.to_numpy().astype(float)
    weights_test = weights_test.to_numpy().astype(float)

    # Convert numpy arrays to PyTorch tensors
    x_test = torch.tensor(x_test, dtype=torch.float32)
    label_test = torch.tensor(label_test, dtype=torch.float32).unsqueeze(1)
    weights_test = torch.tensor(weights_test, dtype=torch.float32)

    # Create datasets
    test_dataset = TensorDataset(x_test, label_test, weights_test)

    # Create DataLoaders
    batch_size = batch_size  # You can adjust the batch size as needed
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader

def train_model(train_loader, val_loader, model, optimizer, criterion, device, save_path, save_name, epochs=100000, patience=10):

    best_val_loss=float('inf')
    patience_counter=0

    model.train()
    for epoch in tqdm(range(epochs), desc = 'training epochs', disable= True):
        running_loss=0.0

        for inputs, labels, weights in train_loader:
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)

            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs, labels)

            loss = (loss * weights).mean()
            # loss = (loss * weights)
            loss.backward()
            optimizer.step()

        correct_predictions_weighted=0.0
        total_weights=0.0
        model.eval()
        with torch.no_grad():
            total_loss=0.0
            for inputs, labels, weights in val_loader:
                inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                outputs=model(inputs)
                loss=criterion(outputs, labels)
                loss = (loss * weights).mean()
                total_loss += loss.item()

                preds = torch.round(outputs)
                correct_predictions_weighted += (preds == labels).float().mul(weights).sum().item()
                total_weights += weights.sum().item()
            val_loss = total_loss / len(val_loader)
            val_acc_weighted = correct_predictions_weighted / total_weights

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), save_path + save_name + '.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping, no improvement for {} epochs'.format(patience))
                    break

            print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Weighted Validation Accuracy: {val_acc_weighted:.4f}')

    return model

def test_model(test_loader, model, criterion, device):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        total_loss=0.0
        total_weights = 0.0
        correct_predictions_weighted = 0.0

        for inputs, labels, weights in test_loader:
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            outputs=model(inputs)
            loss= criterion(outputs, labels)
            loss = (loss * weights).mean()
            total_loss += loss.item()

            preds = torch.round(outputs)
            correct_predictions_weighted += (preds == labels).float().mul(weights).sum().item()
            total_weights += weights.sum().item()

        test_loss = total_loss / len(test_loader)
        test_acc_weighted = correct_predictions_weighted / total_weights

        print(f'Test Loss: {test_loss:.4f}, Weighted Test Accuracy: {test_acc_weighted:.4f}')

    return test_loss, test_acc_weighted