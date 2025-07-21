import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Intel PyTorch Extension for better Intel Arc GPU performance
try:
    import intel_pytorch_extension as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("Intel PyTorch Extension (IPEX) not found, running without Intel GPU acceleration.")

from data.utils import load_raw_data, clean_data, preprocess_features, split_data
from models.lstm_model import LSTMModel
from models.model_utils import save_model

DATA_PATH = 'data/processed/stock_processed.csv'
MODEL_SAVE_PATH = 'models/trained_lstm.pth'

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
INPUT_SIZE = 5
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# Intel Arc GPU typically uses CPU backend or "xpu" device if supported.
# We'll default to CPU for now, and use IPEX if available.
DEVICE = torch.device("cpu")

def create_dataloader(X, y, batch_size):
    tensor_x = torch.tensor(X.values, dtype=torch.float32)
    tensor_y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train():
    print(f"Training on device: {DEVICE}")

    df = pd.read_csv(DATA_PATH)
    
    target_col = 'Close'
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)

    train_loader = create_dataloader(X_train[feature_cols], y_train, BATCH_SIZE)
    test_loader = create_dataloader(X_test[feature_cols], y_test, BATCH_SIZE)

    model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # If IPEX is available, optimize model and optimizer for Intel Arc GPU / CPU
    if IPEX_AVAILABLE:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        print("Model optimized with Intel PyTorch Extension (IPEX).")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # LSTM expects (seq_len, batch, input_size)
            inputs = inputs.unsqueeze(0)  # seq_len=1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(1)

        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.6f}")

    save_model(model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
