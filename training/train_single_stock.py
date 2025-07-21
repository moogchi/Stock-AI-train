import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from data.utils import clean_data, preprocess_features, split_data
from models.lstm_model import LSTMModel
from models.model_utils import save_model

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
INPUT_SIZE = 5  # Number of features (Open, High, Low, Close, Volume)
HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloader(X, y, batch_size):
    tensor_x = torch.tensor(X.values, dtype=torch.float32)
    tensor_y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)  # regression target
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_single_stock(data_path, model_save_path):
    print(f"Training on device: {DEVICE}")
    print(f"Loading data from: {data_path}")

    # Load stock CSV for single stock
    df = pd.read_csv(data_path)
    
    # Clean and preprocess (assuming your utils handle these)
    df = clean_data(df)
    df = preprocess_features(df)
    
    target_col = 'Close'
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    train_loader = create_dataloader(X_train[feature_cols], y_train, BATCH_SIZE)
    test_loader = create_dataloader(X_test[feature_cols], y_test, BATCH_SIZE)

    model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = inputs.unsqueeze(0)  # seq_len=1, adjust if sequences

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(1)

        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.6f}")

    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_single_stock.py <path_to_stock_csv> <path_to_save_model>")
        sys.exit(1)

    stock_csv_path = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.isfile(stock_csv_path):
        print(f"Error: CSV file '{stock_csv_path}' not found.")
        sys.exit(1)

    train_single_stock(stock_csv_path, model_path)
