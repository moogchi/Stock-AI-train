import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

# --- Path Setup ---
# This allows the script to be run from the root directory and find the 'models' module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.lstm_model import LSTMModel

# --- Main Training Block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Model parameters
    # MODIFIED: Changed from 5 to 6 to include the new 'sentiment' feature
    INPUT_DIM = 6
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    OUTPUT_DIM = 1

    # Training parameters
    NUM_EPOCHS = 300  # A high number, early stopping will find the optimal point
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 5 # Number of epochs to wait for improvement before stopping

    # --- File Paths ---
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_model.pth")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- Device Selection ---
    try:
        import intel_extension_for_pytorch as ipex
        device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    except ImportError:
        print("⚠️ WARNING: Intel Extension for PyTorch (IPEX) not found. Defaulting to CPU.")
        device = torch.device("cpu")

    print(f"✅ Using device: {device}")

    # --- Data Loading ---
    try:
        X = np.load(os.path.join(DATA_DIR, "X.npy"))
        y = np.load(os.path.join(DATA_DIR, "y.npy"))
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Data files not found in {DATA_DIR}. Please run process.py first.")
        sys.exit(1)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize Model, Optimizer, and Loss ---
    model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Optimize with IPEX ---
    if "ipex" in sys.modules and device.type == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print("✅ Model optimized with Intel Extension for PyTorch.")

    # --- Early Stopping Setup ---
    best_val_loss = float('inf')
    patience_counter = 0

    # --- Training Loop ---
    print("\n🚀 Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                total_val_loss += criterion(outputs, y_batch).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1:03}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", end="")

        # --- Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
            print(" <-- ✅ Validation loss improved, saving model.")
        else:
            patience_counter += 1
            print(f" <-- ⚠️ No improvement. Counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\n🛑 Early stopping triggered. No improvement in validation loss.")
                break # Exit the training loop

    print(f"\n🎉 Training complete. Best model saved to {BEST_MODEL_PATH}")
