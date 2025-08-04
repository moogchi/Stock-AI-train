import torch
import numpy as np
import os
import sys
import joblib
from torch.utils.data import TensorDataset, DataLoader

# --- Path and Model Setup ---
# Assumes this prediction script is in the project's root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
from models.lstm_model import LSTMModel

def forecast_future(model, initial_sequence, future_steps, scaler):
    """
    Makes multi-step future predictions with a more stable feedback loop.
    """
    model.eval()
    device = initial_sequence.device
    
    current_sequence_scaled = initial_sequence.clone().detach()
    future_predictions_scaled = []

    with torch.no_grad():
        for _ in range(future_steps):
            # 1. Get the next prediction (this is the scaled 'close' price)
            next_pred_scaled = model(current_sequence_scaled.unsqueeze(0)) # Shape: [1, 1]
            future_predictions_scaled.append(next_pred_scaled.item())

            # 2. Construct the next full feature vector for the next input
            # Strategy: Assume future O, H, L are the same as the predicted Close.
            # Assume Volume and Sentiment remain constant from the last known day.
            num_features = current_sequence_scaled.shape[1] # Should be 6
            next_step_features = torch.zeros(1, num_features, device=device)
            
            # Indices: 0:Open, 1:High, 2:Low, 3:Close, 4:Volume, 5:Sentiment
            # Assign the predicted 'close' price to all price-related features
            next_step_features[:, 0:4] = next_pred_scaled.item()
            
            # Carry the last known 'Volume' forward
            last_known_volume = current_sequence_scaled[-1, 4]
            next_step_features[:, 4] = last_known_volume
            
            # Carry the last known 'Sentiment' forward
            last_known_sentiment = current_sequence_scaled[-1, 5]
            next_step_features[:, 5] = last_known_sentiment
            
            # 3. Append the new predicted day and drop the oldest day
            current_sequence_scaled = torch.cat((current_sequence_scaled[1:], next_step_features), dim=0)

    # 4. Inverse transform all scaled predictions at once
    # The scaler was fitted on a single column ('Close'), so we reshape before transforming.
    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    actual_predictions = scaler.inverse_transform(future_predictions_scaled).flatten()
    
    return actual_predictions

if __name__ == "__main__":
    # --- Configuration ---
    # MODIFIED: Changed from 5 to 6 to include sentiment
    INPUT_DIM = 6
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    OUTPUT_DIM = 1

    # --- Paths ---
    # MODIFIED: Path points to the best model saved by the training script
    MODEL_PATH = os.path.join("models", "lstm_model_best.pth")
    DATA_DIR = os.path.join("data", "processed")
    SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")

    # --- Load Scaler and Model ---
    try:
        scaler = joblib.load(SCALER_PATH)
        device = torch.device("cpu") # Prediction is fast, CPU is fine for inference
        model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Model and scaler loaded, using device: {device}")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Please ensure the model and scaler exist at the correct paths by running the full pipeline.")
        sys.exit(1)

    # --- Load the very last sequence from the dataset to start forecasting ---
    try:
        X = np.load(os.path.join(DATA_DIR, "X.npy"))
        last_real_sequence = torch.from_numpy(X[-1]).float().to(device)
    except FileNotFoundError:
        print(f"❌ Error loading data file 'X.npy'. Please run process.py first.")
        sys.exit(1)
    
    # --- Make and Display Predictions ---
    print("\n--- Future Forecast ---")

    forecast_1_day = forecast_future(model, last_real_sequence, 1, scaler)
    print(f"Prediction for tomorrow: ${forecast_1_day[0]:.2f}")

    forecast_1_week = forecast_future(model, last_real_sequence, 7, scaler)
    print(f"\nPrediction for the next 7 days: {[f'${p:.2f}' for p in forecast_1_week]}")

    forecast_1_month = forecast_future(model, last_real_sequence, 30, scaler)
    print(f"\nPrediction for the next 30 days: {[f'${p:.2f}' for p in forecast_1_month]}")
