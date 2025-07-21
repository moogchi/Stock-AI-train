import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import re

def load_and_clean(filepath):
    # Not used in main flow but kept if needed
    df = pd.read_csv(filepath)
    df = df.dropna().drop_duplicates()
    return df

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df_scaled = df.copy()

    # Filter out only numeric columns from feature_cols
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric columns found in feature_cols. Got: " + str(feature_cols))

    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler

def create_sequences(df, feature_cols, target_col, sequence_length=10):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        features_seq = df[feature_cols].iloc[i:i+sequence_length].values
        target_val = df[target_col].iloc[i+sequence_length]
        X.append(features_seq)
        y.append(target_val)
    return np.array(X), np.array(y)

def save_sequences(X, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

def extract_sequence_length_from_filename(filename):
    match = re.search(r"(\d+)td", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Filename must include trading day count like '252td'.")

def process_raw_csv(input_csv, output_dir, sequence_length=30):
    # Skip rows 1 and 2 (Ticker, Date rows), keep first row as header
    df = pd.read_csv(input_csv, skiprows=[1,2])

    # Rename columns: first col is Date, rest are prices/volume
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Drop NA and duplicates
    df = df.dropna().drop_duplicates()

    feature_cols = ['Open', 'High', 'Low', 'Volume']
    target_col = 'Close'

    df_scaled, _ = scale_features(df, feature_cols + [target_col])

    X, y = create_sequences(df_scaled, feature_cols, target_col, sequence_length)
    save_sequences(X, y, output_dir)

    print(f"Processed {len(df)} rows into {len(X)} sequences of length {sequence_length}.")
    print(f"Saved sequences to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stock CSV to training-ready numpy arrays.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw stock CSV file")
    parser.add_argument("--output_dir", type=str, default="data/processed/", help="Directory to save X.npy and y.npy")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length for training data")
    args = parser.parse_args()

    process_raw_csv(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length
    )
