import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna().drop_duplicates()
    return df

def scale_features(df, feature_cols):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
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

def process_raw_csv(
    input_csv,
    output_dir="data/processed/",
    sequence_length=10,
    feature_cols=['Open', 'High', 'Low', 'Volume', 'Close'],
    target_col='Close'
):
    df = load_and_clean(input_csv)
    df_scaled, _ = scale_features(df, feature_cols + [target_col])
    X, y = create_sequences(df_scaled, feature_cols, target_col, sequence_length)
    save_sequences(X, y, output_dir)
    print(f"✅ Processed {X.shape[0]} sequences and saved to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stock CSV to training-ready numpy arrays.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw stock CSV file")
    parser.add_argument("--output_dir", type=str, default="data/processed/", help="Directory to save X.npy and y.npy")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length (number of days)")
    args = parser.parse_args()

    process_raw_csv(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        sequence_length=args.seq_len
    )
