import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse
import joblib
import sys

def process_raw_csv(input_csv, ticker, output_dir, sequence_length=30):
    # --- 1. Load and Clean Price Data ---
    try:
        col_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df_price = pd.read_csv(input_csv, header=None, names=col_names, skiprows=3)
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: The input price file was not found at {input_csv}")
        sys.exit(1)

    df_price['Date'] = pd.to_datetime(df_price['Date'], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_price[col] = pd.to_numeric(df_price[col], errors='coerce')

    df_price.dropna(inplace=True)
    df_price.sort_values('Date', inplace=True)
    
    if df_price.empty:
        print("❌ FATAL ERROR: Price DataFrame is empty after cleaning. Check the input CSV format.")
        sys.exit(1)
        
    print(f"✅ Loaded and cleaned {len(df_price)} rows of price data.")

    # --- 2. Load and Merge Sentiment Data ---
    sentiment_file = os.path.join(f"processed_news/{ticker}_sentiment.csv") # Assumes this path
    try:
        df_sentiment = pd.read_csv(sentiment_file)
        df_sentiment['Date'] = pd.to_datetime(df_sentiment['publishedAt'])
        
        # Merge the two dataframes on the 'Date' column
        df = pd.merge(df_price, df_sentiment[['Date', 'sentiment']], on='Date', how='left')
        
        # Fill days with no news with a neutral sentiment of 0
        df['sentiment'].fillna(0, inplace=True)
        print(f"✅ Merged sentiment data from {sentiment_file}")

    except FileNotFoundError:
        print(f"⚠️ WARNING: Sentiment file not found at {sentiment_file}. Proceeding without sentiment data.")
        df = df_price
        df['sentiment'] = 0 # Add a neutral sentiment column if no news is found

    # --- 3. Scale the 'Close' price and save the scaler ---
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_close'] = close_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(close_scaler, scaler_path)
    print(f"✅ 'Close' price scaler saved to {scaler_path}")

    # --- 4. Scale all features for the input sequences (X) ---
    # ADDED 'sentiment' to the feature columns
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment']
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = features_scaler.fit_transform(df[feature_cols])
    print("✅ Scaled all features (including sentiment) for model input.")

    # --- 5. Create Sequences ---
    if len(df) <= sequence_length:
        print(f"❌ FATAL ERROR: Not enough data ({len(df)} rows) to create sequences of length {sequence_length}.")
        sys.exit(1)

    X, y = [], []
    input_features = df[feature_cols].values
    target_feature = df['scaled_close'].values
    
    for i in range(len(df) - sequence_length):
        X.append(input_features[i:(i + sequence_length)])
        y.append(target_feature[i + sequence_length])
        
    X, y = np.array(X), np.array(y)
    
    # --- 6. Save Final Data ---
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    
    print(f"Processed into {len(X)} sequences of length {sequence_length}.")
    print(f"✅ Saved X.npy and y.npy to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess stock CSV and sentiment data to training-ready numpy arrays.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to raw stock price CSV file")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker (e.g., AAPL) to find the corresponding sentiment file.")
    parser.add_argument("--output_dir", type=str, default="data/processed/", help="Directory to save processed files")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length for training data")
    args = parser.parse_args()
    
    process_raw_csv(
        input_csv=args.input_csv,
        ticker=args.ticker,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length
    )
