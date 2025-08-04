import argparse
import subprocess
import os
import sys

def run_command(command):
    """Runs a shell command, prints it, and checks for errors."""
    print(f"🚀 Running command: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, text=True)
        print("✅ Command completed successfully.\n")
    except FileNotFoundError:
        print(f"❌ Error: Command '{command[0]}' not found. Make sure the script exists and you have permissions.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}.")
        print(f"   Command: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"   stdout:\n{e.stdout}")
        if e.stderr:
            print(f"   stderr:\n{e.stderr}")
        sys.exit(1)

def find_generated_file(directory, prefix):
    """Finds the downloaded file that has an auto-generated suffix."""
    print(f"🔍 Searching for file with prefix '{prefix}' in '{directory}'...")
    try:
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith(".csv"):
                print(f"   Found file: {filename}")
                return os.path.join(directory, filename)
    except FileNotFoundError:
        print(f"❌ Error: The directory '{directory}' does not exist.")
        sys.exit(1)
    
    print(f"❌ Error: Could not find the generated CSV file for '{prefix}' in '{directory}'.")
    sys.exit(1)

def main():
    """Main function to run the stock data and news pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full stock processing, sentiment analysis, and training pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--stock_id",
        type=str,
        required=True,
        help="Stock ID / Ticker symbol to process (e.g., AAPL)."
    )
    args = parser.parse_args()

    stock_id_upper = args.stock_id.upper()
    
    # --- Step 1: Download Historical Price Data ---
    print("--- Step 1: Downloading Historical Price Data ---")
    # The download script will append a suffix like '_250td' to this base name
    base_price_output_path = f"data/raw/{stock_id_upper}_price.csv"
    download_price_command = [
        "python3", "data/download_price_data.py",
        "--ticker", stock_id_upper,
        "--output", base_price_output_path
    ]
    run_command(download_price_command)
    
    # --- Step 2: Download News Data ---
    print("--- Step 2: Downloading News Data ---")
    download_news_command = [
        "python3", "data/download_news.py",
        "--ticker", stock_id_upper
    ]
    run_command(download_news_command)

    # --- Step 3: Analyze News Sentiment ---
    print("--- Step 3: Analyzing News Sentiment ---")
    process_sentiment_command = [
        "python3", "data/process_sentiment.py",
        "--ticker", stock_id_upper
    ]
    run_command(process_sentiment_command)
    
    # --- Step 4: Process Combined Data ---
    print("--- Step 4: Processing Combined Price and Sentiment Data ---")
    # Use the helper function to find the actual filename created in Step 1
    raw_data_directory = "data/raw/"
    price_file_prefix = f"{stock_id_upper}_price_"
    actual_price_file = find_generated_file(raw_data_directory, price_file_prefix)
    
    process_command = [
        "python3", "data/process.py",
        "--input_csv", actual_price_file, # Pass the correct, full filename
        "--ticker", stock_id_upper
    ]
    run_command(process_command)

    # --- Step 5: Train Model ---
    print("--- Step 5: Training the Model ---")
    train_command = ["python3", "training/train.py"]
    run_command(train_command)

    # --- Step 6: Make a Prediction ---
    print("--- Step 6: Running Prediction ---")
    predict_command = ["python3", "predict.py"]
    run_command(predict_command)
    
    print("🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
