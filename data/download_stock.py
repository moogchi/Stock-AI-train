import yfinance as yf
import argparse
import os
from datetime import datetime, timedelta

def download_stock_data(ticker: str, output_csv: str):
    # Set date range: past 365 calendar days
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} data from {start_str} to {end_str}...")
    
    data = yf.download(ticker, start=start_str, end=end_str)
    
    if data.empty:
        print("No data downloaded. Check your ticker or internet connection.")
        return
    
    # Count the actual number of trading days (i.e., rows)
    trading_days = len(data)
    
    # Update output filename to include trading day count
    base_dir = os.path.dirname(output_csv)
    base_name = os.path.basename(output_csv)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_{trading_days}td{ext}"  # 'td' = trading days
    new_path = os.path.join(base_dir, new_name)
    
    # Ensure output directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Save to CSV
    data.to_csv(new_path)
    print(f"Saved CSV with {trading_days} trading days to {new_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 1 calendar year of stock data and save as CSV")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--output", type=str, default="data/processed/stock_raw.csv", help="Output CSV filepath")
    
    args = parser.parse_args()
    
    download_stock_data(args.ticker, args.output)


#running on terminal
# python download_stock.py --ticker (The stock you want) --output (file directory)
# Ex) python download_stock.py --ticker AAPL --output raw/aapl.csv
