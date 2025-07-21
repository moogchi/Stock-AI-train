import yfinance as yf
import argparse
import os
from datetime import datetime

def download_stock_data(ticker: str, start_date: str, output_csv: str):
    # Use today's date as the end date
    end_date = datetime.today().strftime("%Y-%m-%d")
    print(f"Downloading {ticker} data from {start_date} to {end_date}...")
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data downloaded. Check your ticker or date range.")
        return
    
    # Calculate number of days in the requested date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (end_dt - start_dt).days
    
    # Modify output filename to include number of days
    base_dir = os.path.dirname(output_csv)
    base_name = os.path.basename(output_csv)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_{num_days}d{ext}"
    new_path = os.path.join(base_dir, new_name)
    
    # Make sure output directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Save to CSV with new filename
    data.to_csv(new_path)
    print(f"Saved raw CSV to {new_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data and save as CSV")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="data/processed/stock_raw.csv", help="Output CSV filepath")
    
    args = parser.parse_args()
    
    download_stock_data(args.ticker, args.start, args.output)



#running on terminal
# python download_stock.py --ticker (The stock you want) --start (year-month-day) --output (file directory)
# Ex) python download_stock.py --ticker AAPL --start 2025-01-01 --output raw/aapl_2025.csv