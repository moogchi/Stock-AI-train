import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse # Import argparse

def get_stock_news(api_key, ticker, days_ago=30):
    """Fetches news for a specific stock ticker within a defined date range."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    
    from_date_str = start_date.strftime('%Y-%m-%d')
    to_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching news for '{ticker}' from {from_date_str} to {to_date_str}...")
    
    url = (f'https://newsapi.org/v2/everything?'
           f'q={ticker}&'
           f'from={from_date_str}&'
           f'to={to_date_str}&'
           f'sortBy=publishedAt&'
           f'apiKey={api_key}')
           
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == 'ok':
        articles_df = pd.DataFrame(data['articles'])
        if not articles_df.empty:
            articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt']).dt.date
            return articles_df[['publishedAt', 'title']]
        else:
            print("No articles found for the given period.")
            return pd.DataFrame()
    else:
        print(f"Error fetching news: {data.get('message')}")
        return pd.DataFrame()

if __name__ == '__main__':
    # --- MODIFIED: Use argparse to get the ticker from the command line ---
    parser = argparse.ArgumentParser(description="Download news for a specific stock ticker.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to fetch news for (e.g., AAPL).")
    args = parser.parse_args()

    # Your hardcoded API key
    API_KEY = " 
    TICKER = args.ticker.upper()
    
    news_df = get_stock_news(API_KEY, TICKER, days_ago=30)
    
    if not news_df.empty:
        output_dir = "data/raw_news"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/{TICKER}_news.csv"
        news_df.to_csv(output_path, index=False)
        print(f"Saved {len(news_df)} news articles to {output_path}")
