import pandas as pd
from transformers import pipeline
import os
import argparse
import sys

def analyze_sentiment(csv_path):
    """Analyzes sentiment of news headlines and returns daily scores."""
    try:
        news_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: News file not found at {csv_path}")
        sys.exit(1)
        
    if news_df.empty:
        print("⚠️ Warning: News file is empty. No sentiment to analyze.")
        return pd.DataFrame()

    # Load a pre-trained sentiment analysis model
    print("Loading sentiment analysis model...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    print(f"Analyzing sentiment for {len(news_df)} headlines...")
    results = sentiment_pipeline(news_df['title'].tolist())
    
    news_df['sentiment'] = [
        (res['score'] if res['label'] == 'POSITIVE' else -res['score']) 
        for res in results
    ]
    
    # Average the sentiment for each day
    daily_sentiment = news_df.groupby('publishedAt')['sentiment'].mean().reset_index()
    daily_sentiment['publishedAt'] = pd.to_datetime(daily_sentiment['publishedAt'])
    
    return daily_sentiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze sentiment of stock news headlines.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to process (e.g., AAPL).")
    args = parser.parse_args()

    TICKER = args.ticker.upper()
    news_csv_path = f"raw_news/{TICKER}_news.csv"
    
    sentiment_df = analyze_sentiment(news_csv_path)
    
    if not sentiment_df.empty:
        # --- FIX: Create the output directory if it doesn't exist ---
        output_dir = "processed_news"
        os.makedirs(output_dir, exist_ok=True)
        # --- END FIX ---

        output_path = f"{output_dir}/{TICKER}_sentiment.csv"
        sentiment_df.to_csv(output_path, index=False)
        print(f"✅ Saved daily sentiment scores to {output_path}")

