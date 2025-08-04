# streamlit_app.py
#
# To run this:
# 1. Make sure your backend (app.py) is running on your server at 50.47.144.52.
# 2. Install the necessary libraries for this frontend script:
#    pip install streamlit requests pandas matplotlib yfinance
# 3. From your terminal, run the app:
#    streamlit run streamlit_app.py

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
import io
import base64

# --- Configuration ---
# Updated to point to your new server IP address
BACKEND_URL = "http://50.47.144.52:8000/predict/"

# --- Helper Functions ---

def call_prediction_api(stock_id):
    """Calls the backend API to run the full pipeline and get predictions."""
    api_endpoint = f"{BACKEND_URL}{stock_id}"
    
    try:
        # Use st.spinner for a loading animation during the long API call
        with st.spinner('The AI pipeline is running on the remote server... This may take a few minutes.'):
            # Set a long timeout because the backend process can take a while
            response = requests.post(api_endpoint, timeout=600) # 10 minute timeout

        # Check if the request was successful
        if response.status_code == 200:
            st.success("✅ AI Pipeline Completed!")
            return response.json()
        else:
            st.error(f"❌ API Error: Backend returned status code {response.status_code}.")
            try:
                # Try to show the detailed error from the backend's JSON response
                st.json(response.json())
            except requests.exceptions.JSONDecodeError:
                # If the response isn't JSON, show the raw text
                st.text(response.text)
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network Error: Could not connect to the backend at {BACKEND_URL}.")
        st.write("Please ensure the backend container/script is running and accessible at that address.")
        st.code(str(e))
        return None

def generate_forecast_plot(ticker: str, forecast_data: list):
    """
    Generates a plot of the last 30 days of historical data and the 30-day forecast.
    """
    try:
        # 1. Fetch last 30 days of historical data
        hist_data = yf.Ticker(ticker).history(period="30d")
        if hist_data.empty:
            st.warning("Could not fetch historical data to generate the plot.")
            return None
        
        last_hist_date = hist_data.index.max()
        last_hist_close = hist_data.loc[last_hist_date]['Close']

        # 2. Prepare forecast data and dates
        # Prepend the last known closing price to the forecast to create a continuous line
        forecast_values = [last_hist_close] + forecast_data
        # Create future dates starting from the last historical date
        forecast_dates = pd.to_datetime([last_hist_date + timedelta(days=i) for i in range(len(forecast_values))])

        # 3. Generate plot with Matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot historical data
        ax.plot(hist_data.index, hist_data['Close'], label='Historical Price (Last 30 Days)', color='royalblue', lw=2)
        
        # Plot forecast data with a lighter, dashed line
        ax.plot(forecast_dates, forecast_values, label='30-Day Forecast', color='darkorange', linestyle='--', marker='o', markersize=4, alpha=0.7)
        
        ax.set_title(f'30-Day Price Forecast for {ticker.upper()}', fontsize=16, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"An error occurred while generating the plot: {e}")
        return None


# --- Streamlit App UI ---
st.set_page_config(page_title="Stock Prediction AI", layout="wide")
st.title("📈 Stock Prediction AI")
st.write("Enter a stock ticker to trigger the backend AI pipeline. The process involves fetching new data, training the model, and making a forecast, so it may take several minutes to complete.")

# --- Input Section ---
col1, col2 = st.columns([1, 3])
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    predict_button = st.button("Predict", type="primary")

# --- Prediction Execution ---
if predict_button:
    if stock_ticker:
        # Call the API and get the results
        api_result = call_prediction_api(stock_ticker)
        
        if api_result and "forecast" in api_result:
            forecast = api_result.get("forecast", {})
            
            st.subheader("📊 Forecast Results")
            
            # --- Display 7-Day Prediction ---
            if 'week' in forecast:
                st.write("**Prediction for the next 7 days:**")
                # Format the weekly forecast into a nice table-like display
                week_data = {"Day": [f"Day {i+1}" for i in range(7)], "Predicted Price": [f"${p:,.2f}" for p in forecast['week']]}
                st.dataframe(pd.DataFrame(week_data), use_container_width=True)

            # --- Display 30-Day Graph ---
            if 'month' in forecast:
                st.write("**Graph of Historical Price and 30-Day Forecast:**")
                forecast_fig = generate_forecast_plot(stock_ticker, forecast['month'])
                if forecast_fig:
                    st.pyplot(forecast_fig)
            
            # Optionally, show the full log output from the backend in an expander
            if "full_log" in api_result:
                with st.expander("Show Full Backend Log"):
                    st.code(api_result["full_log"], language='bash')
    else:
        st.warning("Please enter a stock ticker.")
