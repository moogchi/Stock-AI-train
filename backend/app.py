# app.py
# Place this file inside the 'backend' directory.
#
# To run this:
# 1. Make sure you have Python installed in your environment.
# 2. Install the necessary libraries for this backend script:
#    pip install fastapi uvicorn python-multipart "yfinance[optional]" pandas matplotlib
# 3. Ensure your 'ai_train' environment has all its dependencies (torch, etc.).
# 4. From inside the 'backend' directory, run the server:
#    uvicorn app:app --reload

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import os
import sys
import re
import logging
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from datetime import timedelta

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Stock Prediction AI Backend",
    description="A backend service that runs a custom PyTorch LSTM pipeline to forecast stock prices.",
    version="2.0.0",
)

# --- Helper Functions ---

def parse_pipeline_output(output: str):
    """
    Parses the stdout from the training/prediction pipeline to extract forecast data.
    """
    try:
        logger.info("Parsing pipeline output...")
        
        # Regex to find the prediction values
        tomorrow_regex = r"Prediction for tomorrow: \$([\d\.]+)"
        week_regex = r"Prediction for the next 7 days: \['\$([\d\.\s,']+)'\]"
        month_regex = r"Prediction for the next 30 days: \['\$([\d\.\s,']+)'\]"

        # Find matches
        tomorrow_match = re.search(tomorrow_regex, output)
        week_match = re.search(week_regex, output)
        month_match = re.search(month_regex, output)

        if not (tomorrow_match and week_match and month_match):
            logger.error("Could not parse all required predictions from the output.")
            raise ValueError("Failed to parse prediction data from pipeline log.")

        # Extract and clean data
        tomorrow = float(tomorrow_match.group(1))
        
        week_str = week_match.group(1).replace("'", "").replace("$", "")
        week = [float(p.strip()) for p in week_str.split(',')]

        month_str = month_match.group(1).replace("'", "").replace("$", "")
        month = [float(p.strip()) for p in month_str.split(',')]
        
        logger.info("Successfully parsed all predictions.")
        return {"tomorrow": tomorrow, "week": week, "month": month}

    except Exception as e:
        logger.error(f"Error parsing pipeline output: {e}")
        raise ValueError(f"Could not parse pipeline output. Error: {e}")


def generate_plot_base64(ticker: str, forecast_data: dict) -> str:
    """
    Generates a plot of historical data and the new forecast, returning a base64 string.
    """
    try:
        logger.info(f"Generating plot for {ticker}...")
        # 1. Fetch historical data for context (last 90 days)
        hist_data = yf.Ticker(ticker).history(period="90d")
        if hist_data.empty:
            logger.warning("Could not fetch historical data for plot.")
            return None
        
        last_hist_date = hist_data.index.max()
        last_hist_close = hist_data.loc[last_hist_date]['Close']

        # 2. Create future dates for the forecast
        future_dates = pd.to_datetime([last_hist_date + timedelta(days=i) for i in range(1, 31)])
        
        # 3. Create a pandas Series for the forecast
        # We prepend the last known closing price to the forecast to create a continuous line
        forecast_values = [last_hist_close] + forecast_data['month']
        forecast_dates = pd.to_datetime([last_hist_date] + list(future_dates))

        forecast_series = pd.Series(forecast_values, index=forecast_dates)

        # 4. Generate plot with Matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(hist_data.index, hist_data['Close'], label='Historical Close Price', color='royalblue', lw=2)
        ax.plot(forecast_series.index, forecast_series.values, label='Forecasted Price', color='darkorange', linestyle='--', marker='o', markersize=4)
        
        ax.set_title(f'Stock Price Forecast for {ticker.upper()}', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 5. Save plot to a bytes buffer and encode in base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        logger.info("Plot generated and encoded successfully.")
        return base64_image

    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        return None


# --- API Endpoint ---
@app.post("/predict/{ticker}")
async def run_prediction_pipeline(ticker: str):
    """
    Triggers the full AI pipeline for a given stock ticker.
    This involves running an external script that handles data download,
    processing, model training, and prediction.
    """
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    logger.info(f"Received request to run full pipeline for ticker: {ticker}")
    
    # Define the path to the ai_train directory and the run.py script
    # This assumes 'backend' and 'ai_train' are sibling directories
    ai_train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_train'))
    run_script_path = os.path.join(ai_train_dir, 'run.py')

    if not os.path.exists(run_script_path):
        logger.error(f"Could not find run.py at {run_script_path}")
        raise HTTPException(status_code=500, detail="Server configuration error: cannot find the training pipeline script.")

    try:
        # The command to execute the pipeline
        command = [sys.executable, run_script_path, "--stock_id", ticker]
        
        logger.info(f"Executing command: {' '.join(command)}")
        
        # Run the subprocess. We set the working directory to ai_train_dir
        # so all relative paths in your scripts work correctly.
        process = subprocess.run(
            command,
            cwd=ai_train_dir,
            capture_output=True,
            text=True,
            check=True, # This will raise CalledProcessError if the script fails
            timeout=600 # 10-minute timeout for the whole pipeline
        )
        
        full_log = process.stdout
        logger.info("Pipeline script executed successfully.")
        
        # Parse the output to get prediction data
        forecast_data = parse_pipeline_output(full_log)
        
        # Generate the plot
        plot_b64 = generate_plot_base64(ticker, forecast_data)

        # Create the final JSON response
        response_data = {
            "ticker": ticker,
            "forecast": forecast_data,
            "plot_base64": plot_b64,
            "full_log": full_log
        }
        
        return JSONResponse(content=response_data)

    except subprocess.CalledProcessError as e:
        # This error is raised if the script returns a non-zero exit code (i.e., it failed)
        error_log = f"Pipeline script failed with exit code {e.returncode}.\n"
        error_log += f"--- STDOUT ---\n{e.stdout}\n"
        error_log += f"--- STDERR ---\n{e.stderr}\n"
        logger.error(error_log)
        raise HTTPException(status_code=500, detail={"message": "The AI pipeline failed during execution.", "log": error_log})
    
    except subprocess.TimeoutExpired as e:
        error_log = f"Pipeline script timed out after {e.timeout} seconds.\n"
        error_log += f"--- STDOUT ---\n{e.stdout}\n"
        error_log += f"--- STDERR ---\n{e.stderr}\n"
        logger.error(error_log)
        raise HTTPException(status_code=500, detail={"message": "The AI pipeline timed out.", "log": error_log})

    except ValueError as e:
        # This catches the parsing error
        raise HTTPException(status_code=500, detail={"message": str(e), "log": process.stdout})

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API. Use the /predict/{ticker} endpoint to run the pipeline."}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # This allows the script to be run directly with `python app.py` from the 'backend' folder
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
