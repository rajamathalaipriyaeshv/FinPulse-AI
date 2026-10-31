import os
import traceback
import concurrent.futures
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.services import DataPipeline

# Global instance for the pipeline
pipeline_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    print("🚀 Starting FinPulse AI Engine...")
    
    # Initializes the pipeline (Make sure your local model path is set inside DataPipeline!)
    pipeline_instance = DataPipeline()
    print("✅ Engine online and ready.")
    
    yield # App runs here
    
    # Cleanup when shutting down
    print("🛑 Shutting down engine...")
    pipeline_instance = None

app = FastAPI(title="FinPulse AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/analyze")
def analyze_ticker(ticker: str = Query("BTC-USD"), interval: str = Query("1d")):
    global pipeline_instance
    if not pipeline_instance:
        raise HTTPException(status_code=500, detail="Data Pipeline is not initialized.")

    try:
        clean_keyword = ticker.split("-")[0].lower()
        if clean_keyword == "btc": clean_keyword = "bitcoin"
        elif clean_keyword == "eth": clean_keyword = "ethereum"
        elif clean_keyword == "sol": clean_keyword = "solana"
        elif clean_keyword == "bnb": clean_keyword = "binancecoin"
        elif clean_keyword == "xrp": clean_keyword = "ripple"

        # Determine yfinance period/interval mapping based on the incoming interval
        yf_period = "30d"
        yf_interval = "1d"
        if interval == "1H":
            yf_period = "7d"
            yf_interval = "1h"
        elif interval == "24H":
            yf_period = "30d"
            yf_interval = "1d"
        elif interval == "7D":
            yf_period = "1y"
            yf_interval = "1wk"
        elif interval == "1M":
            yf_period = "5y"
            yf_interval = "1mo"

        # 1. Fetch Data Concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_df = executor.submit(pipeline_instance.fetch_market_data, ticker, yf_period, yf_interval)
            future_news = executor.submit(pipeline_instance.fetch_news_and_rss, clean_keyword)
            future_fg = executor.submit(pipeline_instance.fetch_fear_greed_index)
            future_gas = executor.submit(pipeline_instance.fetch_etherscan_gas)
            
            df = future_df.result()
            headlines, sentiment_score = future_news.result()
            fear_greed_score = future_fg.result()
            gas_metric = future_gas.result()
        
        # 2. Run Predictions
        prediction = pipeline_instance.run_prediction(
            df=df, sentiment_score=sentiment_score, fear_greed=fear_greed_score, gas_metric=gas_metric
        )

        metrics_payload = {
            "current_price": prediction["current_price"],
            "predicted_price": prediction["predicted_price"],
            "expected_change_pct": prediction["expected_change_pct"],
            "direction": prediction["direction"],
            "sentiment_score": round(sentiment_score, 3),
            "fear_greed_score": round(fear_greed_score, 3),
            "gas_metric": round(gas_metric, 3)
        }

        # 3. Generate Report
        analyst_report = pipeline_instance.generate_llm_analyst_report(
            ticker=ticker, metrics=metrics_payload, headlines=headlines
        )

        # 4. Format Chart Data
        # Depending on interval, format dates properly
        date_format = '%Y-%m-%d %H:%M' if interval == "1H" else '%Y-%m-%d'
        # Get up to 30 points for better graph visuals
        historical_dates = df.index.strftime(date_format).tolist()[-30:]
        historical_prices = df['Close'].tail(30).round(2).tolist()

        return {
            "ticker": ticker,
            "metrics": metrics_payload,
            "headlines": headlines[:5],
            "analyst_report": analyst_report,
            "chart_data": {
                "labels": historical_dates + ["Predicted"],
                "prices": historical_prices + [prediction["predicted_price"]]
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure to run this via the terminal for best results rather than executing the file directly.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)