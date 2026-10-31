import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class BiLSTMForecaster(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super(BiLSTMForecaster, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return out


class DataPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(BASE_DIR, "finbert_model")
        
        # Load model & tokenizer strictly from local files
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)
        
        self.sentiment_engine = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        
        # Initialize Groq client cleanly
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        
        # Simple TTL cache for APIs to improve efficiency and durability
        self._cache = {}
        self._cache_ttl = 300 # 5 minutes

    def _get_cached(self, key):
        import time
        if key in self._cache:
            val, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return val
        return None

    def _set_cached(self, key, value):
        import time
        self._cache[key] = (value, time.time())

    def fetch_market_data(self, ticker: str = "BTC-USD", period: str = "30d", interval: str = "1d") -> pd.DataFrame:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No price data retrieved for ticker: {ticker}")
        
        # Ensure single-level columns if MultiIndex returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['MA7'] = df['Close'].rolling(window=7).mean().bfill()
        df['Vol_Change'] = df['Volume'].pct_change().fillna(0)
        return df

    def fetch_news_and_rss(self, keyword: str = "bitcoin") -> tuple[list[str], float]:
        cache_key = f"news_{keyword}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        headlines = []
        
        # Stream A: RSS Feed
        try:
            res = requests.get("https://cointelegraph.com/rss", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('./channel/item')[:5]:
                    title_elem = item.find('title')
                    if title_elem is not None and title_elem.text:
                        headlines.append(title_elem.text)
        except Exception:
            pass

        # Stream B: NewsAPI
        if NEWS_API_KEY:
            try:
                url = f"https://newsapi.org/v2/everything?q={keyword}&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
                res = requests.get(url, timeout=5).json()
                if res.get("status") == "ok":
                    for article in res.get("articles", []):
                        if article.get("title"):
                            headlines.append(article["title"])
            except Exception:
                pass

        sentiment_score = 0.0
        if headlines:
            try:
                results = self.sentiment_engine(headlines)
                scores = []
                for res in results:
                    # Handle both single list or nested batch structures safely
                    if isinstance(res, list):
                        pos = next((item['score'] for item in res if isinstance(item, dict) and item.get('label', '').lower() == 'positive'), 0.0)
                        neg = next((item['score'] for item in res if isinstance(item, dict) and item.get('label', '').lower() == 'negative'), 0.0)
                        scores.append(pos - neg)
                    elif isinstance(res, dict):
                        if res.get('label', '').lower() == 'positive':
                            scores.append(res.get('score', 0.0))
                        elif res.get('label', '').lower() == 'negative':
                            scores.append(-res.get('score', 0.0))
                
                sentiment_score = float(np.mean(scores)) if scores else 0.0
            except Exception as e:
                print(f"❌ Sentiment calculation error: {e}")
                pass
                
        self._set_cached(cache_key, (headlines, sentiment_score))
        return headlines, sentiment_score

    def fetch_fear_greed_index(self) -> float:
        cached = self._get_cached("fear_greed")
        if cached is not None: return cached
        try:
            res = requests.get("https://api.alternative.me/fng/", timeout=5)
            res.raise_for_status()
            data = res.json()
            val = int(data['data'][0]['value'])
            score = (val - 50) / 50.0
            self._set_cached("fear_greed", score)
            return score
        except Exception:
            return 0.0

    def fetch_etherscan_gas(self) -> float:
        cached = self._get_cached("etherscan_gas")
        if cached is not None: return cached
        if not ETHERSCAN_API_KEY:
            return 0.0
        try:
            url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={ETHERSCAN_API_KEY}"
            res = requests.get(url, timeout=5).json()
            if res.get("status") == "1":
                gas_price = float(res['result']['ProposeGasPrice'])
                score = min(gas_price / 100.0, 1.0)
                self._set_cached("etherscan_gas", score)
                return score
        except Exception:
            return 0.0
        return 0.0

    def run_prediction(self, df: pd.DataFrame, sentiment_score: float, fear_greed: float, gas_metric: float) -> dict:
        recent_returns = df['Return'].tail(10).values
        recent_vol = df['Vol_Change'].tail(10).values
        
        features = []
        for i in range(10):
            features.append([
                float(recent_returns[i]),
                float(recent_vol[i]),
                sentiment_score,
                fear_greed,
                gas_metric
            ])
        
        tensor_in = torch.tensor([features], dtype=torch.float32)
        model = BiLSTMForecaster(input_dim=5)
        model.eval()
        
        with torch.no_grad():
            pred_return = model(tensor_in).item()
            
        # Ensure scalar extract
        close_val = df['Close'].iloc[-1]
        current_price = float(close_val.iloc[0]) if isinstance(close_val, pd.Series) else float(close_val)
        predicted_price = current_price * (1.0 + pred_return)
        
        return {
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "expected_change_pct": round(pred_return * 100, 2),
            "direction": "BULLISH" if pred_return >= 0 else "BEARISH"
        }

    def generate_llm_analyst_report(self, ticker: str, metrics: dict, headlines: list[str]) -> str:
        cache_key = f"llm_{ticker}_{metrics['direction']}"
        cached = self._get_cached(cache_key)
        if cached: return cached

        if not self.groq_client:
            return "Groq API key not configured. LLM Analyst Rationale unavailable."

        prompt = f"""
        System: You are FinPulse AI, a quantitative financial analyst specializing in crypto intelligence.
        Analyze the following real-time data for {ticker}:
        - Current Price: ${metrics['current_price']}
        - Predicted Price (Bi-LSTM): ${metrics['predicted_price']} ({metrics['expected_change_pct']}%)
        - Model Signal: {metrics['direction']}
        - FinBERT Sentiment Score: {metrics['sentiment_score']} (-1.0 negative to +1.0 positive)
        - Fear & Greed Index Score: {metrics['fear_greed_score']}
        - On-Chain Gas Activity Metric: {metrics['gas_metric']}
        - Live Market Headlines: {headlines[:3]}

        Task: Provide a concise, bulleted intelligence brief (maximum 150 words) explaining:
        1. Primary rationale behind the {metrics['direction']} signal.
        2. How news sentiment and on-chain activity align or diverge from the technical price forecast.
        3. Key risk factors for traders to monitor.
        """

        try:
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250
            )
            result = completion.choices[0].message.content
            self._set_cached(cache_key, result)
            return result
        except Exception as e:
            return f"Failed to generate LLM report: {str(e)}"