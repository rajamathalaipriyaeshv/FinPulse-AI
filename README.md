# 📈 FinPulse AI: Real-Time Crypto Sentiment & Price Forecasting Terminal

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FinBERT-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/ProsusAI/finbert)
[![TradingView](https://img.shields.io/badge/UI-TradingView%20Charts-131622?style=flat&logo=tradingview&logoColor=white)](https://www.tradingview.com/lightweight-charts/)

## 📌 Project Overview

**FinPulse AI** is a real-time cryptocurrency sentiment analysis and 24-hour price forecasting engine designed to eliminate the **"retail sentiment trap"** in volatile digital asset markets. Powered by a high-performance FastAPI backend, the system continuously ingests live crypto headlines and market data to bridge qualitative news context with quantitative technical indicators.

It utilizes a localized **FinBERT** transformer model to compute domain-specific sentiment polarity scores (-1.0 to +1.0) without neutral text dilution, and feeds those signals alongside historical OHLCV data into a PyTorch-built **Bidirectional LSTM (Bi-LSTM)** neural network to predict short-term price trajectories. Fronted by a sleek, midnight-navy dark dashboard featuring interactive TradingView Lightweight Charts, FinPulse AI replaces emotional trading and surface-level news hype with low-latency, data-backed predictive analytics on a single unified terminal.

---

## ✨ Core Features

* **Domain-Specific NLP (FinBERT):** Fine-tuned financial language processing to accurately extract sentiment polarity from news headlines without generic dictionary errors.
* **Multimodal Time-Series Forecasting:** Combines qualitative headline sentiment scores with quantitative OHLCV price action inside a PyTorch **Bi-LSTM** model.
* **Low-Latency Async Backend:** Built with **FastAPI** and **Uvicorn** for high-concurrency polling and sub-150ms inference handling.
* **Interactive Dark Terminal UI:** Custom Obsidian glassmorphism dashboard styled with embedded **TradingView Lightweight Charts** for dynamic visualization.
* **Automated Market Pipelines:** Ingests live financial data streams smoothly via `yfinance` and real-time news via `NewsAPI / RSS`.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Data Ingestion
        A1[NewsAPI / RSS Feeds] --> A2[Text Preprocessing]
        B1[yfinance API] --> B2[OHLCV Feature Scaling]
    end

    subgraph Machine Learning Pipeline
        A2 --> C[FinBERT Sentiment Model]
        B2 --> D[Bi-LSTM Neural Network]
        C -->|Polarity Score| D
    end

    subgraph Backend API
        D -->|Price Trajectory| E[FastAPI Async Server]
        E --> F[Pydantic Validation]
    end

    subgraph UI Terminal
        F --> G[TradingView Lightweight Charts]
        F --> H[Obsidian Dark Dashboard]
    end

    Layer,Technologies
Language & Environment,Python 3.10+
Deep Learning & NLP,"PyTorch, Hugging Face transformers (FinBERT), Scikit-Learn"
Backend API,"FastAPI, Uvicorn, Pydantic"
Data Processing,"yfinance, Pandas, NumPy, Requests"
Frontend UI,"HTML5, CSS3 (Obsidian Dark Theme), JavaScript (ES6+), TradingView Lightweight Charts"

Developed as an engineering project at Sri Eshwar College of Engineering, focusing on real-time NLP financial sentiment extraction, deep learning time-series integration, and modern high-concurrency web architectures.
