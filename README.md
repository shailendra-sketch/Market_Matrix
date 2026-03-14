---
title: Market Matrix — NIFTY 50 AI Dashboard
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: mit
---

# Market Matrix — AI-Powered NIFTY 50 Dashboard

An **AI-powered candlestick analysis dashboard** for NIFTY 50, deployable on Hugging Face Spaces.

## Features

| Feature | Description |
|---|---|
| 📊 **Candlestick Chart** | Interactive chart with EMA 50/200 overlays and volume bars |
| 🤖 **AI Forecast** | Rule-based signal engine (Bullish / Bearish / Sideways) with confidence score |
| 🕯️ **Pattern Detection** | Detects 7+ candlestick patterns (Engulfing, Doji, Hammer, Morning Star, Triangles…) |
| 📰 **Sentiment Analysis** | Multi-indicator sentiment score + live Yahoo Finance news headlines |
| ⚡ **Smart Alerts** | MACD crossovers, RSI extremes, EMA breakouts |
| 🕐 **Multi-Timeframe** | 1m · 5m · 15m · 1D · 1W · 1M |
| 🔄 **Auto-Refresh** | Data refreshes every 30 seconds |

## Data Source

- **Yahoo Finance** (`^NSEI`) via `yfinance` library
- Falls back to **synthetic demo data** if Yahoo Finance is unavailable

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7860](http://localhost:7860).

## Deploying to Hugging Face Spaces

1. Create a new **Gradio** Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `app.py`, `requirements.txt`, and this `README.md`
3. The Space auto-installs dependencies and launches

> **Note:** NIFTY 50 data is available from Yahoo Finance with a ~15-minute delay during market hours (Mon–Fri 09:15–15:30 IST). Outside market hours, the last available data is shown.
