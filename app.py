"""
Market Matrix — AI-Powered NIFTY 50 Dashboard
Single-file Gradio app for Hugging Face Spaces deployment.

Architecture:
  - Python handles all data fetching, indicator calculations, and analysis.
  - Gradio gr.HTML() re-renders the full dashboard HTML with baked-in data
    whenever the user changes timeframe or clicks refresh.
  - No separate frontend build needed; everything runs from this one file.
"""

import os
import json
import logging
import random
import time
from datetime import datetime, timedelta

import pytz
import pandas as pd
import numpy as np
import gradio as gr
from fastapi import FastAPI
import uvicorn

# ──────────────────────────────── Logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────── NSE Holiday Calendar ─────────────────────────────
NSE_HOLIDAYS = {
    "2024-01-26","2024-03-08","2024-03-25","2024-03-29","2024-04-11",
    "2024-04-17","2024-05-01","2024-06-17","2024-07-17","2024-08-15",
    "2024-10-02","2024-11-01","2024-11-15","2024-12-25",
    "2025-01-26","2025-02-26","2025-03-14","2025-03-31",
    "2025-04-10","2025-04-14","2025-04-18","2025-05-01",
    "2025-08-15","2025-08-27","2025-10-02","2025-10-21",
    "2025-11-05","2025-12-25",
}

NIFTY_SYMBOL = os.getenv("NIFTY_SYMBOL", "^NSEI")

# ──────────────────────────── yfinance import ──────────────────────────────
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.warning("yfinance not installed — using synthetic demo data.")


# ════════════════════════════════════════════════════════════════════════════
#   DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    if now.strftime("%Y-%m-%d") in NSE_HOLIDAYS:
        return False
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def _synthetic_ohlcv(n: int = 220, base: float = 22000.0, tf: str = "1D") -> list[dict]:
    """Generate realistic-looking synthetic OHLCV data for demo/fallback."""
    rng = np.random.default_rng(42)
    out, price = [], base
    start = datetime(2024, 1, 2)
    for i in range(n):
        if tf == "1D":
            d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        elif tf == "1W":
            d = (start + timedelta(weeks=i)).strftime("%Y-%m-%d")
        elif tf == "1M":
            d = (start + timedelta(days=30 * i)).strftime("%Y-%m-%d")
        else:
            d = (start + timedelta(minutes=i * (1 if tf=="1m" else 5 if tf=="5m" else 15))
                 ).strftime("%Y-%m-%d %H:%M:%S")
        chg   = float(rng.normal(0, 120))
        open_ = price
        close = max(price + chg, 100.0)
        high  = max(open_, close) + float(abs(rng.normal(0, 50)))
        low   = min(open_, close) - float(abs(rng.normal(0, 50)))
        vol   = int(abs(rng.normal(1_500_000, 400_000)))
        out.append({"time": d, "open": round(open_, 2), "high": round(high, 2),
                    "low": round(low, 2), "close": round(close, 2), "volume": vol})
        price = close
    return out


def get_historical_data(timeframe: str = "1D") -> list[dict]:
    tf_map = {
        "1D":  {"interval": "1d",  "period": "6mo"},
        "1W":  {"interval": "1wk", "period": "2y"},
        "1M":  {"interval": "1mo", "period": "5y"},
        "15m": {"interval": "15m", "period": "5d"},
        "5m":  {"interval": "5m",  "period": "5d"},
        "1m":  {"interval": "1m",  "period": "1d"},
    }
    if not YF_AVAILABLE or timeframe not in tf_map:
        n = 220 if timeframe in ("1D","1W","1M") else 300
        return _synthetic_ohlcv(n=n, tf=timeframe)

    cfg = tf_map[timeframe]
    try:
        ticker = yf.Ticker(NIFTY_SYMBOL)
        df = ticker.history(period=cfg["period"], interval=cfg["interval"])
        if df.empty:
            raise ValueError("Empty response from Yahoo Finance")
        df = df.dropna()
        out = []
        for idx, row in df.iterrows():
            if timeframe in ("1D","1W","1M"):
                t = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            else:
                if isinstance(idx, pd.Timestamp):
                    idx = (idx.tz_convert("Asia/Kolkata")
                           if idx.tzinfo else idx.tz_localize("UTC").tz_convert("Asia/Kolkata"))
                t = idx.strftime("%Y-%m-%d %H:%M:%S")
            out.append({"time": t, "open": float(row["Open"]), "high": float(row["High"]),
                        "low": float(row["Low"]), "close": float(row["Close"]),
                        "volume": float(row["Volume"])})
        logger.info(f"Fetched {len(out)} rows for {timeframe} from Yahoo Finance.")
        return out
    except Exception as e:
        logger.error(f"Yahoo Finance error [{timeframe}]: {e} — falling back to synthetic data.")
        return _synthetic_ohlcv(tf=timeframe)


# ════════════════════════════════════════════════════════════════════════════
#   TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"]  = _ema(df["close"], 50)
    df["ema200"] = _ema(df["close"], 200)

    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"]   = macd - sig

    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    low14   = df["low"].rolling(14).min()
    high14  = df["high"].rolling(14).max()
    rang    = (high14 - low14).replace(0, np.nan)
    k_fast  = 100 * (df["close"] - low14) / rang
    df["stoch_k"] = k_fast.rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - ma) / (0.015 * md.replace(0, np.nan))

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df


def generate_alerts(df: pd.DataFrame) -> list:
    if len(df) < 3:
        return [{"message": "Consolidating near Support", "time": "5m ago", "type": "neutral"}]

    l, p = df.iloc[-1], df.iloc[-2]
    ts = "Just now"
    alerts = []

    if p["macd_hist"] <= 0 < l["macd_hist"]:
        alerts.append({"message": "MACD Bullish Crossover",  "time": ts, "type": "bullish"})
    elif p["macd_hist"] >= 0 > l["macd_hist"]:
        alerts.append({"message": "MACD Bearish Crossover",  "time": ts, "type": "bearish"})
    if p["rsi"] < 30 <= l["rsi"]:
        alerts.append({"message": "RSI Oversold Bounce",     "time": ts, "type": "bullish"})
    elif p["rsi"] > 70 >= l["rsi"]:
        alerts.append({"message": "RSI Overbought Drop",     "time": ts, "type": "bearish"})
    if p["close"] < p["ema50"] and l["close"] > l["ema50"]:
        alerts.append({"message": "EMA50 Breakout",          "time": ts, "type": "bullish"})
    elif p["close"] > p["ema50"] and l["close"] < l["ema50"]:
        alerts.append({"message": "EMA50 Breakdown",         "time": ts, "type": "bearish"})
    if p["stoch_k"] <= p["stoch_d"] and l["stoch_k"] > l["stoch_d"]:
        alerts.append({"message": "Stochastic Bull Cross",   "time": ts, "type": "bullish"})
    elif p["stoch_k"] >= p["stoch_d"] and l["stoch_k"] < l["stoch_d"]:
        alerts.append({"message": "Stochastic Bear Cross",   "time": ts, "type": "bearish"})

    if not alerts:
        alerts = [
            {"message": "Consolidating near Support", "time": "5m ago",  "type": "neutral"},
            {"message": "Volume Spike Detected",      "time": "15m ago", "type": "neutral"},
        ]
    return alerts[:5]


# ════════════════════════════════════════════════════════════════════════════
#   PATTERN RECOGNITION
# ════════════════════════════════════════════════════════════════════════════

def detect_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"pattern": "No Active Pattern", "confidence": 0}

    l, p = df.iloc[-1], df.iloc[-2]
    cb = l["close"] - l["open"]
    cr = l["high"]  - l["low"]
    pb = p["close"] - p["open"]

    if pb < 0 and cb > 0 and l["close"] > p["open"] and l["open"] < p["close"]:
        return {"pattern": "Bullish Engulfing",  "confidence": 0.85}
    if pb > 0 and cb < 0 and l["open"] > p["close"] and l["close"] < p["open"]:
        return {"pattern": "Bearish Engulfing",  "confidence": 0.85}
    if cr > 0 and abs(cb) / cr < 0.1:
        return {"pattern": "Doji",               "confidence": 0.70}

    lw = (l["open"] - l["low"])   if cb > 0 else (l["close"] - l["low"])
    uw = (l["high"] - l["close"]) if cb > 0 else (l["high"]  - l["open"])
    if lw > 2 * abs(cb) and uw < 0.2 * max(abs(cb), 0.001):
        return {"pattern": "Hammer",             "confidence": 0.75}

    if len(df) >= 3:
        p2 = df.iloc[-3]
        if ((p2["close"] - p2["open"]) < 0
                and abs(pb) < (p2["high"] - p2["low"]) * 0.3
                and cb > 0
                and l["close"] > (p2["open"] + p2["close"]) / 2):
            return {"pattern": "Morning Star",   "confidence": 0.80}

    if len(df) >= 20:
        recent = df.tail(20)
        if (recent["low"].is_monotonic_increasing
                and l["close"] >= recent["high"].max() * 0.998):
            return {"pattern": "Ascending Triangle",  "confidence": 0.90}
        if (recent["high"].is_monotonic_decreasing
                and l["close"] <= recent["low"].min() * 1.002):
            return {"pattern": "Descending Triangle", "confidence": 0.90}

    return {"pattern": "No Active Pattern", "confidence": 0}


# ════════════════════════════════════════════════════════════════════════════
#   AI FORECAST
# ════════════════════════════════════════════════════════════════════════════

def generate_ai_forecast(df: pd.DataFrame) -> dict:
    if len(df) < 14:
        return {"signal": "Neutral", "confidence": 50,
                "reasons": ["Not enough data"], "target_price": None}

    l = df.iloc[-1]
    score, reasons = 0, []

    if l["close"] > l["ema50"]:   score += 20; reasons.append("Price above 50 EMA")
    else:                          score -= 20; reasons.append("Price below 50 EMA")
    if l["close"] > l["ema200"]:  score += 30; reasons.append("Long-term Bullish (Price > 200 EMA)")
    else:                          score -= 30; reasons.append("Long-term Bearish (Price < 200 EMA)")
    if l["macd_hist"] > 0:        score += 25; reasons.append("MACD Bullish Momentum")
    else:                          score -= 25; reasons.append("MACD Bearish Momentum")
    if l["rsi"] < 30:             score += 25; reasons.append("RSI Oversold — Reversal Potential")
    elif l["rsi"] > 70:           score -= 25; reasons.append("RSI Overbought — Downside Risk")

    std  = float(df["close"].tail(14).std())
    conf = min(abs(score), 95)

    if score > 20:
        sig   = "Bullish"
        delta = random.uniform(0.5 * std, 2.0 * std)
    elif score < -20:
        sig   = "Bearish"
        delta = -random.uniform(0.5 * std, 2.0 * std)
    else:
        sig   = "Sideways"
        delta = random.uniform(-0.5 * std, 0.5 * std)

    return {
        "signal": sig,
        "confidence": round(conf, 1),
        "reasons": reasons[:3],
        "predicted_pts": round(delta, 2),
        "predicted_return_pct": round(delta / float(l["close"]) * 100, 2),
        "target_price": round(float(l["close"]) + delta, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
#   SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def get_sentiment(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"positive_pct": 50,
                "news": [{"title": "Insufficient data", "publisher": "", "link": "", "time": ""}]}

    l = df.iloc[-1]
    bull = 0

    if l.get("macd_hist", 0) > 0:                          bull += 1
    if l.get("rsi", 50) < 30:                              bull += 1
    if l.get("stoch_k", 50) > l.get("stoch_d", 50):       bull += 1
    if l.get("cci", 0) > 100:                              bull += 1
    if l["close"] > l.get("ema50", l["close"]):            bull += 1
    pct = int((bull / 5) * 100)

    news = []
    if YF_AVAILABLE:
        try:
            raw = yf.Ticker(NIFTY_SYMBOL).news
            if raw:
                for item in raw[:4]:
                    title = item.get("title", "")
                    if title:
                        news.append({
                            "title":     title,
                            "publisher": item.get("publisher", "Yahoo Finance"),
                            "link":      item.get("link", "#"),
                            "time":      str(item.get("providerPublishTime", "")),
                        })
        except Exception as ex:
            logger.error(f"News fetch error: {ex}")

    if not news:
        news = [{"title": "Market news unavailable — check internet connectivity",
                 "publisher": "System", "link": "#", "time": ""}]
    return {"positive_pct": pct, "news": news}


# ════════════════════════════════════════════════════════════════════════════
#   STATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_dashboard_data(timeframe: str = "1D") -> dict:
    raw = get_historical_data(timeframe)
    if not raw:
        return {"error": "No data available"}

    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"])
    df = enrich_with_indicators(df)

    if timeframe in ("1D", "1W", "1M"):
        df["time"] = df["time"].dt.strftime("%Y-%m-%d")
    else:
        df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    records = df.to_dict(orient="records")

    patterns  = detect_patterns(df)
    forecast  = generate_ai_forecast(df)
    sentiment = get_sentiment(df)
    alerts    = generate_alerts(df)

    ltp  = float(df.iloc[-1]["close"])
    prev = float(df.iloc[-2]["close"]) if len(df) > 1 else ltp
    chg  = round(ltp - prev, 2)
    chg_pct = round((ltp - prev) / prev * 100, 2) if prev else 0.0

    ist = pytz.timezone("Asia/Kolkata")
    now_str = datetime.now(ist).strftime("%H:%M:%S")

    market_open = is_market_open()
    return {
        "candles":       records[-250:],
        "patterns":      patterns,
        "forecast":      forecast,
        "sentiment":     sentiment,
        "alerts":        alerts,
        "snapshot": {
            "ltp":        ltp,
            "change":     chg,
            "change_pct": chg_pct,
            "support":    round(float(df["low"].tail(14).min()), 0),
            "resistance": round(float(df["high"].tail(14).max()), 0),
        },
        "provider":      "Live (Delayed) · Yahoo Finance" if market_open else "Market Closed · Last available data",
        "is_market_open": market_open,
        "last_updated":  now_str,
        "timeframe":     timeframe,
    }


# ════════════════════════════════════════════════════════════════════════════
#   HTML RENDERER
# ════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#030712;--surface:rgba(15,23,42,0.9);--surface2:rgba(30,41,59,0.75);
  --border:rgba(255,255,255,0.08);--border2:rgba(255,255,255,0.15);
  --text:#e2e8f0;--muted:#64748b;--accent:#3b82f6;--accent2:#6366f1;
  --bull:#10b981;--bear:#ef4444;--warn:#f59e0b;
}
html,body{min-height:100vh;font-family:'Inter',sans-serif;background:var(--bg);color:var(--text)}
body::before{content:'';position:fixed;inset:0;z-index:-1;
  background:radial-gradient(ellipse 80% 55% at 15% 0%,rgba(59,130,246,.13) 0%,transparent 65%),
             radial-gradient(ellipse 65% 50% at 85% 100%,rgba(99,102,241,.10) 0%,transparent 60%),
             var(--bg)}
body::after{content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
  background-size:64px 64px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.app{max-width:1560px;margin:0 auto;padding:24px 18px;display:flex;flex-direction:column;gap:18px}

/* ── Header ── */
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
     padding-bottom:18px;border-bottom:1px solid var(--border2)}
.brand h1{font-size:clamp(1.5rem,3vw,2.2rem);font-weight:700;letter-spacing:-.5px;
  background:linear-gradient(135deg,#fff 30%,#60a5fa 70%,#818cf8 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 18px rgba(96,165,250,.45))}
.brand p{color:var(--muted);font-size:.83rem;margin-top:4px}
.hdr-right{display:flex;flex-direction:column;align-items:flex-end;gap:8px}
.hdr-row{display:flex;align-items:center;gap:10px}

/* Status pill */
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;
      border-radius:20px;font-size:.82rem;background:var(--surface);border:1px solid var(--border2)}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-live{background:var(--bull);box-shadow:0 0 8px var(--bull);animation:pulse 2s infinite}
.dot-closed{background:var(--warn);box-shadow:0 0 8px var(--warn)}
.dot-off{background:var(--bear)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.last-upd{font-size:.71rem;color:var(--muted)}

/* ── KPI Row ── */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:13px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:14px;
     padding:15px 17px;position:relative;overflow:hidden;transition:transform .2s,border-color .2s}
.kpi:hover{transform:translateY(-2px);border-color:var(--border2)}
.kpi::before{content:'';position:absolute;inset:0;border-radius:14px;
  background:linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 60%);pointer-events:none}
.kpi-lbl{font-size:.71rem;text-transform:uppercase;letter-spacing:.8px;color:var(--muted)}
.kpi-val{font-size:1.5rem;font-weight:700;margin-top:5px;line-height:1}
.kpi-sub{font-size:.77rem;margin-top:4px}
.bull{color:var(--bull)}.bear{color:var(--bear)}.neutral{color:var(--warn)}

/* ── Main grid ── */
.main-grid{display:grid;grid-template-columns:1fr 298px;gap:16px}
@media(max-width:1060px){.main-grid{grid-template-columns:1fr}}

/* Chart panel */
.chart-panel{background:var(--surface);border:1px solid var(--border);
             border-radius:16px;padding:17px;display:flex;flex-direction:column;gap:13px}
.chart-hdr{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.chart-title{font-size:1rem;font-weight:600}
.tf-bar{display:flex;gap:6px;flex-wrap:wrap}
.tf-btn{padding:4px 12px;border-radius:5px;font-size:.77rem;cursor:pointer;
        border:1px solid var(--border2);background:var(--surface2);color:var(--text);transition:all .18s}
.tf-btn:hover{background:rgba(59,130,246,.18)}
.tf-btn.active{background:var(--accent);border-color:var(--accent);font-weight:600}
#chart-wrap{min-height:420px;width:100%;border-radius:8px;overflow:hidden}
.chart-foot{display:flex;justify-content:space-between;align-items:center;
            border-top:1px solid var(--border);padding-top:12px}
.cf-sig{font-size:.82rem;color:var(--accent);font-weight:600}
.cf-vol{font-size:.77rem;color:var(--muted)}
.cf-sub{font-size:.79rem;padding:3px 2px}

/* Sidebar */
.sidebar{display:flex;flex-direction:column;gap:13px}

/* Widget base */
.widget{background:var(--surface);border:1px solid var(--border);
        border-radius:14px;padding:15px;transition:border-color .2s}
.widget:hover{border-color:var(--border2)}
.w-title{font-size:.73rem;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);
         margin-bottom:11px;display:flex;align-items:center;gap:6px}
.w-title::before{content:'';display:block;width:3px;height:12px;border-radius:2px;background:var(--accent)}

/* Forecast widget */
.sig-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 15px;
           border-radius:8px;font-weight:700;font-size:1rem;margin-bottom:9px}
.sb-bull{background:rgba(16,185,129,.13);color:var(--bull);border:1px solid rgba(16,185,129,.32)}
.sb-bear{background:rgba(239,68,68,.13); color:var(--bear);border:1px solid rgba(239,68,68,.32)}
.sb-side{background:rgba(245,158,11,.12);color:var(--warn);border:1px solid rgba(245,158,11,.3)}
.conf-bar{height:4px;border-radius:2px;background:rgba(255,255,255,.08);margin-bottom:9px}
.conf-fill{height:100%;border-radius:2px}
.reason{font-size:.77rem;color:var(--muted);padding:3px 0;display:flex;align-items:center;gap:5px}
.reason::before{content:'›';color:var(--accent)}
.fc-meta{margin-top:8px;font-size:.79rem;color:var(--muted)}

/* Pattern */
.pat-name{font-size:1.1rem;font-weight:700;margin-bottom:5px}
.pat-conf-lbl{font-size:.74rem;color:var(--muted)}
.pat-conf-val{font-weight:700;font-size:.9rem}

/* Sentiment */
.sent-meter{height:6px;border-radius:3px;background:rgba(255,255,255,.06);margin:8px 0;overflow:hidden}
.sent-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--bear),var(--warn) 50%,var(--bull))}
.sent-pct{font-size:1.28rem;font-weight:700}
.sent-labels{display:flex;justify-content:space-between;font-size:.69rem;color:var(--muted)}
.news-list{margin-top:9px;display:flex;flex-direction:column;gap:6px}
.news-item{font-size:.74rem;padding:6px 8px;border-radius:6px;
           background:var(--surface2);border:1px solid var(--border);line-height:1.4}
.news-pub{color:var(--muted);font-size:.67rem;margin-top:2px}

/* Alerts */
.alert-item{display:flex;justify-content:space-between;align-items:center;
            padding:7px 11px;margin-bottom:5px;border-radius:8px;font-size:.81rem}
.a-bull{background:rgba(16,185,129,.08);border-left:3px solid var(--bull)}
.a-bear{background:rgba(239,68,68,.08); border-left:3px solid var(--bear)}
.a-neutral{background:rgba(245,158,11,.06);border-left:3px solid var(--warn)}
.a-time{font-size:.69rem;color:var(--muted)}

/* Technicals */
.t-table{width:100%;border-collapse:collapse;font-size:.81rem}
.t-table td{padding:7px 5px;border-bottom:1px solid var(--border)}
.t-table td:first-child{color:var(--muted);font-size:.73rem;text-transform:uppercase;letter-spacing:.5px}
.t-table td:last-child{text-align:right;font-weight:600}
.t-table tr:last-child td{border-bottom:none}

/* Bottom row */
.bot-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:780px){.bot-grid{grid-template-columns:1fr}}

/* Scrollbar */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
</style>"""


def _build_html(d: dict, active_tf: str = "1D") -> str:
    """Render the full dashboard HTML with data baked in as inline JSON."""

    snap     = d.get("snapshot", {})
    forecast = d.get("forecast", {})
    patterns = d.get("patterns", {})
    sentiment= d.get("sentiment", {})
    alerts   = d.get("alerts", [])
    candles  = d.get("candles", [])
    mkt_open = d.get("is_market_open", False)
    provider = d.get("provider", "")
    upd      = d.get("last_updated", "")

    # ── KPI cards ──────────────────────────────────────────────────
    ltp   = snap.get("ltp", 0)
    chg   = snap.get("change", 0)
    chgp  = snap.get("change_pct", 0)
    sup   = snap.get("support", 0)
    res   = snap.get("resistance", 0)
    pos   = chg >= 0
    arrow = "▲" if pos else "▼"
    chg_cls = "bull" if pos else "bear"

    kpi_html = f"""
    <div class="kpi">
      <div class="kpi-lbl">LTP</div>
      <div class="kpi-val {chg_cls}">₹{ltp:,.2f}</div>
      <div class="kpi-sub {chg_cls}">{arrow} {chg:+.2f} ({chgp:+.2f}%)</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Change</div>
      <div class="kpi-val {chg_cls}">{chgp:+.2f}%</div>
      <div class="kpi-sub">vs previous close</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Support</div>
      <div class="kpi-val neutral">₹{sup:,.0f}</div>
      <div class="kpi-sub">14-day low</div>
    </div>
    <div class="kpi">
      <div class="kpi-lbl">Resistance</div>
      <div class="kpi-val neutral">₹{res:,.0f}</div>
      <div class="kpi-sub">14-day high</div>
    </div>"""

    # ── Timeframe buttons ───────────────────────────────────────────
    tfs = ["1m", "5m", "15m", "1D", "1W", "1M"]
    tf_btns = " ".join(
        f'<button class="tf-btn{"  active" if tf == active_tf else ""}" '
        f'hx-on:click="switchTf(\'{tf}\')">{tf}</button>' for tf in tfs
    )
    # Use standard onclick (no htmx needed here)
    tf_btns = " ".join(
        f'<button class="tf-btn{" active" if tf == active_tf else ""}" '
        f'id="tf-{tf}" onclick="switchTf(\'{tf}\')">{tf}</button>' for tf in tfs
    )

    # ── Forecast widget ─────────────────────────────────────────────
    sig   = forecast.get("signal", "Neutral")
    conf  = forecast.get("confidence", 50)
    reasons = forecast.get("reasons", [])
    target  = forecast.get("target_price")
    sig_icon  = "↑" if sig == "Bullish" else "↓" if sig == "Bearish" else "⇄"
    sig_cls   = "sb-bull" if sig == "Bullish" else "sb-bear" if sig == "Bearish" else "sb-side"
    conf_col  = "var(--bull)" if sig == "Bullish" else "var(--bear)" if sig == "Bearish" else "var(--warn)"
    reasons_html = "".join(f'<div class="reason">{r}</div>' for r in reasons)
    fc_meta = f'Confidence: <strong style="color:{conf_col}">{conf}%</strong>'
    if target:
        fc_meta += f' &nbsp;|&nbsp; Target: <strong>₹{target:,.2f}</strong>'

    forecast_html = f"""
    <div class="sig-badge {sig_cls}">{sig_icon} {sig}</div>
    <div class="conf-bar"><div class="conf-fill" style="width:{conf}%;background:{conf_col}"></div></div>
    {reasons_html}
    <div class="fc-meta">{fc_meta}</div>"""

    # ── Pattern widget ──────────────────────────────────────────────
    pat     = patterns.get("pattern", "No Active Pattern")
    pat_conf= float(patterns.get("confidence", 0))
    pat_col = ("var(--bull)" if pat_conf >= 0.80 else
               "var(--warn)" if pat_conf > 0     else "var(--muted)")
    pat_conf_str = f"{pat_conf*100:.0f}%" if pat_conf > 0 else "—"

    pattern_html = f"""
    <div class="pat-name" style="color:{pat_col}">{pat}</div>
    <div class="pat-conf-lbl">Confidence</div>
    <div class="pat-conf-val" style="color:{pat_col}">{pat_conf_str}</div>"""

    # ── Sentiment widget ────────────────────────────────────────────
    pct      = sentiment.get("positive_pct", 50)
    news_items = sentiment.get("news", [])
    sent_cls = "bull" if pct > 55 else "bear" if pct < 45 else "neutral"
    news_html = "".join(
        f'<div class="news-item">'
        f'<a href="{n.get("link","#")}" target="_blank" rel="noopener">{n.get("title","")}</a>'
        f'<div class="news-pub">{n.get("publisher","")}</div></div>'
        for n in news_items
    )
    sentiment_html = f"""
    <div class="sent-pct {sent_cls}">{pct}% Bullish</div>
    <div class="sent-meter"><div class="sent-fill" style="width:{pct}%"></div></div>
    <div class="sent-labels"><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>
    <div class="news-list">{news_html}</div>"""

    # ── Alerts ──────────────────────────────────────────────────────
    alerts_html = "".join(
        f'<div class="alert-item a-{a.get("type","neutral")}">'
        f'<span>{a.get("message","")}</span>'
        f'<span class="a-time">{a.get("time","")}</span></div>'
        for a in alerts
    )

    # ── Technicals ──────────────────────────────────────────────────
    last = candles[-1] if candles else {}
    rsi   = float(last.get("rsi",   0))
    macdh = float(last.get("macd_hist", 0))
    stoch = float(last.get("stoch_k",  0))
    cci   = float(last.get("cci",   0))
    e50   = float(last.get("ema50", 0))
    e200  = float(last.get("ema200", 0))

    rsi_cls  = "bull" if rsi < 30 else "bear" if rsi > 70 else "neutral"
    macd_cls = "bull" if macdh > 0 else "bear"
    rsi_lbl  = "[Oversold]" if rsi < 30 else "[Overbought]" if rsi > 70 else "[Normal]"

    techs_html = f"""
    <tr><td>RSI (14)</td>  <td class="{rsi_cls}">{rsi:.1f} {rsi_lbl}</td></tr>
    <tr><td>MACD Hist</td> <td class="{macd_cls}">{macdh:.2f}</td></tr>
    <tr><td>Stoch %K</td>  <td>{stoch:.1f}</td></tr>
    <tr><td>CCI (20)</td>  <td>{cci:.1f}</td></tr>
    <tr><td>EMA 50</td>    <td>{e50:.2f}</td></tr>
    <tr><td>EMA 200</td>   <td>{e200:.2f}</td></tr>"""

    # ── Chart header / footer ───────────────────────────────────────
    chart_sig     = f"↑ AI Insight: {sig} Signal Detected"
    chart_sig_sub_txt = (
        "✔  Bullish momentum confirmed by indicators" if sig == "Bullish" else
        "✘  Bearish pressure — caution advised"       if sig == "Bearish" else
        "⇄  Sideways consolidation zone"
    )
    chart_sig_sub_col = ("var(--bull)" if sig == "Bullish" else
                         "var(--bear)" if sig == "Bearish" else "var(--warn)")

    # Status dot
    dot_cls  = "dot-live" if mkt_open else "dot-closed"
    stat_txt = provider if provider else ("Live Feed" if mkt_open else "Market Closed")

    # Serialize candles for JavaScript
    candles_json = json.dumps(candles[-250:], default=str)

    # ── Assemble full page ──────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Market Matrix — NIFTY 50 AI Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
{CSS}
</head>
<body>
<div class="app">

  <!-- Header -->
  <header class="hdr">
    <div class="brand">
      <h1>Market Matrix</h1>
      <p>AI-Powered Candlestick Analysis · NIFTY 50</p>
    </div>
    <div class="hdr-right">
      <div class="hdr-row">
        <div class="pill">
          <div class="dot {dot_cls}"></div>
          <span>{stat_txt}</span>
        </div>
      </div>
      <div class="last-upd">{'Last updated ' + upd + ' IST' if upd else ''}</div>
      <div class="last-upd" style="color:rgba(100,116,139,.6);font-size:.68rem">
        Use the timeframe buttons or Gradio controls to refresh data
      </div>
    </div>
  </header>

  <!-- KPI Row -->
  <section class="kpi-row">{kpi_html}</section>

  <!-- Main grid -->
  <div class="main-grid">

    <!-- Chart -->
    <div class="chart-panel">
      <div class="chart-hdr">
        <div class="chart-title">| NIFTY 50 — Technical Analysis</div>
        <div class="tf-bar">{tf_btns}</div>
      </div>
      <div id="chart-wrap"></div>
      <div class="chart-foot">
        <div class="cf-sig">{chart_sig}</div>
        <div class="cf-vol">Volume bars shown</div>
      </div>
      <div class="cf-sub" style="color:{chart_sig_sub_col}">{chart_sig_sub_txt}</div>
    </div>

    <!-- Sidebar -->
    <aside class="sidebar">
      <div class="widget">
        <div class="w-title">AI Forecast</div>
        {forecast_html}
      </div>
      <div class="widget">
        <div class="w-title">Pattern Detection</div>
        {pattern_html}
      </div>
      <div class="widget">
        <div class="w-title">Market Sentiment</div>
        {sentiment_html}
      </div>
    </aside>
  </div>

  <!-- Bottom row -->
  <div class="bot-grid">
    <div class="widget">
      <div class="w-title">Smart Alerts</div>
      {alerts_html}
    </div>
    <div class="widget">
      <div class="w-title">Technical Indicators</div>
      <table class="t-table"><tbody>{techs_html}</tbody></table>
    </div>
  </div>

</div><!-- .app -->

<script>
/* ── Chart initialisation with baked-in data ── */
const CANDLES = {candles_json};
const ACTIVE_TF = '{active_tf}';

(function() {{
  const container = document.getElementById('chart-wrap');
  if (!container || !CANDLES.length) return;

  const chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth || 800,
    height: 440,
    layout: {{ background: {{type:'solid',color:'transparent'}}, textColor:'#94a3b8' }},
    grid: {{ vertLines:{{color:'rgba(255,255,255,0.04)'}}, horzLines:{{color:'rgba(255,255,255,0.04)'}} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    rightPriceScale: {{ borderColor:'rgba(255,255,255,0.1)' }},
    timeScale: {{ borderColor:'rgba(255,255,255,0.1)', timeVisible:true, secondsVisible:false }},
  }});

  const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor:'#10b981', downColor:'#ef4444',
    borderUpColor:'#10b981', borderDownColor:'#ef4444',
    wickUpColor:'#10b981',  wickDownColor:'#ef4444',
  }});
  const ema50Series  = chart.addSeries(LightweightCharts.LineSeries,
    {{color:'#3b82f6',lineWidth:1.5,title:'EMA 50', lastValueVisible:false,priceLineVisible:false}});
  const ema200Series = chart.addSeries(LightweightCharts.LineSeries,
    {{color:'#f59e0b',lineWidth:1.5,title:'EMA 200',lastValueVisible:false,priceLineVisible:false}});
  const volSeries = chart.addSeries(LightweightCharts.HistogramSeries, {{
    priceFormat:{{type:'volume'}}, priceScaleId:'', scaleMargins:{{top:0.82,bottom:0}},
  }});

  const parseTime = t => {{
    if (typeof t === 'string' && t.includes(':'))
      return Math.floor(new Date(t.replace(' ','T')+'Z').getTime()/1000);
    return t;
  }};

  const seen={{}};
  CANDLES.forEach(c=>{{seen[c.time]=c;}});
  const sorted = Object.values(seen)
    .map(c=>{{return{{...c,_t:parseTime(c.time)}}}})
    .sort((a,b)=>a._t-b._t);

  candleSeries.setData(sorted.map(c=>{{return{{time:c._t,open:c.open,high:c.high,low:c.low,close:c.close}}}}));
  ema50Series.setData( sorted.filter(c=>c.ema50).map(c=>{{return{{time:c._t,value:c.ema50}}}}));
  ema200Series.setData(sorted.filter(c=>c.ema200).map(c=>{{return{{time:c._t,value:c.ema200}}}}));
  volSeries.setData(   sorted.map(c=>{{return{{time:c._t,value:c.volume||0,color:c.close>=c.open?'#10b98166':'#ef444466'}}}}));
  chart.timeScale().fitContent();

  // Responsive resize
  new ResizeObserver(e=>{{
    if(chart && e[0]) chart.applyOptions({{width:e[0].contentRect.width}});
  }}).observe(container);
}})();

/* ── Timeframe switch: post a Gradio event to trigger Python re-render ── */
function switchTf(tf) {{
  // Update active button style immediately for UX
  document.querySelectorAll('.tf-btn').forEach(b=>b.classList.toggle('active',b.id==='tf-'+tf));
  // Submit the Gradio dropdown change
  const sel = window.parent ? window.parent.document : document;
  // Find the hidden Gradio components and click through them
  // Approach: dispatch a custom event that the Gradio JS bridge picks up
  window.parent.postMessage({{type:'MARKET_MATRIX_TF', tf: tf}}, '*');
}}
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════
#   GRADIO APPLICATION
# ════════════════════════════════════════════════════════════════════════════

def render_dashboard(timeframe: str) -> str:
    """Main Gradio function: fetch data → render HTML."""
    logger.info(f"render_dashboard called with tf={timeframe}")
    try:
        data = build_dashboard_data(timeframe)
    except Exception as e:
        logger.error(f"render_dashboard error: {e}")
        data = {
            "candles": [], "patterns": {}, "forecast": {}, "sentiment": {},
            "alerts": [{"message": f"Error fetching data: {e}", "time": "", "type": "neutral"}],
            "snapshot": {"ltp":0,"change":0,"change_pct":0,"support":0,"resistance":0},
            "provider": "Error", "is_market_open": False, "last_updated": "",
        }
    return _build_html(data, active_tf=timeframe)


TF_CHOICES = ["1D", "1W", "1M", "15m", "5m", "1m"]

GRADIO_CSS = """
/* Make Gradio chrome minimal so dashboard fills the page */
.gradio-container { padding: 0 !important; background: #030712 !important; }
#component-0 { gap: 0 !important; }
footer { display:none !important; }
#tf-select label { color:#94a3b8 !important; font-size:.8rem; }
#refresh-btn { border-radius:8px !important; }
.control-bar { background:#0f172a; border-bottom:1px solid rgba(255,255,255,.08);
               padding:10px 20px; display:flex; align-items:center; gap:16px; }
"""

GRADIO_THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Market Matrix — NIFTY 50 AI Dashboard") as demo:

    with gr.Row(elem_classes="control-bar"):
        tf_dropdown = gr.Dropdown(
            choices=TF_CHOICES, value="1D", label="Timeframe",
            elem_id="tf-select", scale=1,
        )
        refresh_btn = gr.Button("↻ Refresh Data", variant="primary",
                                elem_id="refresh-btn", scale=0, min_width=140)

    dashboard_output = gr.HTML(
        value=render_dashboard("1D"),
        label="",
    )

    # Wire up interactions
    tf_dropdown.change(fn=render_dashboard, inputs=[tf_dropdown], outputs=[dashboard_output])
    refresh_btn.click( fn=render_dashboard, inputs=[tf_dropdown], outputs=[dashboard_output])


# Expose FastAPI app for Vercel
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# Local development only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
