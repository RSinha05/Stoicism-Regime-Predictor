"""
daily_update.py
Runs every weekday via GitHub Actions.
Fetches live NIFTY50 proxies from yfinance, calls the API, saves result.
"""
import os, json, requests
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── NIFTY 50 tickers on NSE ───────────────────────────────────────────────────
TICKERS = [
    "HDFCBANK.NS","ICICIBANK.NS","RELIANCE.NS","INFY.NS","TCS.NS",
    "WIPRO.NS","HCLTECH.NS","TECHM.NS","SBIN.NS","KOTAKBANK.NS",
    "AXISBANK.NS","BAJFINANCE.NS","ONGC.NS","COALINDIA.NS","NTPC.NS",
    "POWERGRID.NS","BPCL.NS","HINDUNILVR.NS","ITC.NS","NESTLEIND.NS",
    "BRITANNIA.NS","BHARTIARTL.NS","MARUTI.NS","TATAMOTORS.NS","M&M.NS",
    "HEROMOTOCO.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","SUNPHARMA.NS",
    "DRREDDY.NS","CIPLA.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS",
    "ULTRACEMCO.NS","GRASIM.NS","SHREECEM.NS","ASIANPAINT.NS","TITAN.NS",
    "DIVISLAB.NS","LT.NS","ADANIPORTS.NS","SBILIFE.NS","HDFCLIFE.NS",
    "BAJAJFINSV.NS","INDUSINDBK.NS","APOLLOHOSP.NS","ADANIENT.NS",
]

def fetch_pe_estimates():
    """
    Estimate median PE from trailing price / EPS proxy.
    yfinance `info` gives trailingPE directly.
    """
    pes, eps_list = [], []
    for t in TICKERS[:20]:   # limit to avoid rate limits on free tier
        try:
            info = yf.Ticker(t).info
            if info.get("trailingPE"):
                pes.append(info["trailingPE"])
            if info.get("trailingEps") and info.get("forwardEps"):
                g = (info["forwardEps"] - info["trailingEps"]) / (abs(info["trailingEps"]) + 1e-6)
                eps_list.append(g)
        except Exception:
            pass

    pe_median  = float(np.median(pes))  if pes  else 31.0
    eps_growth = float(np.median(eps_list)) if eps_list else 0.10
    return pe_median, eps_growth

def fetch_market_cap_growth():
    """Estimate MCap growth using index level change."""
    nifty = yf.download("^NSEI", period="2y", interval="1mo", progress=False)
    if len(nifty) < 13:
        return 0.10
    latest = float(nifty["Close"].iloc[-1])
    year_ago = float(nifty["Close"].iloc[-13])
    return (latest - year_ago) / year_ago

def compute_signals():
    print("Fetching PE and EPS...")
    pe_median, eps_growth = fetch_pe_estimates()

    print("Fetching MCap growth...")
    mc_growth = fetch_market_cap_growth()

    # Approximate the less-available signals using historical averages
    # In production, these would come from quarterly filings (BSE/NSE APIs)
    signals = {
        "pe_median":   round(pe_median,   2),
        "eps_growth":  round(eps_growth,  4),
        "de_median":   6.5,    # placeholder — update from quarterly data
        "pat_change":  0.0,    # placeholder
        "roe_median":  16.0,   # placeholder
        "ebitda_vol":  8.0,    # placeholder
        "mc_growth":   round(mc_growth, 4),
        "roe_change":  0.0,    # placeholder
    }
    return signals

def main():
    print(f"=== Daily Stoic-HMM Update — {date.today()} ===")

    signals = compute_signals()
    print(f"Signals: {json.dumps(signals, indent=2)}")

    print(f"Calling API at {API_URL}/predict ...")
    resp = requests.post(f"{API_URL}/predict", json=signals, timeout=30)
    result = resp.json()

    output = {
        "date":    str(date.today()),
        "signals": signals,
        "regime":  result["regime"],
        "valence": result["valence"],
        "arousal": result["arousal"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "description": result["description"],
    }

    os.makedirs("data", exist_ok=True)
    with open("data/latest_prediction.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Regime detected: {result['regime']} (confidence {result['confidence']*100:.1f}%)")
    print(f"   Valence: {result['valence']:+.3f}  |  Arousal: {result['arousal']:+.3f}")

if __name__ == "__main__":
    main()
