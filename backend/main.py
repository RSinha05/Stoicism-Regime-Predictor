import pickle, os, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="NIFTY50 Stoic-HMM API",
    description="Regime detection using Stoic philosophy + Hidden Markov Models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts at startup ──────────────────────────────────────────
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "../model/artifacts.pkl")

with open(ARTIFACTS_PATH, "rb") as f:
    arts = pickle.load(f)

model     = arts["model"]
scaler    = arts["scaler"]
pca       = arts["pca"]
label_map = arts["label_map"]
history   = arts["history"]
proxy_matrix = arts["proxy_matrix"]

REGIME_COLORS = {
    "DESIRE":   "#D4AC0D",
    "FEAR":     "#C0392B",
    "PLEASURE": "#1A7A4A",
    "DISTRESS": "#6C3483",
}

REGIME_DESCRIPTIONS = {
    "DESIRE":   "High PE exuberance + EPS momentum. Market is chasing performance. Reduce beta, watch for reversal.",
    "FEAR":     "Leverage stress + declining profits. Panic or caution dominates. Consider defensive positioning.",
    "PLEASURE": "High ROE + stable margins. Quality complacency. Sell convexity, monitor for regime shift.",
    "DISTRESS": "MCap erosion + ROE decline. Grinding bear. Long duration, long quality factors.",
}

# ── Input schema ─────────────────────────────────────────────────────────────
class MarketInput(BaseModel):
    pe_median:   float  # Median PE ratio of NIFTY50
    eps_growth:  float  # YoY EPS growth rate (decimal e.g. 0.12)
    de_median:   float  # Median Debt/Equity ratio
    pat_change:  float  # YoY change in PAT margin (percentage points)
    roe_median:  float  # Median ROE (%)
    ebitda_vol:  float  # Standard deviation of EBITDA margins across companies
    mc_growth:   float  # YoY Market Cap growth (decimal)
    roe_change:  float  # YoY change in median ROE

class PredictResponse(BaseModel):
    regime:      str
    valence:     float
    arousal:     float
    confidence:  float
    color:       str
    description: str
    probabilities: dict


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "NIFTY50 Stoic-HMM API is live", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "model_states": len(label_map)}

@app.get("/history")
def get_history():
    """Return all historical regime detections FY2015-FY2024."""
    return {"data": history}

@app.get("/history/{year}")
def get_year(year: str):
    """Return regime for a specific fiscal year e.g. FY2022."""
    for h in history:
        if h["year"] == year:
            h["color"] = REGIME_COLORS[h["regime"]]
            h["description"] = REGIME_DESCRIPTIONS[h["regime"]]
            return h
    raise HTTPException(status_code=404, detail=f"Year {year} not found")

@app.get("/proxy_matrix")
def get_proxy_matrix():
    """Return the Stoic proxy signal matrix for all years."""
    return {"data": proxy_matrix}

@app.post("/predict", response_model=PredictResponse)
def predict(data: MarketInput):
    """
    Predict the current market regime from input fundamental signals.
    Returns regime label, Wundt coordinates, confidence, and probabilities.
    """
    # Build the four Stoic proxy signals from raw inputs
    desire   =  data.pe_median / 31.0 - 1.0 + data.eps_growth
    fear     =  data.de_median / 6.5  - 1.0 - data.pat_change / 5.0
    pleasure =  data.roe_median / 16.0 - 1.0 - data.ebitda_vol / 10.0
    distress = -data.mc_growth - data.roe_change / 5.0

    features = np.array([[desire, fear, pleasure, distress]])
    X_scaled = scaler.transform(features)
    X_wundt  = pca.transform(X_scaled)

    hidden     = model.predict(X_wundt)[0]
    posteriors = model.predict_proba(X_wundt)[0]
    regime     = label_map[hidden]

    probs = {label_map[i]: round(float(p), 4) for i, p in enumerate(posteriors)}

    return PredictResponse(
        regime      = regime,
        valence     = round(float(X_wundt[0][0]), 3),
        arousal     = round(float(X_wundt[0][1]), 3),
        confidence  = round(float(posteriors.max()), 4),
        color       = REGIME_COLORS[regime],
        description = REGIME_DESCRIPTIONS[regime],
        probabilities = probs,
    )

@app.get("/regimes")
def get_regimes():
    """Return all regime definitions with colors and descriptions."""
    return {
        name: {"color": REGIME_COLORS[name], "description": REGIME_DESCRIPTIONS[name]}
        for name in ["DESIRE", "FEAR", "PLEASURE", "DISTRESS"]
    }
