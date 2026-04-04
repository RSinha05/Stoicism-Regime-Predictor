# ⚖️ NIFTY 50 · Stoic-HMM Regime Detector

Detect NIFTY 50 market emotional regimes using Wundt's Valence-Arousal model and Hidden Markov Machines — rooted in Stoic philosophy.
---
## What is this?

This project detects the emotional regime of the NIFTY 50 market at any 
point in time — classifying it into one of four states borrowed directly 
from Stoic philosophy: Desire, Fear, Pleasure, or Distress.

The core insight is that markets are not just driven by numbers — they are 
driven by collective human emotion. The Stoics identified four irreducible 
irrational passions 2,300 years ago. This project maps them onto measurable 
market signals, projects them onto Wilhelm Wundt's 1896 Valence-Arousal 
emotional plane, and trains a Hidden Markov Model to detect which regime 
the market is currently in.

The alpha does not come from being inside a regime — it comes from detecting 
the transition between regimes. That is when mispricing is largest.

## How it works

1. Four Stoic proxy signals are computed from NIFTY 50 fundamentals:
   - Desire  → high PE ratio + EPS growth momentum
   - Fear    → high leverage + declining profit margins  
   - Pleasure→ high ROE + stable EBITDA margins
   - Distress→ falling market cap + eroding ROE

2. These four signals are projected onto a 2D Wundt Plane (Valence × 
   Arousal) using PCA — giving the market a single emotional coordinate 
   each year.

3. A Gaussian Hidden Markov Model with 4 hidden states is trained on this 
   coordinate sequence. Each hidden state maps to one Stoic regime.

4. The model outputs a regime label, a confidence score, and posterior 
   probabilities — telling you not just what regime you are in, but how 
   certain the model is and whether a transition is approaching.

## What was detected on real NIFTY 50 data (FY2015–FY2024)

| Year   | Regime   | Interpretation                              |
|--------|----------|---------------------------------------------|
| FY2015 | FEAR     | High leverage, PE compression               |
| FY2016 | DISTRESS | Demonetisation shadow, MCap stagnation      |
| FY2017 | DESIRE   | GST optimism, EPS momentum surge            |
| FY2018 | PLEASURE | Stable quality rally, low volatility        |
| FY2019 | PLEASURE | Complacency before IL&FS / NBFC crisis      |
| FY2020 | DESIRE   | Pre-COVID stimulus euphoria                 |
| FY2021 | DISTRESS | COVID earnings collapse                     |
| FY2022 | DESIRE   | Post-COVID reopening peak                   |
| FY2023 | DISTRESS | Rate hike stress, global risk-off           |
| FY2024 | PLEASURE | ROE expansion, quality consolidation        |

## Tech stack

- Python · scikit-learn · hmmlearn · FastAPI · Streamlit · Plotly
- Deployed on Render (backend) + Streamlit Cloud (frontend)
- GitHub Actions for scheduled data updates

## Philosophical foundation

| Stoic Passion | Wundt Quadrant              | Market Behaviour          |
|---------------|-----------------------------|---------------------------|
| Desire        | High Arousal + Positive     | PE exuberance, FOMO       |
| Fear          | High Arousal + Negative     | Panic, leverage stress    |
| Pleasure      | Low Arousal + Positive      | Complacency, low vol      |
| Distress      | Low Arousal + Negative      | Grinding erosion          |

Inspired by: Chrysippus (3rd century BC), Wilhelm Wundt (1896), 
Hamilton's regime-switching models (1989), and Lo's Adaptive 
Markets Hypothesis (2004).
## 🏗️ Project Structure

```
stoic_hmm_app/
├── backend/
│   ├── main.py              ← FastAPI backend (the brain)
│   └── requirements.txt
├── frontend/
│   ├── app.py               ← Streamlit dashboard (the face)
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml      ← Dark theme config
├── model/
│   └── artifacts.pkl        ← Trained HMM + PCA + scaler
├── scripts/
│   └── daily_update.py      ← Cron job: fetch live data + call API
├── data/
│   └── latest_prediction.json
├── .github/
│   └── workflows/
│       └── daily_update.yml ← GitHub Actions schedule
├── Dockerfile               ← Container for backend
├── render.yaml              ← Render.com deployment config
└── README.md
```

---

## 🚀 Quickstart (Local)

### Step 1 — Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/stoic-hmm.git
cd stoic-hmm

# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

### Step 2 — Start the backend

```bash
uvicorn backend.main:app --reload
# API live at http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### Step 3 — Start the frontend

```bash
cd frontend
streamlit run app.py
# Dashboard at http://localhost:8501
```

---

## ☁️ Deploy to the Cloud (Free)

### Backend → Render.com

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys
5. Copy your Render URL (e.g. `https://stoic-hmm-api.onrender.com`)

### Frontend → Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set **Main file path** to `frontend/app.py`
4. Add **Secret**: `API_URL = https://stoic-hmm-api.onrender.com`
5. Click Deploy — live in ~2 minutes

### Automate daily updates → GitHub Actions

1. In your GitHub repo → Settings → Secrets → New secret
2. Add `API_URL` = your Render URL
3. The workflow in `.github/workflows/daily_update.yml` runs automatically
   every weekday at 5:00 PM IST

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Model status |
| `/predict` | POST | Predict regime from inputs |
| `/history` | GET | All historical detections |
| `/history/{year}` | GET | Single year e.g. `/history/FY2022` |
| `/proxy_matrix` | GET | Stoic proxy signals matrix |
| `/regimes` | GET | Regime definitions |
| `/docs` | GET | Interactive API docs (Swagger) |

### Example `/predict` call

```bash
curl -X POST https://your-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pe_median":   31.0,
    "eps_growth":  0.12,
    "de_median":   6.5,
    "pat_change":  1.5,
    "roe_median":  16.0,
    "ebitda_vol":  8.0,
    "mc_growth":   0.15,
    "roe_change":  1.0
  }'
```

### Example response

```json
{
  "regime": "DESIRE",
  "valence": 1.245,
  "arousal": 0.832,
  "confidence": 0.9712,
  "color": "#D4AC0D",
  "description": "High PE exuberance + EPS momentum. Market is chasing performance.",
  "probabilities": {
    "DESIRE": 0.9712,
    "FEAR": 0.0102,
    "PLEASURE": 0.0154,
    "DISTRESS": 0.0032
  }
}
```

---

## 🧠 How It Works

```
Raw NIFTY50 Fundamentals (PE, EPS, DE, ROE, MCap...)
              ↓
    Four Stoic Proxy Signals
    (Desire · Fear · Pleasure · Distress)
              ↓
    StandardScaler → PCA (2D Wundt Projection)
    Valence axis · Arousal axis
              ↓
    Gaussian HMM (4 hidden states)
              ↓
    Regime Label + Posterior Probabilities
              ↓
    Trade Signal on Regime Transition
```

---

## 📜 Philosophical Basis

| Stoic Passion | Wundt Quadrant | Market Signal |
|---|---|---|
| Desire (Epithumia) | High Arousal + Positive Valence | PE exuberance, EPS FOMO |
| Fear (Phobos) | High Arousal + Negative Valence | Leverage stress, profit collapse |
| Pleasure (Hedone) | Low Arousal + Positive Valence | Quality complacency, low vol |
| Distress (Lupe) | Low Arousal + Negative Valence | MCap erosion, ROE grinding down |

---

## 📄 License

MIT License — free to use, modify, and deploy.
