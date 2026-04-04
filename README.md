# ⚖️ NIFTY 50 · Stoic-HMM Regime Detector

Detect NIFTY 50 market emotional regimes using Wundt's Valence-Arousal model and Hidden Markov Machines — rooted in Stoic philosophy.

---

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
