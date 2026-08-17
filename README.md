# Sentiment-Analysis

https://sentiment-analysis-ukri.onrender.com/

# SentiSense AI

A premium, single-file Flask web app for sentiment analysis — built on a TF-IDF vectorizer + Multinomial Naive Bayes classifier, with an emotion-breakdown radar chart and sentence-by-sentence analysis layered on top.

## ✨ Features

- **Sentiment prediction** — classifies text as positive or negative using your trained `TfidfVectorizer` + `MultinomialNB` model, with a confidence score and probability breakdown.
- **Emotion analysis** — a lightweight keyword-based engine scores text across 8 core emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation) and renders them as a radar chart, drawn in pure SVG (no external chart library / CDN dependency).
- **Sentence-by-sentence breakdown** — automatically splits multi-sentence input, auto-formats each sentence (capitalization + terminal punctuation), and classifies each one individually.
- **8 premium themes** — 6 dark (Dark, Emerald, Cyberpunk, Sunset, Ocean, Crimson) and 2 light (Light, Mint), switchable on the fly and saved to `localStorage`.
- **Polished UI** — glassmorphism cards, gradient accents, animated confidence bars, sample text chips, and a fully responsive layout — all embedded in a single `app.py` file (HTML/CSS/JS via `render_template_string`).
- **Robust error handling** — network/server errors and rendering errors are handled separately so a display hiccup is never mistaken for a connectivity issue.

---

## 📁 Project Structure

```
.
├── app.py              # Flask app — backend logic + embedded frontend (HTML/CSS/JS)
├── requirements.txt     # Python dependencies
├── vectorizer.pkl       # Trained TfidfVectorizer (required)
├── sentiment.pkl        # Trained MultinomialNB sentiment classifier (required)
└── README.md            # This file
```

> **Important:** `vectorizer.pkl` and `sentiment.pkl` must sit in the **same folder** as `app.py`. The app loads them by path relative to `app.py` at startup.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or later
- pip

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in your browser
```
http://127.0.0.1:5000
```

The app runs in debug mode by default (auto-reload on code changes). For production, run it behind a proper WSGI server (e.g. `gunicorn app:app`) and set `debug=False` in `app.py`.

---

## 🧠 How It Works

| Step | Description |
|---|---|
| 1. Text cleanup | Strips URLs and normalizes whitespace before vectorizing. |
| 2. Vectorize | Transforms the cleaned text using the loaded `TfidfVectorizer`. |
| 3. Predict | `MultinomialNB.predict()` / `predict_proba()` returns the label and class probabilities. |
| 4. Emotion scoring | A built-in keyword lexicon (covering 8 core emotions) scans the raw text and produces a normalized 0–100 score per emotion. If no emotion keywords are found, a sensible baseline is derived from the overall sentiment so the chart is never empty. |
| 5. Sentence splitting | The input is split into sentences on `.`/`!`/`?` boundaries (with basic abbreviation handling), each one auto-capitalized and punctuated, then classified individually. |

### API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI. |
| `POST` | `/predict` | Accepts `{ "text": "..." }`, returns sentiment, confidence, probabilities, emotions, and sentence breakdown as JSON. |
| `GET` | `/health` | Returns `{ "status": "ok" }` if the models loaded successfully, or an error detail otherwise. |

**Example request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I loved this so much, it was fantastic! But the delivery was late."}'
```

**Example response:**
```json
{
  "sentiment": "positive",
  "raw_label": "positive",
  "confidence": 68.42,
  "probabilities": { "negative": 31.58, "positive": 68.42 },
  "emotions": { "joy": 100, "anger": 50, "trust": 4, "...": "..." },
  "sentences": [
    { "text": "I loved this so much, it was fantastic!", "sentiment": "positive", "confidence": 79.25 },
    { "text": "But the delivery was late.", "sentiment": "negative", "confidence": 61.1 }
  ]
}
```

---

## 🎨 Themes

| Theme | Mode |
|---|---|
| Dark | Dark |
| Emerald | Dark |
| Cyberpunk | Dark |
| Sunset | Dark |
| Ocean | Dark |
| Crimson | Dark |
| Light | Light |
| Mint | Light |

Themes are switched via the color-dot picker in the top-right corner and persisted across sessions using `localStorage`.

---

## ⚠️ Limitations

- The underlying model is **binary** (positive/negative) — it does not natively output emotions or multi-class sentiment. Emotion scores come from a supplementary keyword-based heuristic, not the trained model itself, and should be treated as an illustrative signal rather than a precise measurement.
- Sentence splitting uses a simple punctuation-based heuristic and may occasionally misplit unusual formatting (e.g. abbreviations, ellipses, decimal numbers).
- Model accuracy depends entirely on the training data used to produce `vectorizer.pkl` and `sentiment.pkl` — this app only serves the model, it does not retrain it.

---

## 🛠️ Tech Stack

- **Backend:** Flask, scikit-learn (TfidfVectorizer + MultinomialNB), NumPy, SciPy
- **Frontend:** Vanilla HTML/CSS/JS (single file, no build step), hand-rolled SVG radar chart
- **Fonts:** Poppins, Space Grotesk (Google Fonts)
