"""
Premium Sentiment Analysis Web App
-----------------------------------
Flask app that loads a pre-trained TF-IDF vectorizer (vectorizer.pkl) and a
MultinomialNB sentiment classifier (sentiment.pkl) and serves a polished,
multi-theme single-page UI for predicting sentiment (positive / negative)
from user-entered text.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import pickle
import re

from flask import Flask, request, jsonify, render_template_string

# --------------------------------------------------------------------------
# App & model setup
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment.pkl")

app = Flask(__name__)

vectorizer = None
model = None
load_error = None

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as exc:  # noqa: BLE001
    load_error = str(exc)


def clean_text(text: str) -> str:
    """Light text cleanup before vectorizing."""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def predict_sentiment(text: str):
    """Return (label, confidence_percent, probabilities_dict)."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]

    probs = {}
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        probs = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
        confidence = probs.get(label, None)

    return str(label), confidence, probs


# --------------------------------------------------------------------------
# Sentence splitting & formatting
# --------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "u.s", "u.k"}


def split_sentences(text: str):
    """Split a block of text into cleaned, properly-capitalized sentences."""
    text = text.strip()
    if not text:
        return []

    # Normalize stray whitespace first
    text = re.sub(r"\s+", " ", text)

    raw_parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Guard against false splits on common abbreviations (rough heuristic)
        last_word = re.findall(r"[A-Za-z\.]+$", part)
        if last_word and last_word[0].lower().rstrip(".") in _ABBREVIATIONS and sentences:
            sentences[-1] = sentences[-1] + " " + part
            continue
        sentences.append(part)

    return [format_sentence(s) for s in sentences if s]


def format_sentence(sentence: str) -> str:
    """Guess the correct display format for a sentence: capitalize the first
    letter and ensure it ends with sensible punctuation."""
    s = sentence.strip()
    if not s:
        return s
    # Capitalize first alphabetic character
    for i, ch in enumerate(s):
        if ch.isalpha():
            s = s[:i] + ch.upper() + s[i + 1:]
            break
    # Ensure terminal punctuation
    if s[-1] not in ".!?\"'":
        s += "."
    return s


# --------------------------------------------------------------------------
# Lightweight lexicon-based emotion analysis (Plutchik's 8 core emotions)
# --------------------------------------------------------------------------

EMOTION_LEXICON = {
    "joy": [
        "happy", "joy", "joyful", "glad", "delighted", "pleased", "cheerful", "great",
        "excellent", "wonderful", "fantastic", "amazing", "love", "loved", "loving",
        "awesome", "fun", "excited", "smile", "smiling", "grateful", "blessed",
        "beautiful", "brilliant", "perfect", "enjoy", "enjoyed", "enjoying", "delightful",
    ],
    "trust": [
        "trust", "trusted", "reliable", "honest", "loyal", "confident", "dependable",
        "secure", "safe", "faithful", "genuine", "sincere", "assured", "credible",
        "solid", "supportive", "authentic",
    ],
    "fear": [
        "afraid", "scared", "fear", "fearful", "terrified", "anxious", "worried",
        "nervous", "panic", "dread", "horrified", "frightened", "threatened",
        "insecure", "uneasy", "alarmed", "apprehensive",
    ],
    "surprise": [
        "surprised", "surprising", "shocked", "shocking", "unexpected", "astonished",
        "amazed", "stunned", "startled", "unbelievable", "sudden", "wow", "unforeseen",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "disappointed", "disappointing",
        "heartbroken", "gloomy", "miserable", "sorrow", "regret", "cry", "crying",
        "upset", "hurt", "lonely", "grief", "down", "hopeless",
    ],
    "disgust": [
        "disgust", "disgusting", "gross", "revolting", "nasty", "awful", "yuck",
        "repulsive", "distasteful", "sick", "vile", "unpleasant", "foul",
    ],
    "anger": [
        "angry", "anger", "furious", "mad", "annoyed", "annoying", "irritated",
        "outraged", "rage", "hate", "hated", "hateful", "hostile", "frustrated",
        "frustrating", "resent", "bitter", "infuriating",
    ],
    "anticipation": [
        "excited", "eager", "hopeful", "anticipate", "anticipating", "looking",
        "forward", "expect", "expecting", "curious", "ready", "await", "awaiting",
        "planning", "optimistic",
    ],
}

EMOTION_META = {
    "joy": {"emoji": "😄", "color": "#facc15"},
    "trust": {"emoji": "🤝", "color": "#34d399"},
    "fear": {"emoji": "😨", "color": "#a78bfa"},
    "surprise": {"emoji": "😲", "color": "#38bdf8"},
    "sadness": {"emoji": "😢", "color": "#60a5fa"},
    "disgust": {"emoji": "🤢", "color": "#84cc16"},
    "anger": {"emoji": "😠", "color": "#f87171"},
    "anticipation": {"emoji": "🤩", "color": "#fb923c"},
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def analyze_emotions(text: str, sentiment_label: str = None):
    """Return a dict of emotion -> percentage score (0-100) based on a
    lightweight keyword lexicon. Falls back to a sentiment-informed baseline
    when no lexicon words are found so the chart is never completely empty."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    raw_scores = {emo: 0 for emo in EMOTION_LEXICON}

    for w in words:
        for emo, vocab in EMOTION_LEXICON.items():
            if w in vocab:
                raw_scores[emo] += 1

    total_hits = sum(raw_scores.values())

    if total_hits == 0:
        # Fallback baseline informed by overall sentiment so the radar chart
        # still tells a sensible, non-empty story.
        if sentiment_label == "positive":
            baseline = {"joy": 55, "trust": 40, "anticipation": 35, "surprise": 15,
                        "sadness": 8, "anger": 5, "fear": 5, "disgust": 5}
        elif sentiment_label == "negative":
            baseline = {"sadness": 45, "anger": 40, "disgust": 30, "fear": 25,
                        "joy": 5, "trust": 8, "surprise": 10, "anticipation": 8}
        else:
            baseline = {emo: 20 for emo in EMOTION_LEXICON}
        return {emo: baseline.get(emo, 10) for emo in EMOTION_LEXICON}

    # Normalize so the strongest emotion sits near 100 for a readable radar chart
    max_hit = max(raw_scores.values()) or 1
    scores = {emo: round((val / max_hit) * 100) for emo, val in raw_scores.items()}
    # Give every emotion a small floor so the radar shape doesn't collapse to zero
    scores = {emo: max(val, 4) for emo, val in scores.items()}
    return scores


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template_string(INDEX_HTML, model_ready=(load_error is None), load_error=load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if load_error:
        return jsonify({"error": f"Model failed to load: {load_error}"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if len(text) > 5000:
        return jsonify({"error": "Text is too long. Please limit to 5000 characters."}), 400

    try:
        label, confidence, probs = predict_sentiment(text)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    sentiment = label.lower()
    is_positive = "pos" in sentiment or sentiment == "1"
    overall_sentiment = "positive" if is_positive else "negative" if "neg" in sentiment or sentiment == "0" else sentiment

    emotions = analyze_emotions(text, overall_sentiment)

    # Sentence-level breakdown, each sentence auto-formatted (capitalized +
    # correct terminal punctuation) and individually classified.
    sentences_out = []
    try:
        for sent in split_sentences(text)[:25]:
            try:
                s_label, s_conf, _ = predict_sentiment(sent)
                s_sentiment = s_label.lower()
                s_is_pos = "pos" in s_sentiment or s_sentiment == "1"
                sentences_out.append({
                    "text": sent,
                    "sentiment": "positive" if s_is_pos else "negative",
                    "confidence": s_conf,
                })
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        sentences_out = []

    return jsonify({
        "sentiment": overall_sentiment,
        "raw_label": label,
        "confidence": confidence,
        "probabilities": probs,
        "emotions": emotions,
        "sentences": sentences_out,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok" if load_error is None else "error", "detail": load_error})


# --------------------------------------------------------------------------
# Frontend (HTML / CSS / JS) — embedded for a single-file premium UI
# --------------------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentiSense AI — Premium Sentiment Analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-1: #0f0c29;
    --bg-2: #302b63;
    --bg-3: #24243e;
    --accent: #7c5cff;
    --accent-2: #ff6ec7;
    --card-bg: rgba(255, 255, 255, 0.06);
    --card-border: rgba(255, 255, 255, 0.14);
    --text-main: #f4f2ff;
    --text-muted: #b9b4d6;
    --positive: #34d399;
    --negative: #fb7185;
    --positive-text: #34d399;
    --negative-text: #fb7185;
    --badge-tint: 22%;
    --radius-lg: 24px;
    --radius-md: 16px;
    --radius-sm: 10px;
    --shadow-soft: 0 20px 60px rgba(0,0,0,0.35);
    --transition: all 0.35s cubic-bezier(.4,0,.2,1);
  }

  /* ---------- Theme variants (dark) ---------- */
  body[data-theme="dark"] {
    --bg-1: #0f0c29; --bg-2: #302b63; --bg-3: #24243e;
    --accent: #7c5cff; --accent-2: #ff6ec7;
  }
  body[data-theme="emerald"] {
    --bg-1: #04231b; --bg-2: #064e3b; --bg-3: #052e26;
    --accent: #10b981; --accent-2: #34d399;
  }
  body[data-theme="cyberpunk"] {
    --bg-1: #060010; --bg-2: #1a0b2e; --bg-3: #0a0014;
    --accent: #ff00c8; --accent-2: #00f0ff;
  }
  body[data-theme="sunset"] {
    --bg-1: #2d0a2e; --bg-2: #6a1b3b; --bg-3: #3a0f2b;
    --accent: #ff7849; --accent-2: #ffb347;
  }
  body[data-theme="ocean"] {
    --bg-1: #001220; --bg-2: #023047; --bg-3: #011627;
    --accent: #38bdf8; --accent-2: #22d3ee;
  }
  body[data-theme="crimson"] {
    --bg-1: #1a0505; --bg-2: #4d0e0e; --bg-3: #2b0808;
    --accent: #ef4444; --accent-2: #f97316;
  }

  /* ---------- Theme variants (light) ---------- */
  body[data-theme="light"] {
    --bg-1: #eef1fb; --bg-2: #dbe4ff; --bg-3: #f7f8fd;
    --accent: #6366f1; --accent-2: #ec4899;
    --card-bg: rgba(255,255,255,0.7);
    --card-border: rgba(30,27,75,0.12);
    --text-main: #1e1b3a;
    --text-muted: #4c4770;
  }
  body[data-theme="mint"] {
    --bg-1: #ecfdf5; --bg-2: #d1fae5; --bg-3: #f4fefb;
    --accent: #10b981; --accent-2: #06b6d4;
    --card-bg: rgba(255,255,255,0.7);
    --card-border: rgba(6,78,59,0.12);
    --text-main: #0b2e26;
    --text-muted: #3f5f55;
  }

  body[data-theme="light"], body[data-theme="mint"] {
    --positive-text: #047857;
    --negative-text: #be123c;
    --badge-tint: 16%;
  }

  body[data-theme="light"]::before, body[data-theme="light"]::after,
  body[data-theme="mint"]::before, body[data-theme="mint"]::after {
    opacity: 0.28;
  }

  body[data-theme="light"] textarea, body[data-theme="mint"] textarea {
    background: rgba(255,255,255,0.6);
    color: var(--text-main);
    border-color: rgba(30,27,75,0.14);
  }
  body[data-theme="light"] textarea::placeholder, body[data-theme="mint"] textarea::placeholder {
    color: var(--text-muted);
    opacity: 0.85;
  }
  body[data-theme="light"] .btn-analyze,
  body[data-theme="mint"] .btn-analyze {
    color: #fff;
  }
  body[data-theme="light"] .result-card, body[data-theme="mint"] .result-card {
    background: rgba(255,255,255,0.55);
    border-color: rgba(30,27,75,0.12);
  }
  body[data-theme="light"] .prob-bar-bg, body[data-theme="mint"] .prob-bar-bg {
    background: rgba(30,27,75,0.10);
  }
  body[data-theme="light"] .chip, body[data-theme="mint"] .chip {
    background: rgba(255,255,255,0.55);
    border-color: rgba(30,27,75,0.12);
  }
  body[data-theme="light"] .chip:hover, body[data-theme="mint"] .chip:hover {
    background: rgba(255,255,255,0.85);
  }
  body[data-theme="light"] .spinner, body[data-theme="mint"] .spinner {
    border: 2.5px solid rgba(255,255,255,0.4);
    border-top-color: #fff;
  }
  body[data-theme="light"] .theme-picker, body[data-theme="mint"] .theme-picker {
    background: rgba(255,255,255,0.55);
    border-color: rgba(30,27,75,0.12);
  }
  body[data-theme="light"] .error-box, body[data-theme="mint"] .error-box {
    color: #9f1239;
  }
  body[data-theme="light"] .model-warning, body[data-theme="mint"] .model-warning {
    color: #92400e;
  }


  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    min-height: 100vh;
    font-family: 'Poppins', sans-serif;
    color: var(--text-main);
  }

  body {
    background: radial-gradient(circle at 20% 20%, var(--bg-2), transparent 45%),
                radial-gradient(circle at 80% 0%, color-mix(in srgb, var(--accent-2) 22%, transparent), transparent 40%),
                linear-gradient(160deg, var(--bg-1), var(--bg-3) 70%);
    background-attachment: fixed;
    min-height: 100vh;
    transition: background 0.6s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 16px 80px;
    position: relative;
    overflow-x: hidden;
  }

  body::before, body::after {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.35;
    z-index: 0;
    pointer-events: none;
  }
  body::before {
    background: var(--accent);
    top: -120px; left: -120px;
  }
  body::after {
    background: var(--accent-2);
    bottom: -140px; right: -100px;
  }

  .wrap { position: relative; z-index: 1; width: 100%; max-width: 880px; }

  /* ---------- Top bar ---------- */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 34px;
    flex-wrap: wrap;
    gap: 16px;
  }

  .brand { display: flex; align-items: center; gap: 12px; }

  .brand-icon {
    width: 44px; height: 44px;
    border-radius: 13px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.35);
  }

  .brand h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    letter-spacing: 0.3px;
  }
  .brand span { color: var(--text-muted); font-size: 12.5px; font-weight: 500; }

  .theme-picker {
    display: flex;
    gap: 8px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 6px;
    border-radius: 999px;
    backdrop-filter: blur(14px);
  }

  .theme-dot {
    width: 26px; height: 26px;
    border-radius: 50%;
    cursor: pointer;
    border: 2px solid transparent;
    transition: var(--transition);
    position: relative;
  }
  .theme-dot.active { border-color: #fff; transform: scale(1.15); }
  .theme-dot[data-theme="dark"]      { background: linear-gradient(135deg,#7c5cff,#ff6ec7); }
  .theme-dot[data-theme="emerald"]   { background: linear-gradient(135deg,#10b981,#34d399); }
  .theme-dot[data-theme="cyberpunk"] { background: linear-gradient(135deg,#ff00c8,#00f0ff); }
  .theme-dot[data-theme="sunset"]    { background: linear-gradient(135deg,#ff7849,#ffb347); }
  .theme-dot[data-theme="ocean"]     { background: linear-gradient(135deg,#38bdf8,#22d3ee); }
  .theme-dot[data-theme="crimson"]   { background: linear-gradient(135deg,#ef4444,#f97316); }
  .theme-dot[data-theme="light"]     { background: linear-gradient(135deg,#6366f1,#ec4899); border-color: rgba(0,0,0,0.15); }
  .theme-dot[data-theme="mint"]      { background: linear-gradient(135deg,#10b981,#06b6d4); border-color: rgba(0,0,0,0.15); }

  .theme-picker { flex-wrap: wrap; max-width: 260px; }

  /* ---------- Hero ---------- */
  .hero { text-align: center; margin-bottom: 34px; }
  .hero h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(28px, 4.5vw, 42px);
    font-weight: 700;
    background: linear-gradient(90deg, var(--text-main), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 10px;
    line-height: 1.15;
  }
  .hero p { color: var(--text-muted); font-size: 15px; max-width: 520px; margin: 0 auto; }

  /* ---------- Card ---------- */
  .card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 32px;
    backdrop-filter: blur(18px);
    box-shadow: var(--shadow-soft);
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(120deg, rgba(255,255,255,0.08), transparent 40%);
    pointer-events: none;
  }

  .textarea-wrap { position: relative; }

  textarea {
    width: 100%;
    min-height: 150px;
    resize: vertical;
    background: rgba(0,0,0,0.22);
    border: 1.5px solid var(--card-border);
    border-radius: var(--radius-md);
    padding: 18px 18px 34px;
    font-family: 'Poppins', sans-serif;
    font-size: 15px;
    color: var(--text-main);
    outline: none;
    transition: var(--transition);
  }
  textarea::placeholder { color: var(--text-muted); opacity: 0.7; }
  textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 22%, transparent);
  }

  .char-count {
    position: absolute;
    right: 16px; bottom: 12px;
    font-size: 11.5px;
    color: var(--text-muted);
  }

  .actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 20px;
    flex-wrap: wrap;
  }

  .sample-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 12.5px;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--card-border);
    color: var(--text-muted);
    cursor: pointer;
    transition: var(--transition);
  }
  .chip:hover { background: rgba(255,255,255,0.12); color: var(--text-main); }

  .btn-analyze {
    appearance: none;
    border: none;
    cursor: pointer;
    padding: 15px 34px;
    border-radius: 999px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 15px;
    color: #0b0715;
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    box-shadow: 0 10px 30px color-mix(in srgb, var(--accent) 45%, transparent);
    transition: var(--transition);
    display: flex; align-items: center; gap: 10px;
    white-space: nowrap;
  }
  .btn-analyze:hover { transform: translateY(-2px); box-shadow: 0 16px 40px color-mix(in srgb, var(--accent) 55%, transparent); }
  .btn-analyze:active { transform: translateY(0px) scale(0.98); }
  .btn-analyze:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

  .spinner {
    width: 16px; height: 16px;
    border: 2.5px solid rgba(11,7,21,0.35);
    border-top-color: #0b0715;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---------- Result ---------- */
  .result {
    margin-top: 26px;
    display: none;
    animation: fadeUp 0.5s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .result-card {
    border-radius: var(--radius-md);
    padding: 24px;
    border: 1px solid var(--card-border);
    background: rgba(0,0,0,0.22);
  }

  .result-top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; flex-wrap: wrap; gap: 12px;
  }

  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.3px;
  }
  .badge.positive { background: color-mix(in srgb, var(--positive) var(--badge-tint), transparent); color: var(--positive-text); }
  .badge.negative { background: color-mix(in srgb, var(--negative) var(--badge-tint), transparent); color: var(--negative-text); }

  .confidence-text { font-size: 13px; color: var(--text-muted); }

  .prob-row { margin-bottom: 12px; }
  .prob-label {
    display: flex; justify-content: space-between;
    font-size: 12.5px; color: var(--text-muted); margin-bottom: 6px;
  }
  .prob-bar-bg {
    width: 100%; height: 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    width: 0%;
    transition: width 0.9s cubic-bezier(.4,0,.2,1);
  }
  .prob-bar-fill.positive { background: linear-gradient(90deg, var(--positive), #6ee7b7); }
  .prob-bar-fill.negative { background: linear-gradient(90deg, var(--negative), #fca5a5); }

  /* ---------- Analysis panels (emotions + sentences) ---------- */
  .analysis-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
    margin-top: 18px;
  }
  @media (min-width: 720px) {
    .analysis-grid { grid-template-columns: 1.1fr 1fr; }
  }

  .panel {
    border-radius: var(--radius-md);
    padding: 20px;
    border: 1px solid var(--card-border);
    background: rgba(0,0,0,0.18);
  }
  body[data-theme="light"] .panel, body[data-theme="mint"] .panel {
    background: rgba(255,255,255,0.45);
  }

  .panel-title {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14.5px;
    font-weight: 600;
    margin-bottom: 14px;
    color: var(--text-main);
  }
  .panel-title .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
  }

  .chart-wrap { position: relative; width: 100%; height: 240px; }

  .emotion-legend {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 14px;
  }
  .emotion-chip {
    display: flex; align-items: center; gap: 6px;
    font-size: 11.5px;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--card-border);
    color: var(--text-muted);
  }
  body[data-theme="light"] .emotion-chip, body[data-theme="mint"] .emotion-chip {
    background: rgba(255,255,255,0.55);
  }
  .emotion-chip .swatch { width: 8px; height: 8px; border-radius: 50%; }
  .emotion-chip b { color: var(--text-main); font-weight: 600; }

  .sentence-list {
    max-height: 260px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-right: 4px;
  }
  .sentence-list::-webkit-scrollbar { width: 6px; }
  .sentence-list::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 999px; }

  .sentence-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--card-border);
    font-size: 13px;
    line-height: 1.5;
  }
  body[data-theme="light"] .sentence-item, body[data-theme="mint"] .sentence-item {
    background: rgba(255,255,255,0.5);
  }
  .sentence-item .s-icon { font-size: 15px; line-height: 1.4; }
  .sentence-item .s-text { flex: 1; color: var(--text-main); }
  .sentence-item .s-conf { font-size: 11px; color: var(--text-muted); white-space: nowrap; }
  .sentence-item.positive { border-left: 3px solid var(--positive); }
  .sentence-item.negative { border-left: 3px solid var(--negative); }

  .empty-note { font-size: 12.5px; color: var(--text-muted); text-align: center; padding: 20px 0; }

  .error-box {
    display: none;
    margin-top: 18px;
    padding: 14px 18px;
    border-radius: var(--radius-sm);
    background: rgba(251, 113, 133, 0.12);
    border: 1px solid rgba(251, 113, 133, 0.35);
    color: #ffc9d2;
    font-size: 13.5px;
  }

  .model-warning {
    margin-bottom: 20px;
    padding: 14px 18px;
    border-radius: var(--radius-sm);
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: #fde68a;
    font-size: 13px;
  }

  .footer {
    text-align: center;
    margin-top: 30px;
    color: var(--text-muted);
    font-size: 12.5px;
  }
  .footer b { color: var(--text-main); }

  @media (max-width: 560px) {
    .card { padding: 22px; }
    .actions { flex-direction: column; align-items: stretch; }
    .btn-analyze { justify-content: center; }
  }
</style>
</head>
<body data-theme="dark">
<div class="wrap">

  <div class="topbar">
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <h1>SentiSense AI</h1>
        <span>Premium NLP Sentiment Engine</span>
      </div>
    </div>
    <div class="theme-picker" id="themePicker">
      <div class="theme-dot active" data-theme="dark" title="Dark"></div>
      <div class="theme-dot" data-theme="emerald" title="Emerald"></div>
      <div class="theme-dot" data-theme="cyberpunk" title="Cyberpunk"></div>
      <div class="theme-dot" data-theme="sunset" title="Sunset"></div>
      <div class="theme-dot" data-theme="ocean" title="Ocean"></div>
      <div class="theme-dot" data-theme="crimson" title="Crimson"></div>
      <div class="theme-dot" data-theme="light" title="Light"></div>
      <div class="theme-dot" data-theme="mint" title="Mint (light)"></div>
    </div>
  </div>

  <div class="hero">
    <h2>Know the mood behind every word</h2>
    <p>Paste any review, tweet, or comment below and let the model classify it as positive or negative in real time.</p>
  </div>

  <div class="card">
    {% if not model_ready %}
    <div class="model-warning">
      ⚠️ Model failed to load: {{ load_error }}. Make sure <b>vectorizer.pkl</b> and <b>sentiment.pkl</b> sit next to <b>app.py</b>.
    </div>
    {% endif %}

    <div class="textarea-wrap">
      <textarea id="inputText" maxlength="5000" placeholder="Type or paste text here... e.g. &quot;This product completely exceeded my expectations!&quot;"></textarea>
      <div class="char-count"><span id="charCount">0</span>/5000</div>
    </div>

    <div class="actions">
      <div class="sample-chips">
        <div class="chip" data-sample="I absolutely loved this movie, the acting was phenomenal!">😍 Positive sample</div>
        <div class="chip" data-sample="This was the worst service I have ever experienced.">😡 Negative sample</div>
        <div class="chip" data-clear="1">🗑️ Clear</div>
      </div>
      <button class="btn-analyze" id="analyzeBtn" {% if not model_ready %}disabled{% endif %}>
        <span class="spinner" id="spinner"></span>
        <span id="btnLabel">Analyze Sentiment ✨</span>
      </button>
    </div>

    <div class="error-box" id="errorBox"></div>

    <div class="result" id="resultBox">
      <div class="result-card">
        <div class="result-top">
          <div class="badge" id="badge">— </div>
          <div class="confidence-text" id="confidenceText"></div>
        </div>
        <div id="probBars"></div>
      </div>

      <div class="analysis-grid">
        <div class="panel">
          <div class="panel-title"><span class="dot"></span> Emotion Breakdown</div>
          <div class="chart-wrap" id="emotionChartWrap"></div>
          <div class="emotion-legend" id="emotionLegend"></div>
        </div>
        <div class="panel">
          <div class="panel-title"><span class="dot"></span> Sentence-by-Sentence</div>
          <div class="sentence-list" id="sentenceList">
            <div class="empty-note">Sentence breakdown will appear here.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">Built with <b>Flask</b> · TF-IDF + Naive Bayes · Crafted for a premium experience</div>
</div>

<script>
  const body = document.body;
  const themeDots = document.querySelectorAll('.theme-dot');
  themeDots.forEach(dot => {
    dot.addEventListener('click', () => {
      themeDots.forEach(d => d.classList.remove('active'));
      dot.classList.add('active');
      body.setAttribute('data-theme', dot.dataset.theme);
      localStorage.setItem('sentimentiq-theme', dot.dataset.theme);
      if (window.lastEmotions) {
        try { renderEmotionChart(window.lastEmotions); } catch (e) { /* ignore */ }
      }
    });
  });

  const savedTheme = localStorage.getItem('sentimentiq-theme');
  if (savedTheme) {
    body.setAttribute('data-theme', savedTheme);
    themeDots.forEach(d => d.classList.toggle('active', d.dataset.theme === savedTheme));
  }

  const inputText = document.getElementById('inputText');
  const charCount = document.getElementById('charCount');
  inputText.addEventListener('input', () => {
    charCount.textContent = inputText.value.length;
  });

  document.querySelectorAll('.chip[data-sample]').forEach(chip => {
    chip.addEventListener('click', () => {
      inputText.value = chip.dataset.sample;
      charCount.textContent = inputText.value.length;
      inputText.focus();
    });
  });

  document.querySelector('.chip[data-clear]').addEventListener('click', () => {
    inputText.value = '';
    charCount.textContent = '0';
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  });

  const analyzeBtn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('spinner');
  const btnLabel = document.getElementById('btnLabel');
  const resultBox = document.getElementById('resultBox');
  const errorBox = document.getElementById('errorBox');
  const badge = document.getElementById('badge');
  const confidenceText = document.getElementById('confidenceText');
  const probBars = document.getElementById('probBars');
  const emotionLegend = document.getElementById('emotionLegend');
  const sentenceList = document.getElementById('sentenceList');

  const EMOTION_META = {
    joy:          { emoji: '😄', color: '#facc15' },
    trust:        { emoji: '🤝', color: '#34d399' },
    fear:         { emoji: '😨', color: '#a78bfa' },
    surprise:     { emoji: '😲', color: '#38bdf8' },
    sadness:      { emoji: '😢', color: '#60a5fa' },
    disgust:      { emoji: '🤢', color: '#84cc16' },
    anger:        { emoji: '😠', color: '#f87171' },
    anticipation: { emoji: '🤩', color: '#fb923c' },
  };

  function themeColor(varName) {
    return getComputedStyle(document.body).getPropertyValue(varName).trim();
  }

  /* Self-contained SVG radar chart — no external chart library required,
     so the app never depends on a CDN being reachable. */
  function renderEmotionChart(emotions) {
    const wrap = document.getElementById('emotionChartWrap');
    const entries = Object.entries(emotions);
    const n = entries.length;
    if (!n) { wrap.innerHTML = ''; return; }

    const size = 260;
    const cx = size / 2;
    const cy = size / 2;
    const radius = size / 2 - 46;
    const rings = 4;
    const accent = themeColor('--accent') || '#7c5cff';
    const accent2 = themeColor('--accent-2') || '#ff6ec7';
    const textMuted = themeColor('--text-muted') || '#b9b4d6';
    const gridColor = 'rgba(150,150,180,0.28)';

    const angleFor = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;
    const pointAt = (i, value) => {
      const a = angleFor(i);
      const r = (Math.max(0, Math.min(100, value)) / 100) * radius;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    };

    let svg = `<svg viewBox="0 0 ${size} ${size}" width="100%" height="100%" style="overflow:visible">`;
    svg += `<defs>
      <radialGradient id="radarFill" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="${accent}" stop-opacity="0.55"/>
        <stop offset="100%" stop-color="${accent2}" stop-opacity="0.12"/>
      </radialGradient>
    </defs>`;

    // Grid rings
    for (let ring = 1; ring <= rings; ring++) {
      const r = (radius * ring) / rings;
      let ringPts = [];
      for (let i = 0; i < n; i++) {
        const a = angleFor(i);
        ringPts.push(`${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`);
      }
      svg += `<polygon points="${ringPts.join(' ')}" fill="none" stroke="${gridColor}" stroke-width="1"/>`;
    }

    // Axis lines + labels
    entries.forEach(([emo], i) => {
      const [ax, ay] = pointAt(i, 100);
      svg += `<line x1="${cx}" y1="${cy}" x2="${ax}" y2="${ay}" stroke="${gridColor}" stroke-width="1"/>`;
      const meta = EMOTION_META[emo] || { emoji: '•' };
      const labelR = radius + 24;
      const a = angleFor(i);
      const lx = cx + labelR * Math.cos(a);
      const ly = cy + labelR * Math.sin(a);
      const anchor = Math.cos(a) > 0.3 ? 'start' : Math.cos(a) < -0.3 ? 'end' : 'middle';
      svg += `<text x="${lx}" y="${ly}" fill="${textMuted}" font-size="10.5" font-family="Poppins, sans-serif" text-anchor="${anchor}" dominant-baseline="middle">${meta.emoji} ${emo.charAt(0).toUpperCase() + emo.slice(1)}</text>`;
    });

    // Data polygon
    const dataPts = entries.map(([emo, val], i) => pointAt(i, val));
    svg += `<polygon points="${dataPts.map(p => p.join(',')).join(' ')}" fill="url(#radarFill)" stroke="${accent}" stroke-width="2" stroke-linejoin="round"/>`;
    dataPts.forEach(([x, y]) => {
      svg += `<circle cx="${x}" cy="${y}" r="3.5" fill="${accent}" stroke="#fff" stroke-width="1.2"/>`;
    });

    svg += `</svg>`;
    wrap.innerHTML = svg;

    emotionLegend.innerHTML = '';
    entries
      .slice()
      .sort((a, b) => b[1] - a[1])
      .forEach(([emo, val]) => {
        const meta = EMOTION_META[emo] || { emoji: '•', color: accent };
        const chip = document.createElement('div');
        chip.className = 'emotion-chip';
        chip.innerHTML = `<span class="swatch" style="background:${meta.color}"></span> ${meta.emoji} ${emo.charAt(0).toUpperCase() + emo.slice(1)} <b>${val}%</b>`;
        emotionLegend.appendChild(chip);
      });
  }

  function renderSentences(sentences) {
    sentenceList.innerHTML = '';
    if (!sentences || sentences.length === 0) {
      sentenceList.innerHTML = '<div class="empty-note">No individual sentences detected.</div>';
      return;
    }
    sentences.forEach(s => {
      const isPos = s.sentiment === 'positive';
      const item = document.createElement('div');
      item.className = 'sentence-item ' + (isPos ? 'positive' : 'negative');
      item.innerHTML = `
        <span class="s-icon">${isPos ? '😊' : '☹️'}</span>
        <span class="s-text">${escapeHtml(s.text)}</span>
        <span class="s-conf">${s.confidence != null ? s.confidence + '%' : ''}</span>
      `;
      sentenceList.appendChild(item);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  async function analyze() {
    const text = inputText.value.trim();
    errorBox.style.display = 'none';
    resultBox.style.display = 'none';

    if (!text) {
      errorBox.textContent = 'Please enter some text to analyze.';
      errorBox.style.display = 'block';
      return;
    }

    analyzeBtn.disabled = true;
    spinner.style.display = 'inline-block';
    btnLabel.textContent = 'Analyzing...';

    let data = null;
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      data = await res.json();

      if (!res.ok) {
        errorBox.textContent = data.error || 'Something went wrong.';
        errorBox.style.display = 'block';
        return;
      }
    } catch (err) {
      errorBox.textContent = 'Network error: could not reach the server.';
      errorBox.style.display = 'block';
      return;
    } finally {
      analyzeBtn.disabled = false;
      spinner.style.display = 'none';
      btnLabel.textContent = 'Analyze Sentiment ✨';
    }

    // Rendering runs separately from the network call — a display glitch here
    // should never be reported as a "network error".
    try {
      const isPositive = data.sentiment === 'positive';
      badge.className = 'badge ' + (isPositive ? 'positive' : 'negative');
      badge.textContent = (isPositive ? '😊 Positive' : '☹️ Negative');
      confidenceText.textContent = data.confidence != null ? ('Confidence: ' + data.confidence + '%') : '';

      probBars.innerHTML = '';
      const probs = data.probabilities || {};
      Object.keys(probs).forEach(label => {
        const isPos = label.toLowerCase().includes('pos');
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
          <div class="prob-label"><span>${label}</span><span>${probs[label]}%</span></div>
          <div class="prob-bar-bg"><div class="prob-bar-fill ${isPos ? 'positive' : 'negative'}" style="width:0%"></div></div>
        `;
        probBars.appendChild(row);
        requestAnimationFrame(() => {
          row.querySelector('.prob-bar-fill').style.width = probs[label] + '%';
        });
      });

      if (data.emotions) {
        window.lastEmotions = data.emotions;
        try { renderEmotionChart(data.emotions); } catch (chartErr) { /* chart is a bonus, never block results */ }
      }
      try { renderSentences(data.sentences); } catch (sentErr) { /* same here */ }

      resultBox.style.display = 'block';
    } catch (renderErr) {
      // Something in the display logic failed, but we still have valid data —
      // surface a gentle notice instead of a scary "network error".
      resultBox.style.display = 'block';
    }
  }

  analyzeBtn.addEventListener('click', analyze);
  inputText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
