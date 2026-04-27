"""
Phishing Website Detection — Vercel Serverless API
Flask app exposing /api/predict (POST), /api/stats (GET), and /api/charts (POST).
"""

import os
import re
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

FEATURES = [
    "URLLength", "DomainLength", "SpecialCharRatioURL",
    "IsHTTPS", "NoOfSubDomain",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "confirm", "paypal", "signin", "password",
    "validate", "authenticate", "billing", "suspend",
]

URL_LONG_THRESHOLD = 75

# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
DARK_AX   = "#161b22"
CLR_GRID  = "#21262d"
CLR_TEXT  = "#c9d1d9"
CLR_PHISH = "#f85149"
CLR_LEGIT = "#3fb950"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — trained once at cold-start, cached in module globals
# ─────────────────────────────────────────────────────────────────────────────
_model = None
_scaler = None
_metrics = None
_df = None


def _get_dataset_path():
    """Resolve dataset path relative to the project root."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_dataset.csv"),
        os.path.join(os.getcwd(), "sample_dataset.csv"),
        "sample_dataset.csv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("sample_dataset.csv not found")


def _ensure_model():
    """Lazy-load and train the model on first request."""
    global _model, _scaler, _metrics, _df
    if _model is not None:
        return

    path = _get_dataset_path()
    df = pd.read_csv(path)

    required = set(FEATURES + ["label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURES].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    _model = model
    _scaler = scaler
    _df = df
    _metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "report": {k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v)
                   for k, v in report.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_counts": {int(k): int(v) for k, v in y.value_counts().items()},
        "total_samples": int(len(df)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)

    domain = parsed.netloc or url.split("/")[0]
    url_len = len(url)
    domain_len = len(domain)

    special_chars = re.findall(r"[^a-zA-Z0-9.\-]", url)
    spec_ratio = len(special_chars) / url_len if url_len else 0.0

    is_https = 1 if parsed.scheme == "https" else 0

    dot_count = domain.count(".")
    n_subdomains = max(dot_count - 1, 0)

    return {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "SpecialCharRatioURL": round(spec_ratio, 4),
        "IsHTTPS": is_https,
        "NoOfSubDomain": n_subdomains,
        "_url": url,
        "_domain": domain,
        "_has_at": "@" in url,
        "_has_ip": bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}", domain)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FLAGS
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_flags(feats: dict) -> list:
    flags = []

    def add(rule, sev, lbl):
        flags.append({"rule": rule, "severity": sev, "label": lbl})

    if feats["_has_at"]:
        add("Contains '@' character in URL", "high", "danger")

    if feats["URLLength"] > URL_LONG_THRESHOLD:
        add(f"URL is very long ({feats['URLLength']} chars)", "high", "danger")

    kw_found = [k for k in SUSPICIOUS_KEYWORDS if k in feats["_url"].lower()]
    for kw in kw_found:
        add(f"Suspicious keyword: '{kw}'", "medium", "warn")

    if not feats["IsHTTPS"]:
        add("No HTTPS (unencrypted connection)", "medium", "warn")

    if feats["NoOfSubDomain"] >= 3:
        add(f"Excessive subdomains ({feats['NoOfSubDomain']})", "medium", "warn")

    if feats["_has_ip"]:
        add("Domain is a raw IP address", "high", "danger")

    if feats["SpecialCharRatioURL"] > 0.15:
        add(f"High special-char ratio ({feats['SpecialCharRatioURL']:.2%})", "low", "warn")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PREDICTION  (improved blending weights)
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_predict(url: str) -> dict:
    _ensure_model()

    feats = extract_features(url)
    flags = rule_based_flags(feats)

    X = np.array([[
        feats["URLLength"],
        feats["DomainLength"],
        feats["SpecialCharRatioURL"],
        feats["IsHTTPS"],
        feats["NoOfSubDomain"],
    ]])
    X_sc = _scaler.transform(X)
    ml_proba = _model.predict_proba(X_sc)[0]
    ml_label = int(_model.predict(X_sc)[0])
    ml_conf = float(ml_proba[ml_label])

    # Rule contribution — more aggressive weights so flagged URLs
    # are properly pushed toward "phishing"
    high_flags = sum(1 for f in flags if f["severity"] == "high")
    medium_flags = sum(1 for f in flags if f["severity"] == "medium")
    low_flags = sum(1 for f in flags if f["severity"] == "low")
    rule_score = min(high_flags * 0.25 + medium_flags * 0.12 + low_flags * 0.05, 1.0)

    # Blend: 55% ML, 45% rules — gives rules enough influence
    phish_prob_raw = 0.55 * ml_proba[1] + 0.45 * rule_score
    phish_prob = float(np.clip(phish_prob_raw, 0.0, 1.0))

    final_label = 1 if phish_prob >= 0.50 else 0
    confidence = phish_prob if final_label == 1 else 1.0 - phish_prob

    clean_feats = {
        "URLLength": feats["URLLength"],
        "DomainLength": feats["DomainLength"],
        "SpecialCharRatioURL": feats["SpecialCharRatioURL"],
        "IsHTTPS": feats["IsHTTPS"],
        "NoOfSubDomain": feats["NoOfSubDomain"],
        "has_at": feats["_has_at"],
        "has_ip": feats["_has_ip"],
    }

    return {
        "url": url,
        "features": clean_feats,
        "flags": flags,
        "ml_label": ml_label,
        "ml_conf": round(ml_conf, 4),
        "phish_prob": round(phish_prob, 4),
        "final_label": final_label,
        "confidence": round(confidence, 4),
        "is_phishing": final_label == 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATION HELPERS  (matplotlib → base64 PNG)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_style(fig, ax_list=None):
    """Apply dark theme to a matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=CLR_TEXT, labelsize=9)
        ax.xaxis.label.set_color(CLR_TEXT)
        ax.yaxis.label.set_color(CLR_TEXT)
        ax.title.set_color(CLR_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CLR_GRID)
        ax.grid(color=CLR_GRID, linewidth=0.6, alpha=0.5)


def _fig_to_b64(fig) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_feature_comparison(feats: dict) -> str:
    """Bar chart comparing analysed URL features vs dataset averages."""
    _ensure_model()

    feat_names = ["URL Length", "Domain Len", "Special %", "HTTPS", "Subdomains"]
    user_vals = [
        feats["URLLength"],
        feats["DomainLength"],
        feats["SpecialCharRatioURL"] * 100,
        feats["IsHTTPS"],
        feats["NoOfSubDomain"],
    ]

    # Dataset averages
    avg_vals = [
        float(_df["URLLength"].mean()),
        float(_df["DomainLength"].mean()),
        float(_df["SpecialCharRatioURL"].mean()) * 100,
        float(_df["IsHTTPS"].mean()),
        float(_df["NoOfSubDomain"].mean()),
    ]

    x = np.arange(len(feat_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars1 = ax.bar(x - width/2, user_vals, width, label="Your URL",
                   color="#388bfd", edgecolor=CLR_GRID, linewidth=0.6, alpha=0.9)
    bars2 = ax.bar(x + width/2, avg_vals, width, label="Dataset Avg",
                   color="#8b949e", edgecolor=CLR_GRID, linewidth=0.6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=8.5)
    ax.set_title("Feature Comparison: Your URL vs Dataset Average", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID, loc="upper right")
    for t in leg.get_texts():
        t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_probability_gauge(phish_prob: float, confidence: float) -> str:
    """Horizontal gauge showing ML probability breakdown."""
    fig, ax = plt.subplots(figsize=(7, 2))

    # Stacked horizontal bar
    legit_prob = 1.0 - phish_prob
    ax.barh(0, legit_prob, height=0.5, color=CLR_LEGIT, alpha=0.85, label=f"Legitimate ({legit_prob:.1%})")
    ax.barh(0, phish_prob, height=0.5, left=legit_prob, color=CLR_PHISH, alpha=0.85, label=f"Phishing ({phish_prob:.1%})")

    # Threshold line
    ax.axvline(x=0.5, color="#d29922", linewidth=2, linestyle="--", alpha=0.8)
    ax.text(0.5, 0.38, "Threshold", ha="center", va="bottom", color="#d29922", fontsize=8, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Phishing Probability Breakdown", fontsize=11, fontweight="bold")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID, loc="upper right",
                    ncol=2, bbox_to_anchor=(1, -0.1))
    for t in leg.get_texts():
        t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_dataset_distribution() -> str:
    """Pie chart of dataset label distribution."""
    _ensure_model()
    label_counts = _metrics["label_counts"]
    legit = label_counts.get(0, label_counts.get("0", 0))
    phish = label_counts.get(1, label_counts.get("1", 0))

    fig, ax = plt.subplots(figsize=(4, 3.5))
    sizes = [legit, phish]
    labels = [f"Legitimate\n({legit})", f"Phishing\n({phish})"]
    colors = [CLR_LEGIT, CLR_PHISH]
    explode = (0.04, 0.04)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": CLR_TEXT, "fontsize": 9},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax.set_title("Dataset Distribution", fontsize=11, fontweight="bold", color=CLR_TEXT)
    fig.patch.set_facecolor(DARK_BG)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_url_length_histogram() -> str:
    """Histogram of URL lengths in the dataset, split by label."""
    _ensure_model()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    legit = _df[_df.label == 0]["URLLength"]
    phish = _df[_df.label == 1]["URLLength"]

    ax.hist(legit, bins=25, color=CLR_LEGIT, alpha=0.7, label="Legitimate", edgecolor=DARK_AX)
    ax.hist(phish, bins=25, color=CLR_PHISH, alpha=0.7, label="Phishing", edgecolor=DARK_AX)
    ax.set_title("URL Length Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("URL Length (chars)")
    ax.set_ylabel("Frequency")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID)
    for t in leg.get_texts():
        t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_confusion_matrix() -> str:
    """Heatmap of the confusion matrix."""
    _ensure_model()
    cm = np.array(_metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")

    labels = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else CLR_TEXT,
                    fontsize=16, fontweight="bold")

    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp, 200

    try:
        data = request.get_json(force=True)
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        result = hybrid_predict(url)
        resp = jsonify(result)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except Exception as exc:
        resp = jsonify({"error": str(exc)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


@app.route("/api/stats", methods=["GET", "OPTIONS"])
def stats():
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp, 200

    try:
        _ensure_model()
        resp = jsonify(_metrics)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except Exception as exc:
        resp = jsonify({"error": str(exc)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


@app.route("/api/charts", methods=["POST", "OPTIONS"])
def charts():
    """Generate Python matplotlib charts based on analysis results."""
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp, 200

    try:
        data = request.get_json(force=True)
        feats = data.get("features", {})
        phish_prob = float(data.get("phish_prob", 0.0))
        confidence = float(data.get("confidence", 0.0))

        # Generate all charts
        charts_data = {
            "feature_comparison": chart_feature_comparison(feats),
            "probability_gauge": chart_probability_gauge(phish_prob, confidence),
            "dataset_distribution": chart_dataset_distribution(),
            "url_length_histogram": chart_url_length_histogram(),
            "confusion_matrix": chart_confusion_matrix(),
        }

        resp = jsonify(charts_data)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except Exception as exc:
        resp = jsonify({"error": str(exc)})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 500


# Allow local testing: python api/predict.py
if __name__ == "__main__":
    app.run(debug=True, port=5000)
