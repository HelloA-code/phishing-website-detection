"""
Phishing Website Detection — Vercel Serverless API
Flask app exposing /api/predict (POST) and /api/stats (GET).
"""

import os
import re
import json
import numpy as np
import pandas as pd
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
# MODEL — trained once at cold-start, cached in module globals
# ─────────────────────────────────────────────────────────────────────────────
_model = None
_scaler = None
_metrics = None


def _get_dataset_path():
    """Resolve dataset path relative to the project root."""
    # In Vercel, the working dir is the project root
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
    global _model, _scaler, _metrics
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
# HYBRID PREDICTION
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

    high_flags = sum(1 for f in flags if f["severity"] == "high")
    medium_flags = sum(1 for f in flags if f["severity"] == "medium")
    rule_score = min(high_flags * 0.15 + medium_flags * 0.08, 0.40)

    phish_prob_raw = 0.70 * ml_proba[1] + 0.30 * rule_score
    phish_prob = float(np.clip(phish_prob_raw, 0.0, 1.0))

    final_label = 1 if phish_prob >= 0.50 else 0
    confidence = phish_prob if final_label == 1 else 1.0 - phish_prob

    # Build serialisable features (strip internal keys)
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
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    # CORS preflight
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


# Allow local testing: python api/predict.py
if __name__ == "__main__":
    app.run(debug=True, port=5000)
