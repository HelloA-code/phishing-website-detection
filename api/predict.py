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
from collections import Counter
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength",
    "NoOfSubDomain", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpecialCharRatioURL",
    "IsHTTPS", "CharContinuationRate", "URLCharProb",
]

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "account",
    "update", "confirm", "paypal", "signin", "password",
    "validate", "authenticate", "billing", "suspend",
    "alert", "unlock", "expire", "urgent", "wallet",
]

RISKY_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "buzz", "xyz", "top",
    "club", "work", "info", "online", "site", "icu", "cam",
]

BRAND_DOMAINS = {
    "paypal": "paypal.com", "apple": "apple.com",
    "google": "google.com", "facebook": "facebook.com",
    "microsoft": "microsoft.com", "amazon": "amazon.com",
    "netflix": "netflix.com", "instagram": "instagram.com",
    "twitter": "twitter.com", "linkedin": "linkedin.com",
    "whatsapp": "whatsapp.com", "dropbox": "dropbox.com",
}

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rb.gy",
]

URL_LONG_THRESHOLD = 54

# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
DARK_AX   = "#161b22"
CLR_GRID  = "#21262d"
CLR_TEXT  = "#c9d1d9"
CLR_PHISH = "#f85149"
CLR_LEGIT = "#3fb950"
CLR_ACCENT = "#388bfd"
CLR_WARN  = "#d29922"

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

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )
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
def _char_continuation_rate(s):
    if len(s) <= 1:
        return 0.0
    def ctype(c):
        if c.isalpha(): return 0
        if c.isdigit(): return 1
        return 2
    pairs = sum(1 for i in range(1, len(s)) if ctype(s[i]) == ctype(s[i - 1]))
    return round(pairs / (len(s) - 1), 6)


def _url_char_prob(s):
    if not s:
        return 0.0
    freq = Counter(s)
    total = len(s)
    return round(sum(freq[c] / total for c in s) / total, 9)


def _get_tld(domain):
    parts = domain.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else ""


def extract_features(url):
    url = url.strip()
    full_url = url if "://" in url else "http://" + url
    parsed = urlparse(full_url)

    domain = parsed.netloc or url.split("/")[0]
    url_len = len(url)
    domain_len = len(domain)
    tld = _get_tld(domain)

    n_letters = sum(1 for c in url if c.isalpha())
    n_digits  = sum(1 for c in url if c.isdigit())
    n_equals  = url.count("=")
    n_qmark   = url.count("?")
    n_amp     = url.count("&")
    other_special = len(re.findall(r"[^a-zA-Z0-9.\-]", url))
    spec_ratio = other_special / url_len if url_len else 0.0

    is_https = 1 if parsed.scheme == "https" else 0
    is_ip    = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
    dot_count = domain.count(".")
    n_subdomains = max(dot_count - 1, 0)

    return {
        "URLLength": url_len,
        "DomainLength": domain_len,
        "IsDomainIP": is_ip,
        "TLDLength": len(tld),
        "NoOfSubDomain": n_subdomains,
        "NoOfLettersInURL": n_letters,
        "LetterRatioInURL": round(n_letters / url_len, 4) if url_len else 0.0,
        "NoOfDegitsInURL": n_digits,
        "DegitRatioInURL": round(n_digits / url_len, 4) if url_len else 0.0,
        "NoOfEqualsInURL": n_equals,
        "NoOfQMarkInURL": n_qmark,
        "NoOfAmpersandInURL": n_amp,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpecialCharRatioURL": round(spec_ratio, 4),
        "IsHTTPS": is_https,
        "CharContinuationRate": _char_continuation_rate(url),
        "URLCharProb": _url_char_prob(url),
        "_url": url,
        "_domain": domain,
        "_tld": tld,
        "_has_at": "@" in url,
        "_has_ip": is_ip == 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FLAGS
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_flags(feats):
    flags = []
    url_lower = feats["_url"].lower()
    domain = feats["_domain"].lower()
    tld = feats.get("_tld", "").lower()

    def add(rule, sev, lbl):
        flags.append({"rule": rule, "severity": sev, "label": lbl})

    # HIGH
    if feats["_has_at"]:
        add("Contains '@' character in URL", "high", "danger")
    if feats["_has_ip"]:
        add("Domain is a raw IP address", "high", "danger")
    if feats["URLLength"] > URL_LONG_THRESHOLD:
        add(f"URL is very long ({feats['URLLength']} chars)", "high", "danger")
    for brand, legit_domain in BRAND_DOMAINS.items():
        if brand in domain and legit_domain not in domain:
            add(f"Possible brand impersonation: '{brand}'", "high", "danger")
            break
    if "xn--" in domain:
        add("Punycode / internationalized domain (possible homograph)", "high", "danger")
    if tld in RISKY_TLDS:
        add(f"High-risk TLD: '.{tld}'", "high", "danger")
    for shortener in URL_SHORTENERS:
        if shortener in domain:
            add(f"URL shortener detected: '{shortener}'", "high", "danger")
            break

    # MEDIUM
    kw_found = [k for k in SUSPICIOUS_KEYWORDS if k in url_lower]
    for kw in kw_found[:3]:
        add(f"Suspicious keyword: '{kw}'", "medium", "warn")
    if not feats["IsHTTPS"]:
        add("No HTTPS (unencrypted connection)", "medium", "warn")
    if feats["NoOfSubDomain"] >= 3:
        add(f"Excessive subdomains ({feats['NoOfSubDomain']})", "medium", "warn")
    if domain.count("-") >= 3:
        add(f"Many hyphens in domain ({domain.count('-')})", "medium", "warn")
    if feats["NoOfDegitsInURL"] > 8:
        add(f"Many digits in URL ({feats['NoOfDegitsInURL']})", "medium", "warn")

    # LOW
    if feats["SpecialCharRatioURL"] > 0.12:
        add(f"High special-char ratio ({feats['SpecialCharRatioURL']:.2%})", "low", "warn")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_predict(url):
    _ensure_model()

    feats = extract_features(url)
    flags = rule_based_flags(feats)

    X = np.array([[feats[f] for f in FEATURES]])
    X_sc = _scaler.transform(X)
    ml_proba = _model.predict_proba(X_sc)[0]
    ml_label = int(_model.predict(X_sc)[0])
    ml_conf = float(ml_proba[ml_label])

    high_flags   = sum(1 for f in flags if f["severity"] == "high")
    medium_flags = sum(1 for f in flags if f["severity"] == "medium")
    low_flags    = sum(1 for f in flags if f["severity"] == "low")
    rule_score   = min(high_flags * 0.30 + medium_flags * 0.15 + low_flags * 0.05, 1.0)

    if rule_score >= 0.60:
        phish_prob_raw = 0.25 * ml_proba[1] + 0.75 * rule_score
    elif rule_score >= 0.30:
        phish_prob_raw = 0.40 * ml_proba[1] + 0.60 * rule_score
    elif rule_score > 0:
        phish_prob_raw = 0.55 * ml_proba[1] + 0.45 * rule_score
    else:
        phish_prob_raw = ml_proba[1]

    phish_prob = float(np.clip(max(phish_prob_raw, rule_score * 0.70), 0.0, 1.0))
    final_label = 1 if phish_prob >= 0.40 else 0
    confidence = phish_prob if final_label == 1 else 1.0 - phish_prob

    clean_feats = {k: feats[k] for k in FEATURES}
    clean_feats["has_at"] = feats["_has_at"]
    clean_feats["has_ip"] = feats["_has_ip"]

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
# CHART HELPERS (dynamic, per-analysis)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list or fig.axes):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=CLR_TEXT, labelsize=9)
        ax.xaxis.label.set_color(CLR_TEXT)
        ax.yaxis.label.set_color(CLR_TEXT)
        ax.title.set_color(CLR_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CLR_GRID)
        ax.grid(color=CLR_GRID, linewidth=0.6, alpha=0.4)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_probability_gauge(phish_prob):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    legit_p = 1.0 - phish_prob
    ax.barh(0, legit_p, height=0.6, color=CLR_LEGIT, alpha=0.85,
            label=f"Legitimate  {legit_p:.1%}", edgecolor="none")
    ax.barh(0, phish_prob, height=0.6, left=legit_p, color=CLR_PHISH,
            alpha=0.85, label=f"Phishing  {phish_prob:.1%}", edgecolor="none")
    ax.axvline(x=0.40, color=CLR_WARN, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(0.40, 0.48, "Threshold", ha="center", va="bottom",
            color=CLR_WARN, fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Phishing Probability Breakdown", fontsize=11, fontweight="bold")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID,
                    loc="upper right", ncol=2)
    for t in leg.get_texts():
        t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_feature_comparison(feats):
    _ensure_model()
    keys = ["URLLength", "DomainLength", "NoOfSubDomain",
            "NoOfLettersInURL", "NoOfDegitsInURL"]
    labels = ["URL Len", "Domain Len", "Subdomains", "Letters", "Digits"]
    user_vals = [feats.get(k, 0) for k in keys]
    avg_vals = [float(_df[k].mean()) for k in keys]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - w/2, user_vals, w, label="Your URL",
           color=CLR_ACCENT, edgecolor=CLR_GRID, linewidth=0.5, alpha=0.9)
    ax.bar(x + w/2, avg_vals, w, label="Dataset Avg",
           color="#8b949e", edgecolor=CLR_GRID, linewidth=0.5, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("Feature Comparison: Your URL vs Dataset", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID)
    for t in leg.get_texts():
        t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_risk_donut(flags):
    high   = sum(1 for f in flags if f["severity"] == "high")
    medium = sum(1 for f in flags if f["severity"] == "medium")
    low    = sum(1 for f in flags if f["severity"] == "low")
    safe   = 1 if (high + medium + low) == 0 else 0

    sizes  = [v for v in [high, medium, low, safe] if v > 0]
    labels = [l for l, v in [(f"High ({high})", high), (f"Medium ({medium})", medium),
              (f"Low ({low})", low), ("No Issues", safe)] if v > 0]
    colors = [c for c, v in [(CLR_PHISH, high), (CLR_WARN, medium),
              (CLR_ACCENT, low), (CLR_LEGIT, safe)] if v > 0]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        textprops={"color": CLR_TEXT, "fontsize": 9},
        wedgeprops={"width": 0.4, "edgecolor": DARK_BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold"); at.set_fontsize(10)
    ax.set_title("Risk Severity Breakdown", fontsize=11,
                 fontweight="bold", color=CLR_TEXT)
    fig.patch.set_facecolor(DARK_BG)
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
    """Generate dynamic per-analysis charts."""
    if request.method == "OPTIONS":
        resp = jsonify({})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp, 200

    try:
        data = request.get_json(force=True)
        feats = data.get("features", {})
        flags = data.get("flags", [])
        phish_prob = float(data.get("phish_prob", 0.0))

        charts_data = {
            "probability_gauge": chart_probability_gauge(phish_prob),
            "feature_comparison": chart_feature_comparison(feats),
            "risk_donut": chart_risk_donut(flags),
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
