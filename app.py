"""
Phishing Website Detection using Machine Learning and Hybrid Risk Analysis
A complete, production-ready Streamlit application.
"""

import re
import warnings
from urllib.parse import urlparse
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / STYLES
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] { background-color: #161b22 !important; }

/* ── Headings ── */
h1 { color: #58a6ff !important; font-weight: 800; letter-spacing: -0.5px; }
h2 { color: #79c0ff !important; font-weight: 700; }
h3 { color: #a5d6ff !important; font-weight: 600; }

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.card-danger  { border-color: #f85149; background: #1a0f0f; }
.card-success { border-color: #3fb950; background: #0d1f10; }
.card-warn    { border-color: #d29922; background: #1a1500; }
.card-info    { border-color: #388bfd; background: #0d1832; }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px 3px;
}
.badge-danger  { background:#f8514933; color:#f85149; border:1px solid #f85149; }
.badge-success { background:#3fb95033; color:#3fb950; border:1px solid #3fb950; }
.badge-warn    { background:#d2992233; color:#d29922; border:1px solid #d29922; }
.badge-info    { background:#388bfd33; color:#79c0ff; border:1px solid #388bfd; }

/* ── Metric row ── */
.metric-box {
    background:#1c2128;
    border:1px solid #30363d;
    border-radius:10px;
    padding:14px 18px;
    text-align:center;
}
.metric-box .val { font-size:1.8rem; font-weight:800; color:#58a6ff; }
.metric-box .lbl { font-size:0.78rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; }

/* ── Input ── */
[data-testid="stTextInput"] input {
    background:#21262d !important;
    border:1px solid #30363d !important;
    border-radius:8px !important;
    color:#c9d1d9 !important;
    font-size:0.95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color:#388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.15) !important;
}

/* ── Buttons ── */
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div { border-radius: 99px; }

/* ── Divider ── */
hr { border-color: #21262d; margin: 20px 0; }

/* ── Table ── */
[data-testid="stTable"] { background:#161b22; }
thead th { background:#21262d !important; color:#79c0ff !important; }
tbody tr:nth-child(even) td { background:#1c2128 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background:#161b22 !important;
    border:1px solid #30363d !important;
    border-radius:10px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius:10px !important; }

/* ── Section separator ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 32px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #21262d;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
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

DATASET_PATH = "Phishing_Website_Detection.csv"
URL_LONG_THRESHOLD = 54
HISTORY_MAX = 10

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=HISTORY_MAX)
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "train_metrics" not in st.session_state:
    st.session_state.train_metrics = {}
if "df" not in st.session_state:
    st.session_state.df = None

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(path: str):
    """Load dataset, train RandomForest, return model + metrics."""
    df = pd.read_csv(path)
    required = set(FEATURES + ["label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    X = df[FEATURES].copy()
    y = df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_counts": y.value_counts().to_dict(),
    }
    return model, scaler, metrics, df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _char_continuation_rate(s: str) -> float:
    """Fraction of consecutive same-type character pairs."""
    if len(s) <= 1:
        return 0.0
    def ctype(c):
        if c.isalpha(): return 0
        if c.isdigit(): return 1
        return 2
    pairs = sum(1 for i in range(1, len(s)) if ctype(s[i]) == ctype(s[i - 1]))
    return round(pairs / (len(s) - 1), 6)


def _url_char_prob(s: str) -> float:
    """Average character frequency-based probability."""
    if not s:
        return 0.0
    from collections import Counter
    freq = Counter(s)
    total = len(s)
    return round(sum(freq[c] / total for c in s) / total, 9)


def _get_tld(domain: str) -> str:
    """Extract TLD from domain."""
    parts = domain.rsplit(".", 1)
    return parts[-1] if len(parts) > 1 else ""


def extract_features(url: str) -> dict:
    """Extract all 17 ML features + extras for rule-based logic."""
    url = url.strip()
    full_url = url if "://" in url else "http://" + url
    parsed = urlparse(full_url)

    domain     = parsed.netloc or url.split("/")[0]
    url_len    = len(url)
    domain_len = len(domain)
    tld        = _get_tld(domain)

    # Character counts
    n_letters = sum(1 for c in url if c.isalpha())
    n_digits  = sum(1 for c in url if c.isdigit())
    n_equals  = url.count("=")
    n_qmark   = url.count("?")
    n_amp     = url.count("&")
    other_special = len(re.findall(r"[^a-zA-Z0-9.\-]", url))
    spec_ratio = other_special / url_len if url_len else 0.0

    is_https     = 1 if parsed.scheme == "https" else 0
    is_ip        = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
    dot_count    = domain.count(".")
    n_subdomains = max(dot_count - 1, 0)

    return {
        # ── ML features (must match FEATURES order) ──
        "URLLength":                url_len,
        "DomainLength":             domain_len,
        "IsDomainIP":               is_ip,
        "TLDLength":                len(tld),
        "NoOfSubDomain":            n_subdomains,
        "NoOfLettersInURL":         n_letters,
        "LetterRatioInURL":         round(n_letters / url_len, 4) if url_len else 0.0,
        "NoOfDegitsInURL":          n_digits,
        "DegitRatioInURL":          round(n_digits / url_len, 4) if url_len else 0.0,
        "NoOfEqualsInURL":          n_equals,
        "NoOfQMarkInURL":           n_qmark,
        "NoOfAmpersandInURL":       n_amp,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpecialCharRatioURL":      round(spec_ratio, 4),
        "IsHTTPS":                  is_https,
        "CharContinuationRate":     _char_continuation_rate(url),
        "URLCharProb":              _url_char_prob(url),
        # ── extras for rule-based logic ──
        "_url":      url,
        "_domain":   domain,
        "_tld":      tld,
        "_has_at":   "@" in url,
        "_has_ip":   is_ip == 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_flags(feats: dict) -> list[dict]:
    """Return a list of triggered rule dicts {rule, severity, label}."""
    flags = []
    url_lower = feats["_url"].lower()
    domain    = feats["_domain"].lower()
    tld       = feats.get("_tld", "").lower()

    def add(rule, sev, lbl):
        flags.append({"rule": rule, "severity": sev, "label": lbl})

    # ── HIGH severity ──
    if feats["_has_at"]:
        add("Contains '@' character in URL", "high", "danger")
    if feats["_has_ip"]:
        add("Domain is a raw IP address", "high", "danger")
    if feats["URLLength"] > URL_LONG_THRESHOLD:
        add(f"URL is very long ({feats['URLLength']} chars)", "high", "danger")

    # Brand impersonation
    for brand, legit_domain in BRAND_DOMAINS.items():
        if brand in domain and legit_domain not in domain:
            add(f"Possible brand impersonation: '{brand}'", "high", "danger")
            break

    # Punycode / IDN
    if "xn--" in domain:
        add("Punycode / internationalized domain (possible homograph)", "high", "danger")

    # Risky TLD
    if tld in RISKY_TLDS:
        add(f"High-risk TLD: '.{tld}'", "high", "danger")

    # URL shortener
    for shortener in URL_SHORTENERS:
        if shortener in domain:
            add(f"URL shortener detected: '{shortener}'", "high", "danger")
            break

    # ── MEDIUM severity ──
    kw_found = [k for k in SUSPICIOUS_KEYWORDS if k in url_lower]
    for kw in kw_found[:3]:  # cap at 3
        add(f"Suspicious keyword: '{kw}'", "medium", "warn")

    if not feats["IsHTTPS"]:
        add("No HTTPS (unencrypted connection)", "medium", "warn")
    if feats["NoOfSubDomain"] >= 3:
        add(f"Excessive subdomains ({feats['NoOfSubDomain']})", "medium", "warn")
    if domain.count("-") >= 3:
        add(f"Many hyphens in domain ({domain.count('-')})", "medium", "warn")
    if feats["NoOfDegitsInURL"] > 8:
        add(f"Many digits in URL ({feats['NoOfDegitsInURL']})", "medium", "warn")

    # ── LOW severity ──
    if feats["SpecialCharRatioURL"] > 0.12:
        add(f"High special-char ratio ({feats['SpecialCharRatioURL']:.2%})", "low", "warn")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_predict(url: str, model, scaler) -> dict:
    """Combine ML prediction + rule-based flags into a final verdict."""
    feats = extract_features(url)
    flags = rule_based_flags(feats)

    # ML probability — use all 17 features
    X = np.array([[feats[f] for f in FEATURES]])
    X_sc = scaler.transform(X)
    ml_proba = model.predict_proba(X_sc)[0]   # [P(legit), P(phishing)]
    ml_label = int(model.predict(X_sc)[0])     # 0 = legit, 1 = phishing
    ml_conf  = float(ml_proba[ml_label])

    # Rule contribution — uncapped, aggressive weights
    high_flags   = sum(1 for f in flags if f["severity"] == "high")
    medium_flags = sum(1 for f in flags if f["severity"] == "medium")
    low_flags    = sum(1 for f in flags if f["severity"] == "low")
    rule_score   = min(high_flags * 0.30 + medium_flags * 0.15 + low_flags * 0.05, 1.0)

    # Adaptive blending: more rule flags → more rule influence
    if rule_score >= 0.60:
        phish_prob_raw = 0.25 * ml_proba[1] + 0.75 * rule_score
    elif rule_score >= 0.30:
        phish_prob_raw = 0.40 * ml_proba[1] + 0.60 * rule_score
    elif rule_score > 0:
        phish_prob_raw = 0.55 * ml_proba[1] + 0.45 * rule_score
    else:
        phish_prob_raw = ml_proba[1]

    # Ensure minimum phishing probability when strong rules fire
    phish_prob = float(np.clip(max(phish_prob_raw, rule_score * 0.70), 0.0, 1.0))

    final_label = 1 if phish_prob >= 0.40 else 0
    confidence  = phish_prob if final_label == 1 else 1.0 - phish_prob

    return {
        "url":           url,
        "features":      feats,
        "flags":         flags,
        "ml_label":      ml_label,
        "ml_conf":       ml_conf,
        "phish_prob":    phish_prob,
        "final_label":   final_label,
        "confidence":    confidence,
        "is_phishing":   final_label == 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS (dynamic, per-analysis)
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
CLR_GRID = "#21262d"
CLR_TEXT = "#c9d1d9"
CLR_PHISH = "#f85149"
CLR_LEGIT = "#3fb950"
CLR_ACCENT = "#388bfd"
CLR_WARN = "#d29922"

def _fig_style(fig, ax_list=None):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list or [fig.axes[0]] if fig.axes else []):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=CLR_TEXT, labelsize=9)
        ax.xaxis.label.set_color(CLR_TEXT)
        ax.yaxis.label.set_color(CLR_TEXT)
        ax.title.set_color(CLR_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(CLR_GRID)
        ax.grid(color=CLR_GRID, linewidth=0.6, alpha=0.4)


def chart_probability_gauge(phish_prob: float):
    """Horizontal gauge: legitimate vs phishing probability."""
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
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Phishing Probability Breakdown", fontsize=11, fontweight="bold")
    leg = ax.legend(fontsize=8, facecolor=DARK_AX, edgecolor=CLR_GRID,
                    loc="upper right", ncol=2)
    for t in leg.get_texts(): t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return fig


def chart_feature_bars(feats: dict, df: pd.DataFrame):
    """Side-by-side bars: your URL features vs dataset average."""
    keys = ["URLLength", "DomainLength", "NoOfSubDomain",
            "NoOfLettersInURL", "NoOfDegitsInURL"]
    labels = ["URL Len", "Domain Len", "Subdomains", "Letters", "Digits"]
    user_vals = [feats[k] for k in keys]
    avg_vals  = [float(df[k].mean()) for k in keys]

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
    for t in leg.get_texts(): t.set_color(CLR_TEXT)
    _fig_style(fig)
    plt.tight_layout()
    return fig


def chart_risk_donut(flags: list):
    """Donut chart of risk severity breakdown."""
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
    return fig


def chart_history_trend(history: list):
    """Line chart of phishing probability over scan history."""
    if len(history) < 2:
        return None
    probs = [h["phish_prob"] for h in reversed(history)]
    labels_h = [h["url"][:18] + "…" if len(h["url"]) > 18 else h["url"]
                for h in reversed(history)]
    x = list(range(1, len(probs) + 1))

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [CLR_PHISH if p >= 0.40 else CLR_LEGIT for p in probs]
    ax.plot(x, probs, color=CLR_ACCENT, linewidth=2, marker="o",
            markersize=6, markerfacecolor=CLR_ACCENT, zorder=3)
    ax.scatter(x, probs, c=colors, s=60, zorder=4, edgecolors="white", linewidths=0.8)
    ax.axhline(y=0.40, color=CLR_WARN, linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(len(x), 0.42, "Threshold", ha="right", color=CLR_WARN, fontsize=8)
    ax.fill_between(x, 0.40, 1.0, alpha=0.06, color=CLR_PHISH)
    ax.fill_between(x, 0.0, 0.40, alpha=0.06, color=CLR_LEGIT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_h, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Phishing Probability")
    ax.set_title("Scan History Trend", fontsize=11, fontweight="bold")
    _fig_style(fig)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(icon: str, title: str):
    st.markdown(
        f'<div class="section-title">{icon}&nbsp;{title}</div>',
        unsafe_allow_html=True
    )


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def metric_box(val, lbl: str) -> str:
    return (
        f'<div class="metric-box">'
        f'<div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – model info
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(metrics: dict):
    with st.sidebar:
        st.markdown("## 🛡️ Model Info")
        st.markdown("---")
        acc = metrics.get("accuracy", 0)
        st.markdown(f"**Accuracy:** `{acc:.2%}`")
        st.markdown(f"**Train samples:** `{metrics.get('n_train', '—')}`")
        st.markdown(f"**Test samples:**  `{metrics.get('n_test', '—')}`")
        st.markdown("---")
        st.markdown("**Algorithm:** Random Forest (200 trees)")
        st.markdown("**Scaler:** StandardScaler")
        st.markdown("**Features:** 17")
        st.markdown("**Detection:** Hybrid (ML + Rules)")
        st.markdown("---")
        rep = metrics.get("report", {})
        if rep:
            st.markdown("**Per-class F1**")
            for cls, lbl in [("0", "Legitimate"), ("1", "Phishing")]:
                f1 = rep.get(cls, {}).get("f1-score", 0)
                st.markdown(f"• {lbl}: `{f1:.3f}`")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="text-align:center;">🛡️ Phishing Website Detector</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center;color:#8b949e;margin-top:-8px;">'
        'Hybrid ML + Rule-based detection with explainable results</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Load model ──────────────────────────────────────────────────────────
    with st.spinner("⚙️ Initialising model…"):
        try:
            model, scaler, train_metrics, df = load_and_train(DATASET_PATH)
            st.session_state.update(
                model=model, scaler=scaler,
                train_metrics=train_metrics, df=df
            )
        except FileNotFoundError:
            st.error(
                f"❌ Dataset file `{DATASET_PATH}` not found. "
                "Place the CSV in the same directory as `app.py`."
            )
            st.stop()
        except ValueError as exc:
            st.error(f"❌ {exc}")
            st.stop()

    render_sidebar(st.session_state.train_metrics)

    # ── Model accuracy banner ────────────────────────────────────────────────
    acc = st.session_state.train_metrics["accuracy"]
    lbl_cnt = st.session_state.train_metrics["label_counts"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_box(f"{acc:.1%}", "Model Accuracy"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_box(
            st.session_state.train_metrics["n_train"] +
            st.session_state.train_metrics["n_test"], "Total Samples"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_box(lbl_cnt.get(1, 0), "Phishing URLs"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_box(lbl_cnt.get(0, 0), "Legitimate URLs"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 – URL INPUT
    # ════════════════════════════════════════════════════════════════════════
    section("🔗", "URL Analysis")

    url_input = st.text_input(
        label="Enter URL to analyse",
        placeholder="e.g.  https://example.com  or  http://login-verify-bank.com/secure",
        key="url_input",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyse_clicked = st.button("🔍 Analyse", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🗑️ Clear History", use_container_width=False):
            st.session_state.history.clear()
            st.rerun()

    result = None
    if analyse_clicked:
        url_input = url_input.strip()
        if not url_input:
            st.warning("⚠️ Please enter a URL first.")
        else:
            with st.spinner("Analysing…"):
                result = hybrid_predict(
                    url_input,
                    st.session_state.model,
                    st.session_state.scaler,
                )
            # Push to history (most-recent first)
            st.session_state.history.appendleft({
                "url":         url_input[:60] + ("…" if len(url_input) > 60 else ""),
                "prediction":  "🚨 Phishing" if result["is_phishing"] else "✅ Legitimate",
                "confidence":  f"{result['confidence']:.1%}",
                "phish_prob":  result["phish_prob"],
            })

    # ════════════════════════════════════════════════════════════════════════
    # RESULT SECTIONS (only shown after analysis)
    # ════════════════════════════════════════════════════════════════════════
    if result:
        feats   = result["features"]
        flags   = result["flags"]
        is_phish = result["is_phishing"]

        # ── SECTION 2 – Extracted Features ──────────────────────────────────
        section("🔬", "Extracted Features")

        fc1, fc2, fc3, fc4, fc5, fc6 = st.columns(6)
        pairs = [
            (fc1, "URL Length",       feats["URLLength"],          "chars"),
            (fc2, "Domain Length",    feats["DomainLength"],        "chars"),
            (fc3, "Special-Char %",   f"{feats['SpecialCharRatioURL']:.2%}", ""),
            (fc4, "Subdomains",       feats["NoOfSubDomain"],       "count"),
            (fc5, "HTTPS",            "Yes" if feats["IsHTTPS"] else "No", ""),
            (fc6, "IP Domain",        "Yes" if feats.get("IsDomainIP") else "No", ""),
        ]
        for col, lbl, val, unit in pairs:
            with col:
                disp = f"{val} {unit}".strip()
                st.markdown(metric_box(disp, lbl), unsafe_allow_html=True)

        # Extra flags inline
        extras_html = ""
        if feats["_has_at"]:
            extras_html += badge("Contains @", "danger")
        if feats["_has_ip"]:
            extras_html += badge("IP-based domain", "danger")
        if not feats["IsHTTPS"]:
            extras_html += badge("No HTTPS", "warn")
        if extras_html:
            st.markdown(f"<div style='margin-top:10px;'>{extras_html}</div>",
                        unsafe_allow_html=True)

        # ── SECTION 3 – Prediction Result ───────────────────────────────────
        section("🎯", "Prediction Result")

        card_cls = "card-danger" if is_phish else "card-success"
        icon     = "🚨" if is_phish else "✅"
        verdict  = "PHISHING" if is_phish else "LEGITIMATE"
        vc       = CLR_PHISH if is_phish else CLR_LEGIT

        bar_color = "#f85149" if is_phish else "#3fb950"
        conf_pct  = int(result["confidence"] * 100)

        st.markdown(
            f'<div class="card {card_cls}">'
            f'<div style="font-size:2rem;font-weight:900;color:{vc};">'
            f'{icon} {verdict}</div>'
            f'<div style="color:#8b949e;margin:6px 0;">Confidence: '
            f'<strong style="color:{vc};">{result["confidence"]:.1%}</strong>'
            f' &nbsp;|&nbsp; Phishing probability: '
            f'<strong style="color:{vc};">{result["phish_prob"]:.1%}</strong>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Custom progress bar
        filled = conf_pct
        st.markdown(
            f'<div style="background:#21262d;border-radius:99px;height:14px;overflow:hidden;">'
            f'<div style="width:{filled}%;background:{bar_color};'
            f'height:100%;border-radius:99px;transition:width .4s;"></div></div>'
            f'<div style="text-align:right;color:#8b949e;font-size:0.8rem;'
            f'margin-top:4px;">{filled}% confidence</div>',
            unsafe_allow_html=True
        )

        # ── SECTION 4 – Explanation ──────────────────────────────────────────
        section("💡", "Why This Decision")

        if flags:
            for f in flags:
                sev_map = {"high": "danger", "medium": "warn", "low": "info"}
                icon_map = {"high": "🔴", "medium": "🟡", "low": "🔵"}
                kind = sev_map.get(f["severity"], "info")
                ico  = icon_map.get(f["severity"], "⚪")
                st.markdown(
                    f'<div class="card card-{kind}" style="padding:12px 16px;'
                    f'margin-bottom:8px;display:flex;align-items:center;gap:10px;">'
                    f'{ico} <span>{f["rule"]}</span>'
                    f'{badge(f["severity"].upper(), kind)}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="card card-success" style="padding:12px 16px;">'
                '✅ No rule-based risk factors detected. '
                'Decision based primarily on ML model.'
                '</div>',
                unsafe_allow_html=True
            )

        if not is_phish and not flags:
            st.info(
                "ℹ️ The ML model classified this URL as **legitimate** and no "
                "suspicious patterns were detected. Always exercise caution "
                "with unfamiliar sites."
            )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 – Dynamic Analysis Charts (shown after each analysis)
    # ════════════════════════════════════════════════════════════════════════
    if result:
        section("📊", "Live Analysis Charts")

        # Row 1: probability gauge (full width)
        st.pyplot(chart_probability_gauge(result["phish_prob"]),
                  use_container_width=True)

        # Row 2: feature comparison + risk donut
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            st.pyplot(chart_feature_bars(feats, st.session_state.df),
                      use_container_width=True)
        with ch2:
            st.pyplot(chart_risk_donut(flags), use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 – History + Trend
    # ════════════════════════════════════════════════════════════════════════
    section("📋", "Recent Analysis History")

    history = list(st.session_state.history)
    if not history:
        st.markdown(
            '<div class="card" style="color:#8b949e;text-align:center;">'
            'No URLs analysed yet. Enter a URL above to get started.</div>',
            unsafe_allow_html=True
        )
    else:
        hist_df = pd.DataFrame(history)[["url", "prediction", "confidence"]]
        hist_df.columns = ["URL", "Prediction", "Confidence"]
        st.table(hist_df)

        # Trend chart (shows after 2+ scans)
        trend_fig = chart_history_trend(history)
        if trend_fig:
            section("📈", "Scan Trend")
            st.pyplot(trend_fig, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#484f58;font-size:0.8rem;">'
        'Phishing Detector · Random Forest + Rule-based Hybrid · '
        'Built with Streamlit</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
