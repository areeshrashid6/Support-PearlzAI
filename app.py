import os
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
from openai import (
    OpenAI,
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY_URL = "https://platform.openai.com/api-keys"

# Keep the existing model IDs. The default is intentionally the
# oldest chat-capable GPT model in this list, as requested.
MODEL_OPTIONS = [
    ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
    ("GPT-4", "gpt-4"),
    ("GPT-4 Turbo", "gpt-4-turbo"),
    ("GPT-4o", "gpt-4o"),
    ("GPT-4o Mini", "gpt-4o-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("GPT-4.1 Mini", "gpt-4.1-mini"),
    ("GPT-5.6 Luna", "gpt-5.6-luna"),
    ("GPT-5.6 Terra", "gpt-5.6-terra"),
    ("GPT-5.6 Sol", "gpt-5.6-sol"),
]

MODEL_LABELS = [label for label, _ in MODEL_OPTIONS]
MODEL_IDS = {label: model_id for label, model_id in MODEL_OPTIONS}

# IMPORTANT: oldest model is the default, not the latest model.
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MODEL_LABEL = "GPT-3.5 Turbo"

st.set_page_config(
    page_title="Support Pearl AI",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "connected": False,
    "runtime_api_key": None,
    "runtime_model": DEFAULT_MODEL,
    "messages": [],
    "last_error": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================
def get_saved_api_key():
    """Read an optional server-side key without ever persisting a
    user-entered key. The UI still requires an explicit connection
    before the workspace is shown.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def model_label(model_id):
    for label, value in MODEL_OPTIONS:
        if value == model_id:
            return label
    return model_id


def save_run(data):
    """Persist only non-secret run information."""
    path = RUNS_DIR / f"{data['run_id']}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_corpus():
    path = CORPUS_DIR / "demo.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def search_corpus(query, max_results=5):
    """Small local keyword retrieval layer for the demo."""
    docs = load_corpus()
    words = set(re.findall(r"\w+", query.lower()))
    if not words:
        return []

    results = []
    for doc in docs:
        searchable = " ".join(
            [
                str(doc.get("title", "")),
                str(doc.get("content", "")),
                str(doc.get("company", "")),
            ]
        ).lower()

        score = sum(1 for word in words if word in searchable)
        if score:
            results.append(
                {
                    "source_ref": doc.get("id", "unknown"),
                    "title": doc.get("title", "Untitled source"),
                    "snippet": str(doc.get("content", ""))[:500],
                    "score": score,
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max_results]


def call_openai(client, model, messages):
    response = client.responses.create(
        model=model,
        input=messages,
        timeout=60,
    )

    text = getattr(response, "output_text", "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")

    usage = getattr(response, "usage", None)
    usage_data = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
    }
    return text, usage_data


def friendly_connection_error(exc):
    if isinstance(exc, AuthenticationError):
        return (
            "We couldn't connect to OpenAI. The API key appears to be "
            "invalid or unavailable for this account."
        )
    if isinstance(exc, RateLimitError):
        return (
            "OpenAI rejected the connection because of a usage, billing, "
            "or rate-limit restriction."
        )
    if isinstance(exc, APIConnectionError):
        return (
            "We couldn't reach OpenAI. Check your internet connection "
            "and try again."
        )
    if isinstance(exc, APIError):
        return "OpenAI returned an API error. Please try again or choose another model."
    return "We couldn't connect using the selected model. Please check the key and model."


def reset_connection():
    st.session_state.connected = False
    st.session_state.runtime_api_key = None
    st.session_state.runtime_model = DEFAULT_MODEL
    st.session_state.messages = []
    st.session_state.last_error = None


# ============================================================
# GLOBAL UI
# ============================================================
st.markdown(
    """
<style>
    html, body {
        background: linear-gradient(180deg, #f2f5fb 0%, #eef3ff 100%);
    }
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {
        max-width: 1280px;
        padding-top: 18px;
        padding-bottom: 60px;
    }

    .pearl-shell {
        position: relative;
        min-height: 100vh;
        overflow: hidden;
    }
    .pearl-shell::before,
    .pearl-shell::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        filter: blur(10px);
        opacity: 0.9;
        z-index: 0;
    }
    .pearl-shell::before {
        width: 440px;
        height: 440px;
        left: -130px;
        bottom: -150px;
        background: rgba(167, 155, 255, 0.24);
    }
    .pearl-shell::after {
        width: 520px;
        height: 520px;
        right: -90px;
        bottom: -200px;
        background: rgba(161, 220, 245, 0.22);
    }

    .pearl-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0 26px;
        position: relative;
        z-index: 1;
    }
    .pearl-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .pearl-mark {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7f7df6, #5ab6f4);
        box-shadow: inset 0 0 0 4px rgba(255,255,255,0.8), 0 8px 18px rgba(95, 108, 230, 0.2);
        position: relative;
        overflow: hidden;
    }
    .pearl-mark::before {
        content: "";
        position: absolute;
        inset: 8px 9px 8px 10px;
        border-radius: 50%;
        background: rgba(255,255,255,.9);
        opacity: .28;
    }
    .pearl-name {
        font-size: 18px;
        font-weight: 800;
        color: #171f2d;
        letter-spacing: -0.04em;
    }
    .pearl-tagline {
        color: #64748b;
        font-size: 12px;
        font-weight: 500;
    }
    .pearl-nav {
        display: flex;
        align-items: center;
        gap: 18px;
        color: #475467;
        font-size: 13px;
    }
    .pearl-powered {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #667085;
        font-size: 12px;
    }
    .pearl-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 14px;
        border-radius: 10px;
        border: 1px solid #dfe6ef;
        background: rgba(255,255,255,0.4);
        color: #475467;
        font-size: 12px;
        font-weight: 600;
        text-decoration: none;
    }
    .pearl-pill:hover {
        text-decoration: none;
    }
    .pearl-openai {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #fff 0%, #dce0ef 14%, #c7d0ef 35%, #b9c4f4 100%);
        border: 1px solid rgba(83,96,124,0.28);
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.8);
    }

    .pearl-hero {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: flex-start;
        gap: 54px;
        padding-top: 10px;
    }
    .pearl-copy {
        flex: 1.05;
        padding-top: 28px;
    }
    .pearl-kicker {
        color: #6a66d6;
        font-size: 11px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .pearl-title {
        max-width: 520px;
        font-size: clamp(48px, 4.2vw, 92px);
        line-height: 0.92;
        letter-spacing: -0.065em;
        font-weight: 800;
        color: #1b2333;
        margin: 0;
    }
    .pearl-title .highlight {
        display: block;
        background: linear-gradient(90deg, #4a53d6 0%, #7b5de7 38%, #2f9be9 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .pearl-subcopy {
        max-width: 590px;
        margin-top: 26px;
        font-size: 18px;
        line-height: 1.5;
        color: #667085;
    }
    .pearl-feature-list {
        margin-top: 26px;
        display: grid;
        gap: 18px;
    }
    .pearl-feature {
        display: flex;
        align-items: center;
        gap: 14px;
        color: #1f2937;
    }
    .pearl-icon {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 700;
        box-shadow: inset 0 0 0 1px rgba(122,132,148,0.15);
    }
    .pearl-icon.blue { background: rgba(120,132,255,0.12); color: #4f67ff; }
    .pearl-icon.green { background: rgba(69,205,153,0.12); color: #1f9d73; }
    .pearl-icon.purple { background: rgba(145,117,255,0.12); color: #7667df; }
    .pearl-feature strong {
        display: block;
        font-size: 18px;
        letter-spacing: -0.03em;
    }
    .pearl-feature span {
        display: block;
        color: #667085;
        font-size: 13px;
        line-height: 1.5;
    }
    .pearl-illustration {
        position: relative;
        margin-top: 40px;
        margin-left: 8px;
        width: 460px;
        height: 230px;
    }
    .pearl-device {
        position: absolute;
        left: 30px;
        bottom: 18px;
        width: 360px;
        height: 170px;
        background: rgba(255,255,255,0.55);
        border: 0.5px solid rgba(120,132,148,0.2);
        border-radius: 22px;
        backdrop-filter: blur(5px);
        box-shadow: 0 26px 50px rgba(136,146,162,0.18);
        transform: rotate(-12deg);
    }
    .pearl-device::before {
        content: "";
        position: absolute;
        inset: 18px 18px auto 18px;
        height: 10px;
        border-radius: 100px;
        background: linear-gradient(90deg, rgba(182,190,210,0.42), rgba(233,238,245,0.9));
    }
    .pearl-device::after {
        content: "";
        position: absolute;
        left: 20px;
        right: 20px;
        top: 42px;
        bottom: 20px;
        border-radius: 15px;
        background: linear-gradient(180deg, rgba(255,255,255,0.15), rgba(210,220,238,0.2));
    }
    .pearl-floating-tag {
        position: absolute;
        right: 45px;
        top: 25px;
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(178,188,204,0.6);
        color: #2a8a74;
        font-weight: 700;
        font-size: 13px;
        padding: 10px 14px;
        border-radius: 12px;
        transform: rotate(-10deg);
        box-shadow: 0 14px 28px rgba(92,106,140,0.08);
    }
    .pearl-swatch {
        position: absolute;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(129,123,255,0.45), rgba(183,206,255,0.28));
        right: -8px;
        bottom: -10px;
        box-shadow: 0 28px 50px rgba(136,146,162,0.15);
    }
    .pearl-swatch.alt {
        width: 90px;
        height: 90px;
        right: 94px;
        bottom: 52px;
        background: linear-gradient(135deg, rgba(118,201,255,0.38), rgba(230,181,255,0.22));
    }

    .pearl-form-wrap {
        flex: 0.92;
        background: rgba(255,255,255,0.48);
        border: 1px solid rgba(214,221,233,0.9);
        border-radius: 28px;
        box-shadow: 0 22px 42px rgba(70,86,115,0.08);
        padding: 22px 22px 18px;
        backdrop-filter: blur(2px);
    }
    .pearl-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(123, 107, 255, 0.12);
        border: 1px solid rgba(123,107,255,0.18);
        color: #4f46c8;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 18px;
    }
    .pearl-card-title {
        font-size: 36px;
        line-height: 1.15;
        letter-spacing: -0.05em;
        color: #1e2535;
        margin: 0 0 12px;
        font-weight: 800;
    }
    .pearl-card-sub {
        font-size: 13px;
        color: #667085;
        line-height: 1.5;
        margin-bottom: 18px;
    }
    .pearl-label {
        display: block;
        color: #2f3c4d;
        font-weight: 700;
        font-size: 12px;
        margin: 0 0 8px;
    }
    div[data-testid="stTextInput"] > div,
    div[data-testid="stSelectbox"] > div {
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        border: 1px solid #dfe7f0;
    }
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[role="button"] {
        font-size: 14px;
        color: #1f2937;
        min-height: 48px !important;
    }
    .pearl-field-note {
        font-size: 11px;
        color: #8995a3;
        margin-top: 8px;
        line-height: 1.45;
    }
    .pearl-security {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-top: 16px;
        color: #7b8393;
        font-size: 11px;
        line-height: 1.45;
    }
    .pearl-lock {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #dae0e8;
        background: rgba(245,247,250,0.6);
        color: #6c7587;
        font-size: 10px;
        flex-shrink: 0;
    }
    .pearl-primary {
        margin-top: 20px;
        width: 100% !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #262f52, #262d4d) !important;
        color: white !important;
        border: none !important;
        font-weight: 800 !important;
        min-height: 54px !important;
        box-shadow: none !important;
    }
    .pearl-primary:hover {
        filter: brightness(1.02);
    }
    .pearl-alt-row {
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid #edf0f4;
    }
    .pearl-or {
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 24px 0 18px;
        color: #8a94a3;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.2em;
    }
    .pearl-or::before,
    .pearl-or::after {
        content: "";
        height: 1px;
        background: #e7ebf0;
        flex: 1;
    }
    .pearl-key-box {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        background: rgba(243,246,252,0.9);
        border: 1px solid #dfe6ef;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 8px;
    }
    .pearl-key-box .left {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #1f2937;
        font-weight: 600;
    }
    .pearl-key-box .left .mini {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        background: rgba(91,100,255,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4d59d8;
        font-size: 14px;
    }
    .pearl-key-box .right {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 36px;
        padding: 0 14px;
        border-radius: 10px;
        border: 1px solid #dfe6ef;
        background: rgba(255,255,255,0.6);
        color: #1f2937;
        font-size: 12px;
        font-weight: 700;
        text-decoration: none;
    }

    .workspace-title {
        color:#172033;
        font-size:30px;
        font-weight:760;
        letter-spacing:-.035em;
        margin-bottom:4px;
    }
    .workspace-copy {
        color:#697586;
        font-size:13px;
        margin-bottom:22px;
    }
    .answer-card {
        background:#fff;
        border:1px solid #e3e7ed;
        border-radius:16px;
        padding:20px;
        box-shadow:0 8px 25px rgba(16,24,40,.04);
    }
    .answer-meta {
        color:#7a8494;
        font-size:11px;
        margin-bottom:10px;
    }
    .source-card {
        background:#f8fafc;
        border:1px solid #e8ecf1;
        border-radius:12px;
        padding:12px 14px;
        margin-top:8px;
    }
    .source-title {
        color:#344054;
        font-size:12px;
        font-weight:700;
        margin-bottom:3px;
    }
    .source-snippet {
        color:#7a8494;
        font-size:11px;
        line-height:1.5;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius:10px;
        border-color:#d1d7e0;
    }
    div[data-baseweb="select"] > div {
        border-radius:10px;
        min-height:46px;
        border-color:#d1d7e0;
    }
    .stButton > button,
    .stLinkButton > a {
        border-radius:10px !important;
        min-height:46px !important;
        font-weight:700 !important;
    }
    .stLinkButton > a {
        width:100%;
        justify-content:center;
        text-decoration:none !important;
    }
    .sidebar-status {
        padding:12px;
        border:1px solid #e3e7ed;
        border-radius:12px;
        background:#fff;
        color:#596579;
        font-size:11px;
        line-height:1.5;
    }

    @media(max-width: 1100px) {
        .pearl-hero { display:block; }
        .pearl-copy, .pearl-form-wrap { width:100%; }
        .pearl-illustration { transform: scale(0.82); transform-origin: center left; }
    }
    @media(max-width: 800px) {
        .block-container {padding:20px 18px 50px;}
        .pearl-topbar { display:block; }
        .pearl-brand { margin-bottom:10px; }
        .pearl-nav { justify-content: flex-start; flex-wrap: wrap; }
        .pearl-title {font-size:54px;}
        .pearl-form-wrap { margin-top: 28px; }
        .pearl-card-title {font-size:30px;}
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OPTIONAL SERVER KEY: DO NOT AUTO-SHOW WORKSPACE
# ============================================================
# We deliberately require the connection step. A server key may be
# available for deployment, but it does not bypass the first screen.
if not st.session_state.connected:
    # --------------------------------------------------------
    # PAGE 1 — OPENAI CONNECTION
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="pearl-shell">
            <div class="pearl-topbar">
                <div class="pearl-brand">
                    <div class="pearl-mark"></div>
                    <div>
                        <div class="pearl-name">Support Pearl AI</div>
                        <div class="pearl-tagline">Your Knowledge. Always On.</div>
                    </div>
                </div>
                <div class="pearl-nav">
                    <div class="pearl-powered"><span class="pearl-openai"></span> Powered by OpenAI</div>
                    <a class="pearl-pill" href="https://platform.openai.com/" target="_blank">Need help?</a>
                    <a class="pearl-pill" href="https://platform.openai.com/" target="_blank">Learn more</a>
                </div>
            </div>

            <div class="pearl-hero">
                <div class="pearl-copy">
                    <div class="pearl-kicker">AI-powered customer support</div>
                    <h1 class="pearl-title">Turn customer<br>questions into<br><span class="highlight">trusted answers.</span></h1>
                    <div class="pearl-subcopy">
                        Connect your own OpenAI account to start getting accurate,
                        document-based answers from Support Pearl AI.
                    </div>

                    <div class="pearl-feature-list">
                        <div class="pearl-feature">
                            <div class="pearl-icon blue">🔒</div>
                            <div>
                                <strong>Your API key, your control</strong>
                                <span>We never store your API key in the project files or run logs.</span>
                            </div>
                        </div>
                        <div class="pearl-feature">
                            <div class="pearl-icon green">⚡</div>
                            <div>
                                <strong>Choose the right model</strong>
                                <span>Select from a range of OpenAI models for your needs.</span>
                            </div>
                        </div>
                        <div class="pearl-feature">
                            <div class="pearl-icon purple">💬</div>
                            <div>
                                <strong>Start asking, get answers</strong>
                                <span>Once connected, you can ask support questions and get accurate answers.</span>
                            </div>
                        </div>
                    </div>

                    <div class="pearl-illustration">
                        <div class="pearl-device"></div>
                        <div class="pearl-floating-tag">Smarter support<br>starts here.</div>
                        <div class="pearl-swatch"></div>
                        <div class="pearl-swatch alt"></div>
                    </div>
                </div>

                <div class="pearl-form-wrap">
                    <div class="pearl-step">Step 1 of 2</div>
                    <h2 class="pearl-card-title">Connect your OpenAI account</h2>
                    <div class="pearl-card-sub">Add your OpenAI API key and choose the model you want to use. We'll verify the connection before you continue.</div>

                    <label class="pearl-label">API Key</label>
                    <div>
                        """,
        unsafe_allow_html=True,
    )

    entered_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-••••••••••••••••••••",
        label_visibility="collapsed",
    )

    st.markdown(
        """
                    </div>
                    <div class="pearl-field-note">Your key is used only to authenticate your OpenAI requests.</div>

                    <div style="margin-top: 18px;">
                        <label class="pearl-label">Select GPT model</label>
                        """,
        unsafe_allow_html=True,
    )

    selected_model_label = st.selectbox(
        "AI model",
        MODEL_LABELS,
        index=MODEL_LABELS.index(DEFAULT_MODEL_LABEL),
        help="The default is intentionally set to the oldest model in the current list.",
        label_visibility="collapsed",
    )
    selected_model = MODEL_IDS[selected_model_label]

    descriptions = {
        "GPT-3.5 Turbo": "Fast, affordable, and great for general tasks.",
        "GPT-4": "Older high-intelligence GPT model.",
        "GPT-4 Turbo": "Older fast GPT model.",
        "GPT-4o": "General-purpose multimodal GPT model.",
        "GPT-4o Mini": "Smaller, faster GPT model.",
        "GPT-4.1": "Strong non-reasoning GPT model.",
        "GPT-4.1 Mini": "Smaller, faster GPT-4.1 variant.",
        "GPT-5.6 Luna": "Cost-sensitive GPT-5.6 model.",
        "GPT-5.6 Terra": "Balances intelligence and cost.",
        "GPT-5.6 Sol": "Flagship GPT-5.6 model.",
    }
    st.markdown(
        f'<div class="pearl-field-note">{descriptions.get(selected_model_label, "")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Connect & Continue →", type="primary", use_container_width=True, key="pearl_connect"):
        st.session_state.last_error = None

        if not entered_key.strip():
            st.session_state.last_error = "Please enter your OpenAI API key."
        else:
            try:
                test_client = OpenAI(
                    api_key=entered_key.strip(),
                    timeout=60,
                )
                with st.spinner("Verifying your OpenAI connection..."):
                    response = test_client.responses.create(
                        model=selected_model,
                        input="Reply with exactly: CONNECTED",
                        timeout=60,
                    )

                text = getattr(response, "output_text", "").strip()
                if not text:
                    raise RuntimeError("OpenAI returned an empty response.")

                st.session_state.runtime_api_key = entered_key.strip()
                st.session_state.runtime_model = selected_model
                st.session_state.connected = True
                st.session_state.messages = []
                st.session_state.last_error = None
                st.rerun()

            except Exception as exc:
                st.session_state.last_error = friendly_connection_error(exc)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    st.markdown(
        """
                    <div class="pearl-security">
                        <div class="pearl-lock">✓</div>
                        <div>Your API key is kept in session memory and is never written to the project files or run logs.</div>
                    </div>
                    <div class="pearl-or">or</div>
                    <div class="pearl-key-box">
                        <div class="left">
                            <div class="mini">⌁</div>
                            <span>Don't have an OpenAI API key?</span>
                        </div>
                        <a class="right" href="https://platform.openai.com/api-keys" target="_blank">Get an OpenAI API key ↗</a>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# PAGE 2 — QUESTION / RESPONSE WORKSPACE
# ============================================================
runtime_key = st.session_state.runtime_api_key
runtime_model = st.session_state.runtime_model

if not runtime_key:
    reset_connection()
    st.rerun()

client = OpenAI(api_key=runtime_key, timeout=60)

with st.sidebar:
    st.markdown(
        """
        <div class="mm-brand" style="margin-bottom:20px;">
            <div class="mm-mark">M</div>
            <div class="mm-name">MarketMind AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-status">
            <strong style="color:#344054;">Connected</strong><br>
            Model: {model_label(runtime_model)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Disconnect", use_container_width=True):
        reset_connection()
        st.rerun()

st.markdown('<div class="workspace-title">Ask MarketMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="workspace-copy">Ask a question and receive a response from your selected OpenAI model.</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Your question",
    placeholder="Example: What are the key opportunities and risks for an AI customer-support company?",
    height=130,
)

if st.button("Ask MarketMind →", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        user_text = question.strip()

        retrieved = search_corpus(user_text, max_results=5)
        context = ""
        if retrieved:
            context = "\n\n".join(
                f"Source: {item['title']}\n{item['snippet']}"
                for item in retrieved
            )

        system_prompt = (
            "You are MarketMind AI, a professional business research assistant. "
            "Answer clearly and directly. If local demo sources are supplied, "
            "use them as supporting context, but do not pretend they are exhaustive. "
            "Do not invent citations or claim research was performed when it was not."
        )

        if context:
            system_prompt += (
                "\n\nRelevant local demo sources:\n" + context
            )

        # Keep the visible conversation, but send only the current request plus
        # recent history to avoid uncontrolled prompt growth.
        history = []
        for item in st.session_state.messages[-8:]:
            history.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        try:
            with st.spinner("MarketMind is thinking..."):
                answer, usage = call_openai(
                    client,
                    runtime_model,
                    messages,
                )

            st.session_state.messages.append(
                {"role": "user", "content": user_text}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            save_run(
                {
                    "run_id": str(uuid.uuid4()),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": runtime_model,
                    "question": user_text,
                    "answer": answer,
                    "sources": retrieved,
                    "usage": usage,
                    # Deliberately no API key.
                }
            )

        except AuthenticationError:
            st.error("Your OpenAI API key is no longer valid. Please reconnect.")
        except RateLimitError:
            st.error("OpenAI returned a rate-limit or billing error. Please check your account.")
        except APIConnectionError:
            st.error("Could not connect to OpenAI. Please try again.")
        except APIError as exc:
            st.error(f"OpenAI API error: {exc}")
        except Exception as exc:
            st.error(f"Something went wrong: {type(exc).__name__}: {exc}")

# Render chat history.
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if st.session_state.messages:
    last_sources = search_corpus(
        st.session_state.messages[-2]["content"]
        if len(st.session_state.messages) >= 2
        else "",
        max_results=5,
    )
    if last_sources:
        st.markdown("#### Sources used for the latest question")
        for item in last_sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{item['title']}</div>
                    <div class="source-snippet">{item['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
