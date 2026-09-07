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
    page_title="SupportPearlz AI",
    page_icon="M",
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
# GLOBAL UI — SUPPORT PEARLZ AI REDESIGN
# ============================================================
st.markdown(
    """
<style>
    #MainMenu, footer {visibility:hidden;}
    header[data-testid="stHeader"] {background:transparent;}
    .block-container {
        max-width: 100%;
        padding: 0 !important;
    }
    .stApp {
        background: #f8faff;
    }

    /* Hide Streamlit chrome that competes with the product UI */
    [data-testid="stDecoration"] {display:none;}
    [data-testid="stToolbar"] {display:none;}

    .sp-page {
        min-height: calc(100vh - 8px);
        display:grid;
        grid-template-columns: 47% 53%;
        overflow:hidden;
        background:
            radial-gradient(circle at 8% 84%, rgba(117,154,255,.12), transparent 26%),
            radial-gradient(circle at 39% 72%, rgba(154,124,255,.10), transparent 25%),
            #f8faff;
    }

    .sp-left {
        position:relative;
        min-height:100vh;
        padding:34px 58px 42px;
        overflow:hidden;
        border-right:1px solid #edf0f6;
    }

    .sp-left:before {
        content:"";
        position:absolute;
        width:480px;
        height:480px;
        left:-240px;
        bottom:-260px;
        border-radius:50%;
        background:rgba(104,151,255,.10);
        filter:blur(2px);
    }

    .sp-left:after {
        content:"";
        position:absolute;
        width:360px;
        height:360px;
        right:-210px;
        bottom:-160px;
        border-radius:50%;
        background:rgba(171,137,255,.10);
    }

    .sp-brand {
        position:relative;
        z-index:2;
        display:flex;
        align-items:center;
        gap:11px;
        margin-bottom:72px;
    }

    .sp-logo {
        width:42px;
        height:42px;
        border-radius:13px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#fff;
        background:linear-gradient(145deg,#15213a,#243556);
        font-size:17px;
        font-weight:800;
        box-shadow:0 8px 18px rgba(21,33,58,.15);
    }

    .sp-brand-name {
        color:#111b32;
        font-size:18px;
        line-height:1;
        font-weight:780;
        letter-spacing:-.03em;
    }

    .sp-brand-name span {
        color:#5c72ff;
    }

    .sp-powered {
        margin-left:auto;
        display:flex;
        align-items:center;
        gap:8px;
        padding:10px 15px;
        border:1px solid #e4e8f0;
        background:rgba(255,255,255,.72);
        border-radius:999px;
        color:#526079;
        font-size:11px;
        font-weight:650;
    }

    .sp-openai {
        font-size:15px;
        color:#17233b;
        font-weight:750;
    }

    .sp-hero {
        position:relative;
        z-index:2;
        max-width:610px;
    }

    .sp-eyebrow {
        color:#6673ff;
        font-size:11px;
        font-weight:800;
        letter-spacing:.16em;
        text-transform:uppercase;
        margin-bottom:18px;
    }

    .sp-title {
        color:#111b32;
        font-size:51px;
        line-height:1.05;
        letter-spacing:-.055em;
        font-weight:790;
        max-width:590px;
        margin-bottom:22px;
    }

    .sp-title .gradient {
        background:linear-gradient(90deg,#348eff,#695cff);
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
    }

    .sp-copy {
        color:#5f6e88;
        font-size:16px;
        line-height:1.62;
        max-width:535px;
        margin-bottom:36px;
    }

    .sp-feature {
        display:flex;
        align-items:flex-start;
        gap:15px;
        margin-bottom:24px;
    }

    .sp-feature-icon {
        width:42px;
        height:42px;
        flex:0 0 42px;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:18px;
        border:1px solid rgba(255,255,255,.7);
    }

    .sp-feature-icon.blue {background:#eaf3ff;color:#3d8eff;}
    .sp-feature-icon.green {background:#e8faf3;color:#20b983;}
    .sp-feature-icon.purple {background:#f0edff;color:#695cff;}

    .sp-feature-title {
        color:#16213a;
        font-size:14px;
        font-weight:760;
        margin:1px 0 4px;
    }

    .sp-feature-copy {
        color:#65738c;
        font-size:12px;
        line-height:1.55;
        max-width:390px;
    }

    .sp-art {
        position:absolute;
        z-index:1;
        left:22px;
        bottom:-30px;
        width:560px;
        height:230px;
        pointer-events:none;
    }

    .sp-art-circle {
        position:absolute;
        left:0;
        bottom:5px;
        width:190px;
        height:190px;
        border-radius:50%;
        background:linear-gradient(145deg,#c8dcff,#dfe8ff);
    }

    .sp-art-card {
        position:absolute;
        left:72px;
        bottom:22px;
        width:420px;
        height:154px;
        transform:rotate(-4deg);
        border:1px solid rgba(255,255,255,.9);
        border-radius:20px;
        background:rgba(255,255,255,.82);
        box-shadow:0 18px 38px rgba(70,91,143,.13);
        padding:25px;
        backdrop-filter:blur(10px);
    }

    .sp-art-mini {
        width:44px;
        height:44px;
        border-radius:12px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#f3f5fa;
        color:#18233a;
        font-weight:800;
        font-size:17px;
        margin-bottom:13px;
    }

    .sp-line {
        height:9px;
        border-radius:99px;
        background:#d9dfeb;
        margin-bottom:9px;
    }
    .sp-line.one {width:165px;}
    .sp-line.two {width:105px;}
    .sp-line.three {width:300px;margin-top:16px;}
    .sp-line.four {width:230px;}

    .sp-art-tag {
        position:absolute;
        right:-18px;
        top:18px;
        width:145px;
        padding:16px;
        border:1px solid #e5e8f3;
        border-radius:15px;
        background:rgba(255,255,255,.9);
        color:#6257ff;
        font-size:12px;
        line-height:1.35;
        font-weight:700;
        transform:rotate(5deg);
        box-shadow:0 12px 25px rgba(70,91,143,.08);
    }

    .sp-right {
        min-height:100vh;
        padding:34px 58px 38px;
        display:flex;
        flex-direction:column;
        justify-content:center;
    }

    .sp-topbar {
        position:absolute;
        top:35px;
        right:58px;
        display:flex;
        align-items:center;
        gap:10px;
        color:#75829a;
        font-size:12px;
    }

    .sp-topbar a {
        color:#18243d;
        text-decoration:none;
        font-weight:700;
        border:1px solid #dfe5ee;
        background:#fff;
        padding:11px 18px;
        border-radius:12px;
    }

    .sp-form-card {
        width:min(100%, 690px);
        margin:42px auto 0;
        padding:35px 38px 32px;
        border:1px solid #e4e8f0;
        border-radius:24px;
        background:rgba(255,255,255,.94);
        box-shadow:0 20px 60px rgba(35,53,91,.08);
    }

    .sp-step {
        display:inline-flex;
        align-items:center;
        padding:8px 13px;
        border-radius:999px;
        background:#f0f2ff;
        color:#5d62f5;
        font-size:10px;
        font-weight:800;
        letter-spacing:.13em;
        margin-bottom:17px;
    }

    .sp-form-title {
        color:#111b32;
        font-size:31px;
        line-height:1.1;
        letter-spacing:-.04em;
        font-weight:790;
        margin-bottom:9px;
    }

    .sp-form-copy {
        color:#687793;
        font-size:14px;
        line-height:1.55;
        margin-bottom:27px;
    }

    .sp-label {
        color:#18233a;
        font-size:13px;
        font-weight:750;
        margin:0 0 8px;
    }

    .sp-help {
        color:#75839b;
        font-size:11px;
        line-height:1.5;
        margin:7px 0 22px;
    }

    .sp-security {
        display:flex;
        align-items:center;
        gap:9px;
        padding:12px 13px;
        border-radius:12px;
        background:#f7f9fc;
        border:1px solid #edf0f5;
        color:#758198;
        font-size:11px;
        line-height:1.4;
        margin-top:16px;
    }

    .sp-security-icon {
        width:22px;
        height:22px;
        flex:0 0 22px;
        display:flex;
        align-items:center;
        justify-content:center;
        border:1px solid #dce3ed;
        border-radius:7px;
        background:#fff;
    }

    .sp-or {
        display:flex;
        align-items:center;
        gap:12px;
        color:#7d899e;
        font-size:11px;
        font-weight:750;
        margin:22px 0;
    }

    .sp-or:before,.sp-or:after {
        content:"";
        height:1px;
        flex:1;
        background:#e7ebf1;
    }

    .sp-key-row {
        display:flex;
        align-items:center;
        gap:14px;
        padding:13px 14px;
        border:1px solid #e5e9f2;
        border-radius:14px;
        background:#f8f9ff;
    }

    .sp-key-icon {
        width:40px;
        height:40px;
        border-radius:11px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#eeecff;
        color:#635bff;
        font-size:18px;
    }

    .sp-key-copy {
        flex:1;
    }

    .sp-key-title {
        color:#18233a;
        font-size:12px;
        font-weight:750;
        margin-bottom:3px;
    }

    .sp-key-sub {
        color:#71809a;
        font-size:10px;
    }

    /* Streamlit controls restyled to match the mockup */
    div[data-testid="stTextInput"] input {
        height:54px;
        border-radius:12px !important;
        border:1px solid #dce2eb !important;
        background:#fff !important;
        color:#17233b !important;
        font-size:13px !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color:#7480ff !important;
        box-shadow:0 0 0 3px rgba(104,110,255,.10) !important;
    }

    div[data-baseweb="select"] > div {
        min-height:58px !important;
        border-radius:13px !important;
        border:1px solid #dce2eb !important;
        background:#fff !important;
    }

    div[data-baseweb="select"] * {
        color:#18233a;
    }

    .stButton > button {
        min-height:56px !important;
        border-radius:13px !important;
        border:0 !important;
        background:linear-gradient(100deg,#18253e,#253354) !important;
        color:#fff !important;
        font-size:14px !important;
        font-weight:760 !important;
        box-shadow:0 9px 20px rgba(27,42,72,.15);
    }

    .stButton > button:hover {
        border:0 !important;
        filter:brightness(1.04);
    }

    .stLinkButton > a {
        min-height:44px !important;
        border-radius:11px !important;
        border:1px solid #dbe2ec !important;
        background:#fff !important;
        color:#1a2740 !important;
        font-size:11px !important;
        font-weight:720 !important;
        text-decoration:none !important;
    }

    div[data-testid="stAlert"] {
        border-radius:11px;
    }

    /* Workspace */
    .workspace-shell {
        max-width:900px;
        margin:0 auto;
        padding-top:35px;
    }

    .workspace-title {
        color:#111b32;
        font-size:31px;
        font-weight:790;
        letter-spacing:-.04em;
    }

    .workspace-copy {
        color:#697793;
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

    .answer-meta {color:#7a8494;font-size:11px;margin-bottom:10px;}
    .source-card {
        background:#f8fafc;
        border:1px solid #e8ecf1;
        border-radius:12px;
        padding:12px 14px;
        margin-top:8px;
    }
    .source-title {color:#344054;font-size:12px;font-weight:700;margin-bottom:3px;}
    .source-snippet {color:#7a8494;font-size:11px;line-height:1.5;}

    @media(max-width: 950px) {
        .sp-page {display:block;overflow:visible;}
        .sp-left {
            min-height:auto;
            padding:28px 24px 34px;
            border-right:0;
        }
        .sp-brand {margin-bottom:42px;}
        .sp-powered {display:none;}
        .sp-title {font-size:39px;}
        .sp-art {display:none;}
        .sp-right {
            min-height:auto;
            padding:20px 20px 45px;
        }
        .sp-topbar {position:static;justify-content:flex-end;margin-bottom:10px;}
        .sp-form-card {margin:0 auto;}
    }

    @media(max-width: 560px) {
        .sp-left {padding:24px 18px 28px;}
        .sp-right {padding:12px 12px 35px;}
        .sp-form-card {padding:25px 20px 23px;border-radius:18px;}
        .sp-form-title {font-size:26px;}
        .sp-title {font-size:34px;}
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
    # PAGE 1 — OPENAI CONNECTION (UI REDESIGN ONLY)
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="sp-page">
            <section class="sp-left">
                <div class="sp-brand">
                    <div class="sp-logo">S</div>
                    <div class="sp-brand-name">Support Pearlz <span>AI</span></div>
                    <div class="sp-powered">◉ <span>Powered by</span> <span class="sp-openai">OpenAI</span></div>
                </div>

                <div class="sp-hero">
                    <div class="sp-eyebrow">AI-POWERED CUSTOMER SUPPORT</div>
                    <div class="sp-title">
                        Turn customer questions into
                        <span class="gradient">trusted answers.</span>
                    </div>
                    <div class="sp-copy">
                        Connect your own OpenAI account to start getting accurate,
                        document-based answers from SupportPearlz AI.
                    </div>

                    <div class="sp-feature">
                        <div class="sp-feature-icon blue">♙</div>
                        <div>
                            <div class="sp-feature-title">Your API key, your control</div>
                            <div class="sp-feature-copy">
                                We never store your API key. It stays only in your browser session.
                            </div>
                        </div>
                    </div>

                    <div class="sp-feature">
                        <div class="sp-feature-icon green">ϟ</div>
                        <div>
                            <div class="sp-feature-title">Choose the right model</div>
                            <div class="sp-feature-copy">
                                Pick from a range of OpenAI models for your support needs.
                            </div>
                        </div>
                    </div>

                    <div class="sp-feature">
                        <div class="sp-feature-icon purple">▣</div>
                        <div>
                            <div class="sp-feature-title">Start asking, get answers</div>
                            <div class="sp-feature-copy">
                                Once connected, ask customer questions and get accurate,
                                helpful responses.
                            </div>
                        </div>
                    </div>
                </div>

                <div class="sp-art">
                    <div class="sp-art-circle"></div>
                    <div class="sp-art-card">
                        <div class="sp-art-mini">◎</div>
                        <div class="sp-line one"></div>
                        <div class="sp-line two"></div>
                        <div class="sp-line three"></div>
                        <div class="sp-line four"></div>
                        <div class="sp-art-tag">Smarter support<br>starts here.</div>
                    </div>
                </div>
            </section>

            <section class="sp-right">
                <div class="sp-topbar">
                    <span>Need help?</span>
                    <a href="#" onclick="return false;">Learn more</a>
                </div>

                <div class="sp-form-card">
                    <div class="sp-step">STEP 1 OF 2</div>
                    <div class="sp-form-title">Connect your OpenAI account</div>
                    <div class="sp-form-copy">
                        Add your OpenAI API key and choose the model you want to use.
                        We'll verify the connection before you continue.
                    </div>
        """,
        unsafe_allow_html=True,
    )

    entered_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-••••••••••••••••••••••••",
        label_visibility="visible",
    )

    st.markdown(
        '<div class="sp-help">Your key is used only to authenticate your requests and is not stored on our servers.</div>',
        unsafe_allow_html=True,
    )

    selected_model_label = st.selectbox(
        "Select GPT model",
        MODEL_LABELS,
        index=MODEL_LABELS.index(DEFAULT_MODEL_LABEL),
        help="The default is intentionally set to the oldest model in the current list.",
    )
    selected_model = MODEL_IDS[selected_model_label]

    descriptions = {
        "GPT-3.5 Turbo": "This is the oldest available model in the list. You can change it later.",
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
        f'<div class="sp-help">{descriptions.get(selected_model_label, "")}</div>',
        unsafe_allow_html=True,
    )

    if st.button("Connect & Continue   →", type="primary", use_container_width=True):
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
                    <div class="sp-security">
                        <div class="sp-security-icon">✓</div>
                        <div>Your API key is kept in session memory and is never written to project files or run logs.</div>
                    </div>

                    <div class="sp-or"><span>OR</span></div>

                    <div class="sp-key-row">
                        <div class="sp-key-icon">↗</div>
                        <div class="sp-key-copy">
                            <div class="sp-key-title">Don't have an OpenAI API key?</div>
                            <div class="sp-key-sub">Create your own key on the official OpenAI platform.</div>
                        </div>
                    </div>
        """,
        unsafe_allow_html=True,
    )

    st.link_button(
        "Get an OpenAI API Key ↗",
        OPENAI_API_KEY_URL,
        use_container_width=True,
    )

    st.markdown(
        """
                </div>
            </section>
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
            <div class="mm-name">SupportPearlz AI</div>
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

st.markdown('<div class="workspace-title">Ask SupportPearlz AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="workspace-copy">Ask a question and receive a response from your selected OpenAI model.</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Your question",
    placeholder="Example: What are the key opportunities and risks for an AI customer-support company?",
    height=130,
)

if st.button("Ask SupportPearlz AI →", type="primary", use_container_width=True):
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
            "You are SupportPearlz AI, a professional business research assistant. "
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
            with st.spinner("SupportPearlz AI is thinking..."):
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
