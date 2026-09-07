Yesterday 11:27 PM
SupportPearlz_LangChain_RAG_Assignment_Manual.pdf
PDF
give all files these in zip. first page take api ask model and connect. if dont api key user give then give link to redirect user to buy key rom openai latform. after connection, user ask question and get response  .[most important show get api key button in last bottom of the screen frst show enter api field and then gpt model selection. one more thig iin got model selectionlatest model is selected by default so set the oldest model by default]  

Done. I created the complete ZIP with the requested flow.

Included flow

First screen

OpenAI API key field
GPT model selection
GPT-3.5 Turbo selected by default — the oldest model in the existing list
Connect & Continue
Security message
Get an OpenAI API Key button at the very bottom, linking to OpenAI's API-key page
Real API connection verification before entering the workspace

After connection

Ask a question
Send it to the selected OpenAI model
Display the response
Conversation history
Local demo-source retrieval
Run JSON persistence without saving the API key
Disconnect option

The model list from your existing MarketMind code is preserved, with only the default changed from GPT-5.6 Luna to GPT-3.5 Turbo. Your existing code showed that model list and connection workflow.

OpenAI currently lists GPT-3.5 Turbo as a legacy model that is still available through the API, so I've used it as the requested oldest default.

ZIP

Download MarketMind_OpenAI_Connection_QA.zip

It contains:

MarketMind_OpenAI_Connection_QA/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── corpus/
│   └── demo.json
├── runs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── README.md
└── REFERENCE_Script_Listing_Flow.txt

The OpenAI API-key destination used in the app is the official API keys page. OpenAI API Keys

redesign this i have already designed tpage same like this
Edit
chane name in above Support pearlz ai
Edit
Pasted text(10).txt
Document
design this in code change but keep working same only vhange ui

Done. I redesigned the existing page in code to match the screenshot style and changed the branding to Support Pearlz AI.

Changed UI
Support Pearlz AI branding
Split-screen layout
Left-side AI support hero section
Gradient headline
Feature blocks with icons
OpenAI “Powered by” badge
Large rounded connection card
API key field
GPT model selector
Connect & Continue button
Security notice
OR divider
Get API Key section
Responsive mobile layout
Kept functionality unchanged
OpenAI API key connection
Connection verification
Model selection
GPT-3.5 Turbo remains the default
API errors shown on Connect
Question → OpenAI response flow
Corpus retrieval
Session handling
Disconnect
Run logging without storing API keys
Download

Download Support Pearlz AI redesigned ZIP

It contains only the 4 main files:
app.py, requirements.txt, README.md, corpus/demo.json.

give in app.py file

Sure — here is the redesigned app.py file:

Download app.py

app.py
Code

Yes — I found the problem in the previous app.py.

The HTML was being placed around Streamlit widgets incorrectly, so Streamlit was displaying the HTML tags as text instead of rendering the design.

I fixed app.py by:

Removing the problematic HTML wrapper around Streamlit inputs/buttons
Using Streamlit-safe columns for the two-panel layout
Keeping the new Support Pearlz AI UI
Keeping the OpenAI connection functionality exactly the same
Keeping GPT-3.5 Turbo as the default
Keeping API-key validation and question/answer functionality
Fixed app.py

Download the corrected app.py

You can replace your current GitHub app.py with this file.

app.py
Code
 code whoing not frontend

Yes — the issue was the HTML indentation, which made Streamlit interpret it as a code block.

I fixed the actual app.py and verified that it passes the Python syntax check.

Fixed file

Download fixed app.py

Also fixed ZIP

Download fixed 4-file ZIP

Now the page should render the Support Pearlz AI frontend, not show <div class="sp-hero">... as text. 

app.py
Code

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

============================================================
CONFIG
============================================================

BASE_DIR = Path(file).parent
CORPUS_DIR = BASE_DIR / "corpus"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY_URL = "https://platform.openai.com/api-keys"

Keep the existing model IDs. The default is intentionally the
oldest chat-capable GPT model in this list, as requested.

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

IMPORTANT: oldest model is the default, not the latest model.

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MODEL_LABEL = "GPT-3.5 Turbo"

st.set_page_config(
page_title="MarketMind AI",
page_icon="M",
layout="wide",
initial_sidebar_state="expanded",
)

============================================================
SESSION STATE
============================================================

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

============================================================
HELPERS
============================================================

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

============================================================
GLOBAL UI
============================================================

st.markdown(
"""

<style> #MainMenu, footer {visibility: hidden;} header[data-testid="stHeader"] {background: transparent;} .block-container { max-width: 1180px; padding-top: 34px; padding-bottom: 60px; } .mm-brand { display:flex; align-items:center; gap:10px; margin-bottom:34px; } .mm-mark { width:34px; height:34px; border-radius:10px; background:#172033; color:#fff; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:800; letter-spacing:-.02em; } .mm-name { font-size:16px; font-weight:750; color:#172033; letter-spacing:-.02em; } .mm-pill { margin-left:auto; display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border:1px solid #e2e7ee; border-radius:999px; color:#667085; font-size:11px; background:#fff; } .mm-dot { width:6px; height:6px; border-radius:50%; background:#20a464; } .setup-eyebrow { color:#687385; font-size:11px; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin-bottom:10px; } .setup-title { color:#172033; font-size:40px; line-height:1.08; letter-spacing:-.04em; font-weight:760; margin-bottom:12px; } .setup-copy { color:#667085; font-size:14px; line-height:1.65; max-width:590px; margin-bottom:26px; } .setup-card { background:#fff; border:1px solid #e3e7ed; border-radius:18px; padding:24px; box-shadow:0 12px 34px rgba(16,24,40,.06); } .card-kicker { color:#172033; font-size:15px; font-weight:750; margin-bottom:3px; } .card-sub { color:#7a8494; font-size:12px; line-height:1.5; margin-bottom:20px; } .field-note { color:#8a94a3; font-size:11px; line-height:1.45; margin-top:5px; } .security-row { display:flex; align-items:flex-start; gap:8px; color:#7a8494; font-size:11px; line-height:1.45; margin-top:13px; } .security-icon { width:18px; height:18px; border:1px solid #dce2e9; border-radius:6px; display:flex; align-items:center; justify-content:center; flex:0 0 auto; font-size:9px; color:#566274; } .get-key-wrap { margin-top:22px; padding-top:18px; border-top:1px solid #edf0f4; } .get-key-label { color:#8a94a3; font-size:11px; margin-bottom:7px; } .workspace-title { color:#172033; font-size:30px; font-weight:760; letter-spacing:-.035em; margin-bottom:4px; } .workspace-copy { color:#697586; font-size:13px; margin-bottom:22px; } .answer-card { background:#fff; border:1px solid #e3e7ed; border-radius:16px; padding:20px; box-shadow:0 8px 25px rgba(16,24,40,.04); } .answer-meta { color:#7a8494; font-size:11px; margin-bottom:10px; } .source-card { background:#f8fafc; border:1px solid #e8ecf1; border-radius:12px; padding:12px 14px; margin-top:8px; } .source-title { color:#344054; font-size:12px; font-weight:700; margin-bottom:3px; } .source-snippet { color:#7a8494; font-size:11px; line-height:1.5; } div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea { border-radius:10px; border-color:#d1d7e0; } div[data-baseweb="select"] > div { border-radius:10px; min-height:46px; border-color:#d1d7e0; } .stButton > button, .stLinkButton > a { border-radius:10px !important; min-height:46px !important; font-weight:700 !important; } .stLinkButton > a { width:100%; justify-content:center; text-decoration:none !important; } .sidebar-status { padding:12px; border:1px solid #e3e7ed; border-radius:12px; background:#fff; color:#596579; font-size:11px; line-height:1.5; } @media(max-width: 800px) { .setup-title {font-size:32px;} .block-container {padding:24px 18px 50px;} } </style>

""",
unsafe_allow_html=True,
)

============================================================
OPTIONAL SERVER KEY: DO NOT AUTO-SHOW WORKSPACE
============================================================
We deliberately require the connection step. A server key may be
available for deployment, but it does not bypass the first screen.

if not st.session_state.connected:
# --------------------------------------------------------
# PAGE 1 — OPENAI CONNECTION
# --------------------------------------------------------
st.markdown(
"""
<div class="mm-brand">
<div class="mm-mark">M</div>
<div class="mm-name">MarketMind AI</div>
<div class="mm-pill"><span class="mm-dot"></span> Secure setup</div>
</div>
""",
unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="setup-eyebrow">SET UP YOUR WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="setup-title">Connect your AI research engine.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="setup-copy">
            Add an OpenAI API key and choose the model MarketMind should
            use. We verify the connection before opening the research workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )

    bullets = [
        ("01", "Connect directly to your OpenAI account"),
        ("02", "Choose the GPT model used for responses"),
        ("03", "Ask research questions after the connection succeeds"),
    ]
    for number, text in bullets:
        st.markdown(
            f"""
            <div style="display:flex;gap:12px;align-items:center;margin:0 0 14px 0;">
                <div style="font-size:10px;font-weight:800;color:#7a8494;
                            border:1px solid #e2e7ee;border-radius:7px;padding:5px 7px;">
                    {number}
                </div>
                <div style="font-size:13px;color:#475467;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
        <div class="setup-card">
            <div class="card-kicker">Connect OpenAI</div>
            <div class="card-sub">Your API key is used only for requests made in this session.</div>
        """,
        unsafe_allow_html=True,
    )

    entered_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="sk-••••••••••••••••••••",
        label_visibility="visible",
    )
    st.markdown(
        '<div class="field-note">Your key is used only to authenticate your OpenAI requests.</div>',
        unsafe_allow_html=True,
    )

    selected_model_label = st.selectbox(
        "AI model",
        MODEL_LABELS,
        index=MODEL_LABELS.index(DEFAULT_MODEL_LABEL),
        help="The default is intentionally set to the oldest model in the current list.",
    )
    selected_model = MODEL_IDS[selected_model_label]

    descriptions = {
        "GPT-3.5 Turbo": "Legacy, lowest-generation chat model in this list.",
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
        f'<div class="field-note">{descriptions.get(selected_model_label, "")}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button("Connect & Continue →", type="primary", use_container_width=True):
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

                # Key is kept only in Streamlit session state.
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
        <div class="security-row">
            <div class="security-icon">✓</div>
            <div>Your API key is kept in session memory and is never written to the project files or run logs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # IMPORTANT: Get API Key is intentionally the LAST control.
    st.markdown(
        """
        <div class="get-key-wrap">
            <div class="get-key-label">Don't have an OpenAI API key?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button(
        "Get an OpenAI API Key ↗",
        OPENAI_API_KEY_URL,
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


st.stop()
============================================================
PAGE 2 — QUESTION / RESPONSE WORKSPACE
============================================================

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
Render chat history.

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

Close
