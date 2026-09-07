# MarketMind AI — OpenAI Connection + Q&A

A Streamlit implementation of the MarketMind connection flow.

## Required flow

1. The first screen asks for the OpenAI API key.
2. The model selector appears **after** the API-key field.
3. The **oldest model in the existing model list — GPT-3.5 Turbo — is selected by default**.
4. The **Get an OpenAI API Key** button is the final control at the bottom of the connection card.
5. Clicking **Connect & Continue** makes a real OpenAI request to verify the key/model before the workspace opens.
6. After connection, the user can ask questions and receive OpenAI responses.
7. The API key is kept only in Streamlit session state and is never written to run JSON files.

## Model list

The existing model IDs are preserved:

- GPT-3.5 Turbo — `gpt-3.5-turbo` **(default)**
- GPT-4 — `gpt-4`
- GPT-4 Turbo — `gpt-4-turbo`
- GPT-4o — `gpt-4o`
- GPT-4o Mini — `gpt-4o-mini`
- GPT-4.1 — `gpt-4.1`
- GPT-4.1 Mini — `gpt-4.1-mini`
- GPT-5.6 Luna — `gpt-5.6-luna`
- GPT-5.6 Terra — `gpt-5.6-terra`
- GPT-5.6 Sol — `gpt-5.6-sol`

Note: OpenAI may retire or restrict legacy models. If a selected model is unavailable for an account, the connection screen shows a friendly error and the user can select another model.

## Run locally

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## API key

The connection screen links to:

https://platform.openai.com/api-keys

Do not commit a real key to GitHub.

For Streamlit Community Cloud, you can optionally configure `OPENAI_API_KEY` in App Settings → Secrets. The app still requires the connection screen before opening the workspace.

## Security

- No API key is hard-coded.
- User-entered API keys are stored only in `st.session_state` for the active session.
- API keys are not written to `runs/*.json`.
- API keys are not placed in corpus metadata.
- Do not put secrets in screenshots, GitHub issues, or documentation.

## Project structure

```text
marketmind_openai_connection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── corpus/
│   └── demo.json
├── runs/
│   └── .gitkeep
└── src/
    ├── __init__.py
    └── README.md
```

## Demo corpus

`corpus/demo.json` is only a small local source example for the Q&A screen. It is not required for the OpenAI connection itself.

## Notes

The app uses the OpenAI Responses API for the connection test and question answering. The UI order intentionally remains:

**API key → model → Connect → security → Get API Key**

The last item is the Get API Key button, as requested.
