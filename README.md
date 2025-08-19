# 🧠 NeuroNexus — AI Life Companion

NeuroNexus is an **all‑in‑one, privacy‑friendly AI companion** built with Streamlit. It blends **mood support**, **human‑like chat**, **goal planning**, **wellness coaching**, **habit tracking**, and a little **creative spark**—all in one clean UI.

This repo includes a ready‑to‑deploy Streamlit app plus simple, local data exports (no external DB required).

---

## ✨ What’s inside

- **💬 Mood Check‑in** — quick journal + sentiment analysis (Transformers), actionable nudges, optional Spotify embeds.
- **🤖 AI Chat Companion** — human‑like responses with adjustable temperature/top‑p, optional “typing” animation.
- **📅 AI Goal Planner** — SMART goals, auto‑generated steps, one‑click PDF export.
- **💪 Wellness & Fitness Coach** — habit tips, meal ideas, breathing guides.
- **📈 Habit & Mood Tracker** — log moods, visualize trends with charts.
- **🎨 Creativity Corner** — quotes, poems, idea starters.
- **🎯 Add‑ons** — weather lookup (OpenWeather), fun facts.

### 🔧 New in this version
- **Profile in sidebar**: personalize with your name.
- **One‑click data export**: download **chat history (JSON)** and **mood tracker (CSV)** from the sidebar.
- **Safe reset**: clear app state from the sidebar without restarting the app.
- **Smoother chat**: humanized replies with optional typing effect.

---

## 🚀 Quick start

```bash
# 1) Create & activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run streamlit_app.py
```

> Python **3.10** is recommended (see `runtime.txt`).

---

## 🔐 Secrets (optional but recommended)

Create `.streamlit/secrets.toml` and add your keys:

```toml
# .streamlit/secrets.toml
OPENWEATHER_API_KEY = "your_openweather_key"
```

These are used in **Add‑ons → Weather**.

---

## 🗂️ Project structure

```
neuronexus/
  app.py
  streamlit_app.py
  neuronexus.py         # main UI/logic
  requirements.txt
  requirements_streamlit.txt
  DEPLOYMENT.md
  README.md
  .streamlit/
    secrets.toml        # (you create this locally)
```

---

## 🛠️ Configuration (Sidebar)

- **Chat settings**: max tokens, temperature, top‑p, repetition penalty, typing speed.
- **Profile**: your name (used to personalize UI text).
- **Data**: download chat history (JSON) and tracker data (CSV); reset app state.

---

## 📤 Exports

- **Chat history** → `neuronexus_chat.json`
- **Mood tracker** → `mood_tracker.csv`
- **Goal plans** → PDF (from the Goal Planner tab)

> All data stays in the session until you download it—no cloud DBs by default.

---

## 🧩 Tech stack

- **Streamlit** UI
- **Transformers (Hugging Face)** for sentiment
- **Plotly Express** for charts
- **ReportLab** for PDF export
- **Requests** for weather API

---

## 📦 Deploying

See **`DEPLOYMENT.md`** for Streamlit Community Cloud steps. TL;DR:
- Push this folder to a public repo.
- Set `Main file path` to `streamlit_app.py`.
- (Optional) Add `OPENWEATHER_API_KEY` in the app’s Secrets.
- Click **Deploy**.

---

## 🗺️ Roadmap ideas

- Local persistence (SQLite) behind a toggle
- Voice input & TTS
- Calendar sync for goals
- AI coach personas (calm / energetic / clinical)
- Theming via `config.toml`

PRs welcome!
