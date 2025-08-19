from __future__ import annotations
# === GLOBAL SANITIZE HELPER ===
def _sanitize(text: str) -> str:
    """Lightly clean model output for UI safety/clarity."""
    import re
    text = (text or "").strip()
    text = re.sub(r"https?://\S+|\S+@\S+", "", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sents[:3]).strip()
    return text

# neuronexus.py
# NeuroNexus – AI Life Companion (Human-like Chat + Refactor)
# Streamlit app

import os, io, base64, datetime as dt, random, time
from typing import List, Tuple, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas
from transformers import pipeline


# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="NeuroNexus – AI Life Companion", layout="wide")
st.title("🧠 NeuroNexus – AI Life Companion")
st.write("Your all-in-one AI-powered wellness, productivity, and creativity hub.")


# -----------------------
# CACHED MODELS
# -----------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )

@st.cache_resource(show_spinner=True)

@st.cache_resource(show_spinner=False)
def get_cached_chatbot():
    """Return (tokenizer, model) cached across reruns to prevent reloads."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "microsoft/DialoGPT-small"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    return tok, mdl

def load_chatbot_safe():
    # Small + widely available model for Streamlit deployments
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


# -----------------------
# HELPERS
# -----------------------
def analyze_mood(user_input: str) -> Tuple[Optional[str], Optional[float]]:
    try:
        sentiment = load_sentiment_model()
        result = sentiment(user_input, truncation=True)[0]
        return result.get("label"), float(result.get("score", 0))
    except Exception:
        return None, None

def generate_pdf(content: str, filename: str = "report.pdf") -> str:
    buffer = io.BytesIO()
    c = rl_canvas.Canvas(buffer, pagesize=letter)
    text_object = c.beginText(50, 750)
    for line in content.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF</a>'

def get_weather(city: str) -> Tuple[Optional[float], Optional[str]]:
    api_key = st.secrets.get("OPENWEATHER_API_KEY")
    if not api_key:
        return None, None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        r = requests.get(url, params={"q": city, "appid": api_key, "units": "metric"}, timeout=10)
        data = r.json()
        if "main" in data:
            return float(data["main"]["temp"]), data["weather"][0]["description"]
    except Exception:
        pass
    return None, None

def init_state():
    st.session_state.setdefault("chat_tokens", None)  # (tokenizer, model)
    st.session_state.setdefault("chat_history", [])   # [{"role":"user|assistant","text": str}]
    st.session_state.setdefault("tracker_df", pd.DataFrame(columns=["Date", "Mood"]))
    st.session_state.setdefault("typing_speed", 0.02)

init_state()


# -----------------------
# SIDEBAR SETTINGS
# -----------------------
with st.sidebar:
    st.subheader("Chat Settings")
    max_new = st.slider("Max new tokens", 16, 512, 160, 16)
    temp = st.slider("Temperature", 0.0, 1.5, 0.9, 0.05)
    top_p = st.slider("Top‑p", 0.1, 1.0, 0.9, 0.05)
    rep_pen = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)
    human_mode = st.toggle("Human-like responses", value=True,
                           help="Adds empathy, concise phrasing, and a follow‑up question.")
    typing_speed = st.slider("Typing speed (seconds per word)", 0.0, 0.08, st.session_state.typing_speed, 0.005)
    st.session_state.typing_speed = typing_speed

    st.divider()
    # New: Profile & Data
    st.subheader("Profile")
    st.session_state.setdefault("profile_name", "")
    st.session_state.profile_name = st.text_input("Your name (optional)", value=st.session_state.profile_name)

    st.subheader("Data Export / Reset")
    # Prepare downloads
    chat_json = io.StringIO()
    try:
        import json as _json
        _json.dump(st.session_state.get("chat_history", []), chat_json, ensure_ascii=False, indent=2)
    except Exception:
        chat_json = io.StringIO("[]")
    chat_bytes = chat_json.getvalue().encode("utf-8")

    if st.download_button("⬇️ Download chat history (JSON)", data=chat_bytes, file_name="neuronexus_chat.json", mime="application/json"):
        st.toast("Chat history downloaded")

    # Tracker CSV
    try:
        _df = st.session_state.get("tracker_df")
        if _df is not None and not _df.empty:
            csv_bytes = _df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download mood tracker (CSV)", data=csv_bytes, file_name="mood_tracker.csv", mime="text/csv")
        else:
            st.caption("No tracker data yet.")
    except Exception:
        pass

    if st.button("🗑️ Reset app state"):
        for k in list(st.session_state.keys()):
            if k not in ("typing_speed",):
                del st.session_state[k]
        st.session_state.typing_speed = 0.02
        st.rerun()

    st.caption("Tip: set an OpenWeather key in `.streamlit/secrets.toml`:\n\n"
               "[OPENWEATHER_API_KEY]\nOPENWEATHER_API_KEY=\"your_key_here\"")


# -----------------------
# TABS
# -----------------------
tabs = st.tabs([
    "💬 Mood Check-in",
    "🤖 AI Chat",
    "📅 Goal Planner",
    "💪 Wellness & Fitness",
    "📈 Tracker",
    "🎨 Creativity Corner",
    "🎯 Add-ons"
])


# -----------------------
# TAB 1: Mood Check-in
# -----------------------
with tabs[0]:
    st.header("💬 Mood Check-in")
    mood_input = st.text_input("How are you feeling today?")
    if mood_input:
        label, score = analyze_mood(mood_input)
        if label:
            st.write(f"**Sentiment:** {label.title()} ({score:.2f})")
        if label == "POSITIVE":
            st.success("Love the energy! Here’s some upbeat music:")
            st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC", height=200)
        elif label == "NEGATIVE":
            st.warning("It’s okay to have down days. Try this calming playlist:")
            st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1DWZeKCadgRdKQ", height=200)


# -----------------------
# TAB 2: AI Chat (Human-like)
# -----------------------
with tabs[1]:
    st.header("🤖 AI Chat Companion")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Load Chatbot"):
    with st.spinner("Loading AI Chatbot…"):
        try:
            tok, mdl = get_cached_chatbot()
            st.session_state["chat_tokens"] = (tok, mdl)
            st.session_state["chat_ready"] = True
            st.success("✅ Chatbot loaded! Start chatting below.")
        except Exception as e:
            st.error(f"⚠️ Failed to load chatbot: {e}")
with c2:
    if st.button("Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()
with c3:
    st.caption("DialoGPT-small • local inference")


    # Render chat using modern UI
    if st.session_state.get("chat_tokens") or st.session_state.get("chat_ready"):
        tok, mdl = st.session_state["chat_tokens"]

        # Show history
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"], avatar=("👤" if m["role"] == "user" else "🤖")):
                st.markdown(m["text"])

        # Input
        user_msg = st.chat_input("Type your message…")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "text": user_msg})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_msg)

            try:
                # Build dialogue: keep last few user turns (DialoGPT is single-turn leaning)
                eos = tok.eos_token or ""
                user_turns = [m["text"] for m in st.session_state.chat_history if m["role"] == "user"]
                primer = (
                    "You are Neuro, a warm, concise chat companion.\n"
                    "Guidelines: be empathetic, natural, short sentences, avoid overpromising. "
                    "Ask one relevant follow-up. Avoid medical/legal claims.\n"
                ) if human_mode else ""
                dialogue = primer + eos.join(user_turns[-6:]) + eos
                input_ids = tok.encode(dialogue, return_tensors="pt")

                output_ids = mdl.generate(
                    input_ids,
                    max_new_tokens=int(max_new),
                    do_sample=True,
                    temperature=float(temp),
                    top_p=float(top_p),
                    repetition_penalty=float(rep_pen),
                    pad_token_id=tok.eos_token_id,
                    early_stopping=True,
                )
                raw = tok.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
raw = _sanitize(raw)
if not raw:
    raw = "I'm here. Tell me a bit more so I can help."

                # Humanize + sanitize pass
                def _sanitize(text: str) -> str:
                    import re
                    text = (text or "").strip()
                    text = re.sub(r"https?://\S+|\S+@\S+", "", text)
                    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
                    sents = re.split(r"(?<=[.!?])\s+", text)
                    text = " ".join(sents[:3]).strip()
                    return text

                # Humanize pass
                def humanize(text: str, context: str) -> str:
                    # Trim & tidy
                    text = (text or "").replace("\n\n", "\n").strip()
                    if len(text) > 700:
                        text = text[:700].rsplit(". ", 1)[0] + "."
                    # Mood-aware opener
                    label, _ = analyze_mood(context)
                    opener = "I hear you. " if (label and label.upper() == "NEGATIVE") else ""
                    # One follow-up
                    follow = "\n\nWhat would be a good next step for you?" if human_mode else ""
                    # Keep it conversational
                    parts = [p.strip() for p in text.split("\n") if p.strip()]
                    concise = " ".join(parts)
                    # Light variety
                    prefix = random.choice(["", "", "Hmm, ", "Okay, "])
                    return (opener + prefix + concise).strip() + follow

                reply = humanize(raw, user_msg) if human_mode else (raw or "(no response)")

                # Typing effect
                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    shown = ""
                    for token in reply.split(" "):
                        shown = (shown + " " + token).strip()
                        placeholder.markdown(shown)
                        if st.session_state.typing_speed:
                            time.sleep(st.session_state.typing_speed)

                st.session_state.chat_history.append({"role": "assistant", "text": reply})
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Chatbot error: {e}")
    else:
        st.info("Click **Load Chatbot** to begin.")
        # Quick starters
        st.caption("Try a starter:")
        cols = st.columns(4)
        starters = [
            "Help me plan my day",
            "I feel overwhelmed—any quick tips?",
            "Give me a 2‑minute breathing routine",
            "Suggest a quick healthy lunch",
        ]
        for i, s in enumerate(starters):
            if cols[i].button(s):
                st.session_state.chat_history.append({"role": "user", "text": s})
                st.rerun()


# -----------------------
# TAB 3: Goal Planner
# -----------------------
with tabs[2]:
    st.header("📅 AI Goal Planner")
    goal = st.text_input("Enter your goal:")
    if goal:
        # Build a simple SMART-ish plan without external models
        plan = [
            f"Goal: {goal}",
            "",
            "Timeline:",
            "- Week 1: Define scope and break goal into tasks",
            "- Week 2: Gather resources and block sessions",
            "- Week 3: Execute core tasks and track progress",
            "- Week 4: Review outcomes, fix gaps, and finalize",
            "",
            "Tasks:",
            f"- Outline what '{goal}' means and success criteria",
            f"- Prepare materials/tools for '{goal}'",
            f"- Do 3 focused sessions (45–60 min) for '{goal}'",
            f"- Summarize learnings and next steps",
            "",
            "Resources:",
            "- Calendar blocks for deep work",
            "- Notes app or doc for tracking",
            "",
            "Risks & Mitigations:",
            "- Distractions → Use Do Not Disturb",
            "- Over-scope → Keep tasks small and time-boxed",
        ]
        body = "\n".join(plan)
        st.markdown(body)
        pdf_link = generate_pdf(body, "goal_plan.pdf")
        st.markdown(pdf_link, unsafe_allow_html=True)


# -----------------------
# TAB 4: Wellness & Fitness
# -----------------------
with tabs[3]:
    st.header("💪 Wellness & Fitness Coach")

    st.subheader("Breathing Exercise")
    st.write("Inhale 4s • Hold 4s • Exhale 6s — repeat for 1–2 minutes.")

    st.subheader("Healthy Meal Suggestion")
    meals = [
        "Grilled salmon with veggies",
        "Quinoa salad with chickpeas",
        "Oatmeal with berries & nuts",
        "Avocado toast + egg",
        "Tofu stir-fry with greens",
    ]
    st.info(random.choice(meals))

    st.subheader("Exercise Suggestion")
    exercises = ["Push-ups (2×10)", "Bodyweight squats (2×12)", "Yoga sun salutation (3 rounds)", "Plank (2×30s)"]
    st.success(random.choice(exercises))


# -----------------------
# TAB 5: Tracker
# -----------------------
with tabs[4]:
    st.header("📈 Habit & Mood Tracker")

    date = st.date_input("Date", dt.date.today())
    mood = st.selectbox("Mood", ["Happy", "Sad", "Neutral", "Stressed", "Calm", "Excited"])

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Save Entry"):
            row = pd.DataFrame([[str(date), mood]], columns=["Date", "Mood"])
            st.session_state.tracker_df = pd.concat([st.session_state.tracker_df, row], ignore_index=True)
            st.success("Saved!")

    with c2:
        if st.button("Clear All"):
            st.session_state.tracker_df = pd.DataFrame(columns=["Date", "Mood"])
            st.warning("Cleared.")

    with c3:
        # CSV download
        if not st.session_state.tracker_df.empty:
            csv = st.session_state.tracker_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "progress.csv", "text/csv")

    # Upload CSV merge
    up = st.file_uploader("Upload a CSV to merge (columns: Date,Mood)", type=["csv"])
    # Template download
    st.download_button('Download CSV template', data='Date,Mood\n2025-01-01,Happy\n', file_name='tracker_template.csv', mime='text/csv')
    if up is not None:
        try:
            up_df = pd.read_csv(up)
            up_df = up_df[["Date", "Mood"]]
            st.session_state.tracker_df = pd.concat([st.session_state.tracker_df, up_df], ignore_index=True)
            st.success("Merged uploaded entries.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    # Show & simple viz
    if st.session_state.tracker_df.empty:
        st.write("No data yet.")
    else:
        st.dataframe(st.session_state.tracker_df, use_container_width=True)
        try:
            fig = px.histogram(st.session_state.tracker_df, x="Mood", title="Mood Frequency")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# -----------------------
# TAB 6: Creativity Corner
# -----------------------
with tabs[5]:
    st.header("🎨 Creativity Corner")
    st.write("AI-generated Quote:")
    quotes = [
        "Believe you can and you're halfway there.",
        "Every day is a new beginning.",
        "Small steps become big changes.",
        "Progress, not perfection.",
    ]
    st.success(random.choice(quotes))


# -----------------------
# TAB 7: Add-ons
# -----------------------
with tabs[6]:
    st.header("🎯 Fun & Add-ons")

    st.subheader("Weather")
    city = st.text_input("Enter your city:")
    if city:
        temp_c, desc = get_weather(city)
        if temp_c is not None:
            st.info(f"{city}: {temp_c:.1f}°C, {desc}")
        else:
            if "OPENWEATHER_API_KEY" not in st.secrets:
                st.error("Add OPENWEATHER_API_KEY to `.streamlit/secrets.toml`.")
            else:
                st.error("City not found or API error.")

    st.subheader("Random Fun Fact")
    facts = [
        "Honey never spoils.",
        "Bananas are berries.",
        "Sharks existed before trees.",
        "Octopuses have three hearts.",
    ]
    st.write(random.choice(facts))