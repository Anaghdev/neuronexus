# neuronexus.py
# NeuroNexus – AI Life Companion
# Developed for InnoCode Competition

import streamlit as st
import pandas as pd
import plotly.express as px
import random
import datetime
import requests
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline
from PIL import Image
import base64
import torch

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="NeuroNexus – AI Life Companion", layout="wide")

st.title("🧠 NeuroNexus – AI Life Companion")
st.write("Your all-in-one AI-powered wellness, productivity, and creativity hub.")

# -----------------------
# CACHED FUNCTIONS
# -----------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_chatbot_safe():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "microsoft/DialoGPT-small"  # smaller model for Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def analyze_mood(user_input):
    sentiment_pipeline = load_sentiment_model()
    result = sentiment_pipeline(user_input)[0]
    return result['label'], result['score']

def chatbot_reply(user_input, chat_history_ids, tokenizer, model):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply, chat_history_ids

def get_weather(city):
    API_KEY = "YOUR_OPENWEATHER_API_KEY"  # Replace
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("main"):
        return response["main"]["temp"], response["weather"][0]["description"]
    return None, None

def generate_pdf(content, filename="report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text_object = c.beginText(50, 750)
    for line in content.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF</a>'
    return href

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
        st.write(f"**Sentiment:** {label} ({score:.2f})")
        if label == "POSITIVE":
            st.success("Keep up the good vibes! Here's some upbeat music:")
            st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC", height=200)
        else:
            st.warning("It's okay to have down days. Try this calming playlist:")
            st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1DWZeKCadgRdKQ", height=200)

# -----------------------
# TAB 2: AI Chat
# -----------------------
with tabs[1]:
    st.header("🤖 AI Chat Companion")

    if st.button("Load Chatbot"):
        with st.spinner("Loading AI Chatbot... Please wait ~10-15 seconds"):
            try:
                tokenizer, model = load_chatbot_safe()
                st.session_state['chatbot'] = (tokenizer, model, None)
                st.success("✅ Chatbot loaded! Start chatting below.")
            except Exception as e:
                st.error(f"⚠️ Failed to load chatbot: {e}")

    if 'chatbot' in st.session_state:
        user_msg = st.text_input("You:")
        if user_msg:
            tokenizer, model, chat_history = st.session_state['chatbot']
            try:
                reply, chat_history = chatbot_reply(user_msg, chat_history, tokenizer, model)
                st.session_state['chatbot'] = (tokenizer, model, chat_history)
                st.text_area("Bot:", value=reply, height=100)
            except Exception as e:
                st.error(f"⚠️ Chatbot error: {e}")

# -----------------------
# TAB 3: Goal Planner
# -----------------------
with tabs[2]:
    st.header("📅 AI Goal Planner")
    goal = st.text_input("Enter your goal:")
    if goal:
        steps = [f"Step {i+1}: {goal} - subtask {i+1}" for i in range(5)]
        st.write("\n".join(steps))
        pdf_link = generate_pdf("\n".join(steps), "goal_plan.pdf")
        st.markdown(pdf_link, unsafe_allow_html=True)

# -----------------------
# TAB 4: Wellness & Fitness
# -----------------------
with tabs[3]:
    st.header("💪 Wellness & Fitness Coach")
    st.subheader("Breathing Exercise")
    st.write("Inhale... Hold... Exhale... (Repeat for 1 min)")
    st.subheader("Healthy Meal Suggestion")
    meals = ["Grilled salmon with veggies", "Quinoa salad", "Oatmeal with berries", "Avocado toast"]
    st.info(random.choice(meals))
    st.subheader("Exercise Suggestion")
    exercises = ["Push-ups", "Squats", "Yoga", "Plank"]
    st.success(random.choice(exercises))

# -----------------------
# TAB 5: Tracker
# -----------------------
with tabs[4]:
    st.header("📈 Habit & Mood Tracker")
    date = st.date_input("Date", datetime.date.today())
    mood = st.selectbox("Mood", ["Happy", "Sad", "Neutral", "Stressed"])
    if st.button("Save Entry"):
        df = pd.DataFrame([[date, mood]], columns=["Date", "Mood"])
        try:
            existing = pd.read_csv("progress.csv")
            df = pd.concat([existing, df], ignore_index=True)
        except FileNotFoundError:
            pass
        df.to_csv("progress.csv", index=False)
        st.success("Saved!")
    try:
        df = pd.read_csv("progress.csv")
        fig = px.histogram(df, x="Mood")
        st.plotly_chart(fig)
    except FileNotFoundError:
        st.write("No data yet.")

# -----------------------
# TAB 6: Creativity Corner
# -----------------------
with tabs[5]:
    st.header("🎨 Creativity Corner")
    st.write("AI-generated Quote:")
    quotes = ["Believe you can and you're halfway there.", "Every day is a new beginning."]
    st.success(random.choice(quotes))

# -----------------------
# TAB 7: Add-ons
# -----------------------
with tabs[6]:
    st.header("🎯 Fun & Add-ons")
    city = st.text_input("Enter your city for weather:")
    if city:
        temp, desc = get_weather(city)
        if temp:
            st.info(f"{city}: {temp}°C, {desc}")
        else:
            st.error("City not found.")
    st.subheader("Random Fun Fact")
    facts = ["Honey never spoils.", "Bananas are berries.", "Sharks existed before trees."]
    st.write(random.choice(facts))
