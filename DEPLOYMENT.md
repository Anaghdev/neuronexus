# Deploying NeuroNexus on Streamlit Community Cloud

1. **Create a GitHub repo** and push the `neuronexus/` folder contents.
2. Go to **https://share.streamlit.io** → **New app**.
3. Select your repo and branch. Set:
   - **Main file path:** `streamlit_app.py`
   - (Optional) **Python version:** from `runtime.txt` if prompted (3.10)
4. Click **Advanced settings → Secrets** and add (optional):
   ```toml
   OPENWEATHER_API_KEY = "your_openweather_key"
   ```
5. Click **Deploy**. Your app will build and go live.

### Alternative: Local Docker (optional)

```bash
# Build
docker build -t neuronexus .

# Run
docker run -p 8501:8501 neuronexus
```

### Troubleshooting

- If build fails on Community Cloud, switch to the looser pins in `requirements_streamlit.txt` (set that as your install file).
- If you see “No module named ...”, ensure your **Main file path** is `streamlit_app.py` (not `neuronexus.py`).
- For slow cold starts, use smaller models or pre‑load them inside an action button.
