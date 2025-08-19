# Deploying NeuroNexus on Streamlit Cloud

## 1) Repo contents
Make sure your repo looks like this at the root:
- `streamlit_app.py` (entrypoint)
- `neuronexus.py` (the app logic)
- `requirements.txt` (or use `requirements_streamlit.txt`)
- `.streamlit/secrets.toml` (add your keys)
- `runtime.txt` (Python version)

## 2) Requirements
If your current `requirements.txt` fails to build, rename it and use `requirements_streamlit.txt`.

## 3) Secrets
In Streamlit Cloud → **⚙️ Settings → Secrets**, paste:
```
OPENWEATHER_API_KEY = "your-openweather-key"
```
> A local `.streamlit/secrets.toml` file is provided as a template for local dev only.

## 4) Deploy
- Push this folder to GitHub.
- In Streamlit Cloud, create an app → pick the repo and branch.
- **Main file path:** `streamlit_app.py` (or `app.py`).
- Click **Deploy**.

## 5) Local run
```
pip install -r requirements.txt   # or requirements_streamlit.txt
streamlit run streamlit_app.py
```
