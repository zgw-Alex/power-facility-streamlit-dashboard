# COMP5339 Assignment 2 Streamlit Dashboard

Streamlit dashboard for NEM facility power and emissions data.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

## Deploy on Streamlit Community Cloud

Use `streamlit_dashboard.py` as the main app file and `requirements.txt` as the Python dependency file.

If environment variables are required, add them in Streamlit Cloud under app settings or secrets. Do not commit `.env` or `.streamlit/secrets.toml`.

