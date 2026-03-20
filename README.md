
# Predictive Maintenance System (MRI/CT)

## Overview
End-to-end LSTM-based predictive maintenance system using IoT data.

## Features
- LSTM model
- FastAPI backend
- Streamlit dashboard
- Dockerized deployment

## Run
pip install -r requirements.txt
python src/train.py
uvicorn api.app:app --reload

## Dashboard
streamlit run dashboard/streamlit_app.py
