#!/bin/bash
# Start FastAPI backend
PYTHONPATH=. uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 &


# Start Streamlit Dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
