# 🛡️ YSS AI: Hybrid Safety Navigation System

An AI-powered navigation engine that predicts the safest route through Delhi using historical crime patterns and real-time street-level routing.

## 🚀 Key Features
- **Dual-Path Visualization:** Compares the shortest path (Red) vs. the AI-recommended Safe Path (Green).
- **Dynamic Risk Multiplier:** Adjusts safety scores based on the Time of Day (Night-time high-risk logic).
- **Hybrid Geolocation:** Supports both Browser GPS Tracking and Manual Area Selection.
- **XGBoost Integration:** Predicts destination risk using a pre-trained ML model.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Routing Engine:** OSRM (Open Source Routing Machine) API
- **Maps:** Folium & Leaflet.js
- **ML Model:** XGBoost Regressor

## 🔧 Installation
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/YSS-AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run src/app.py`
