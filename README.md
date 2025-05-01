<p align="center">
  <img src="https://raw.githubusercontent.com/zosialenarczyk/wildfire-prevention/main/data/raw/banner3.png" alt="Wildfire Prevention AI System Banner" width="100%">
</p>

## Project Overview

This project was developed for the **42578 Advanced Business Analytics** course at DTU, under the theme **"AI for the Betterment of Society"**. The objective is to support wildfire prevention and emergency preparedness in Greece through a collaborative **multi-agent AI system** powered by machine learning and simulation.

Wildfires in Greece have become more frequent and destructive due to climate change, threatening ecosystems, tourism, agriculture, and public safety. To address this, we built an AI-based decision framework that predicts wildfire risk, simulates fire spread, and supports emergency resource allocation.

At the heart of the system are **three core models** and **six collaborative agents**, orchestrated via a FastAPI backend and optionally integrated into a Streamlit UI:

---

### Wildfire Prediction Model 
A binary classification model trained on 25 years of daily weather data to predict the likelihood of wildfire occurrence in each Greek region. The model outputs both probability scores and fire/no-fire predictions, and was optimized using **F1 score**

---

### Fire Spread Simulation Model 
A convolutional neural network inspired by the FireFore architecture, using terrain raster data to predict how fires might expand the following day. This spatial model helps simulate dynamic fire paths across 64×64 km satellite image tiles.

---

### Resource Optimization Model
A rule-based prioritization system that translates predicted fire probabilities into actionable firefighting deployment strategies. It incorporates severity, regional exposure, and resource limits to suggest where firefighting units should be positioned. This model supports the **Resource Planner Agent** by simulating firefighting needs per region.

---

## Multi-Agent System

The models are embedded into a **multi-agent CrewAI architecture** that mimics a real-world emergency response workflow. Each agent focuses on a domain-specific task:

- **Fire Prediction Agent** – Uses the XGBoost model to assess daily fire risk per region.
- **Fire Spread Agent** – Predicts next-day spatial fire movement using the CNN model.
- **Resource Planner Agent** – Translates risk and severity scores into optimal firefighting deployment strategies.
- **Evacuation Coordinator Agent** – Prioritizes areas for evacuation based on risk and terrain conditions.
- **Tourism Safety Advisor Agent** – Issues safety guidance for travelers in affected regions.
- **Public Communication Manager Agent** – Aggregates agent outputs and issues structured public alerts.

---

Together, the models and agents deliver real-time, location-specific recommendations to help authorities **detect, simulate, and respond** to wildfire threats in Greece—supporting faster and more coordinated decision-making under climate uncertainty.




## Project Structure

```bash
wildfire-prediction/
├── data/
│   ├── raw/                         # Weather, fire, GADM, satellite data
│   ├── processed/                   # Cleaned datasets
│   └── demos/                       # Video demos of agents and alerts
│       ├── agents_communication.mp4
│       ├── wildfire_demo.mp4
│       └── safety_alert.mp4
│
├── docs/                            # Reports and papers
│   └── agents_description.docx
│
├── models/                          # Trained model files
│
├── notebooks/
│   ├── Agents/
│   │   └── Multi_Agent_System.ipynb
│   ├── Fire_prediction/
│   │   ├── DataMerging.ipynb
│   │   ├── dynamic_weather_forecast.ipynb
│   │   └── XGBoost_Predict_wildfire.ipynb
│   └── firespread_modellisation/
│       ├── firespread_cnn.py
│       ├── collect_environment_parameters.py
│       └── res/ ...
│
├── src/
│   ├── api/
│   │   ├── fire_agents_api.py        # FastAPI endpoints
│   │   └── models/                   # Prediction scripts
│   │       └── xgboost_model.pkl
│   ├── agents/                       # Agent code (CrewAI)
│   ├── orchestrator/                # Coordination logic
│   └── utils/                       # Helper functions
│
├── streamlit_app/                  # Optional frontend 
│
├── .env
├── .gitignore
├── requirements.txt
├── run_api.sh
└── README.md                   