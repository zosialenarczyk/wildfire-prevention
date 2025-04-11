<p align="center">
  <img src="https://raw.githubusercontent.com/zosialenarczyk/wildfire-prevention/main/Banner2.png" alt="Wildfire Prediction AI System Banner" width="100%">
</p>

## Project overview

This project was developed as part of the 42578 Advanced Business Analytics course at DTU, under the theme “AI for the Betterment of Society.” Our aim is to contribute to wildfire prevention and emergency preparedness in Greece through the development of a multi-agent AI system.

Wildfires in Greece have increased in frequency and intensity over recent decades, driven by climate change and unpredictable weather. These events carry significant environmental, social, and economic costs. Our project addresses this challenge by using AI to improve fire risk prediction, simulate fire spread, and optimize emergency response strategies.

At its core, the system consists of several AI agents that collaborate through a FastAPI framework:

- A fire risk prediction agent based on XGBoost on daily weather data (2000–2024)  
- A fire spread agent, using the FireFore model and landscape data  
- A resource optimization agent to simulate emergency response planning  
- A recommendation agent that synthesizes model outputs via an LLM (DeepSeek)  

The project focuses on real-time, region-specific insights to support rapid, data-driven decisions in wildfire management.


## Project structure

```bash
wildfire-prediction/
├── data/                  # Raw and processed data
├── notebooks/             # ML training notebooks
├── models/                # Trained model files
├── src/
│   ├── api/               # FastAPI endpoints
│   ├── agents/            # Agent logic (CrewAI)
│   ├── models/            # Model loading/prediction
│   ├── orchestrator/      # Multi-agent chaining
│   └── utils/             # Helper functions (e.g. API key loader)
├── streamlit_app/         # Streamlit UI 
├── .env                   # API keys and secrets
├── .gitignore             # Files to exclude from version control
├── requirements.txt       # Python dependencies
├── run_api.sh             # Script to run FastAPI server
└── README.md              # You're here!