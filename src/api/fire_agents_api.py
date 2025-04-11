# load the .env file and get the api key from it
import os
from dotenv import load_dotenv
import requests  # make sure requests is installed: pip install requests

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from fastapi import FastAPI #Create your app and define HTTP endpoints (API routes)
from pydantic import BaseModel #Validate and define structure for incoming/outgoing JSON
from typing import List, Dict #Type hinting for better code readability


app = FastAPI()

#--------------------------------------------------------------   
# Input models: define what kind of data each endpoint expects
#--------------------------------------------------------------   
# fire prediction model:
# This model defines the weather input expected by the fire prediction model.

class WeatherData(BaseModel):
    temperature: float
    wind_speed: float
    humidity: float

#    
class FireRiskInput(BaseModel):
    region: str
    weather: WeatherData
    
# fire spread model:
# This model defines the input expected by the fire spread model.    
class SpreadInput(BaseModel):
    region: str
    risk_score: float
    wind_direction: str
    
# resource model:
# This model defines the input expected by the resource model.    
class ResourceInput(BaseModel):
    region: str
    spread_area_km2: float
    direction: str
    
    
class RecommendationInput(BaseModel):
    region: str
    risk_score: float
    spread: dict
    resources: dict
                    
                
#--------------------------------------------------------------                
# (Mocked Agents): API Functions for each agent -  Endpoints 
# using FastAPI to expose them as web endpoints
#--------------------------------------------------------------  

@app.post("/predict_fire_risk")
def predict_fire_risk(data: FireRiskInput):
    # MOCK: Replace this with actual model call
    return {"risk_score": 0.87} # we will replace 0.87 with the actual output from the XGBoost model.

@app.post("/predict_spread")
def predict_spread(data: SpreadInput):
    # MOCK: Replace with FireFore model
    return {
        "area_km2": 42.5,
        "direction": data.wind_direction,
        "map_path": "mock_spread_map.png"
    }

@app.post("/optimize_resources")
def optimize_resources(data: ResourceInput):
    # MOCK: Replace with resource planning logic
    return {
        "teams": ["Team A", "Team B"], # mocked plan for deployment
        "aircraft": 1,
        "deployment_zone": data.region
    }
           
           
#replace this function now 
#@app.post("/generate_recommendation")
#def generate_recommendation(data: RecommendationInput):
    # Simulated recommendation (mocked logic for now)
    text = (
        f"⚠️ Fire risk in {data.region} is {data.risk_score:.2f}. "
        f"Fire may spread {data.spread['direction']} across {data.spread['area_km2']} km². "
        f"Deploy {', '.join(data.resources['teams'])} and {data.resources['aircraft']} aircraft "
        f"to the {data.resources['deployment_zone']} region."
    )
    return {"recommendation": text}                

@app.post("/generate_recommendation")
def generate_recommendation(data: RecommendationInput):
    prompt = (
        f"Region: {data.region}\n"
        f"Risk Score: {data.risk_score:.2f}\n"
        f"Spread Direction: {data.spread['direction']}, Area: {data.spread['area_km2']} km²\n"
        f"Resources: Teams - {', '.join(data.resources['teams'])}, Aircraft: {data.resources['aircraft']}\n\n"
        "Based on this data, give a short and clear firefighting deployment recommendation for emergency response teams."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-v3-base:free",
        "messages": [
    {
        "role": "system",
        "content": "You are an expert firefighting assistant. Based on fire risk data, provide clear and actionable deployment recommendations."
    },
    {
        "role": "user",
        "content": prompt  # keep your prompt string here
    }
]
,
        "temperature": 0.7
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    output = response.json()
    print("Full DeepSeek API Response:", output)

    if "choices" in output:
        return {"recommendation": output["choices"][0]["message"]["content"]}
    else:
        return {"error": output.get("error", "Unknown error")}
