# This is the orchestrator script.

# It calls our agents in the correct order, like this:

# Call → /predict_fire_risk → Get risk_score
# Call → /predict_spread → Use risk_score
# Call → /optimize_resources → Use spread

import requests

# 1. Fire Risk Prediction
weather_input = {
    "region": "Crete",
    "weather": {
        "temperature": 34.2,
        "wind_speed": 12.5,
        "humidity": 23
    }
}

res_risk = requests.post("http://localhost:8001/predict_fire_risk", json=weather_input)
risk_score = res_risk.json()["risk_score"]
print("Fire Risk Score:", risk_score)

# 2. Fire Spread Prediction
spread_input = {
    "region": "Crete",
    "risk_score": risk_score,
    "wind_direction": "NE"
}

res_spread = requests.post("http://localhost:8001/predict_spread", json=spread_input)
spread_data = res_spread.json()
print("Fire Spread:", spread_data)

# 3. Resource Optimization
resource_input = {
    "region": "Crete",
    "spread_area_km2": spread_data["area_km2"],
    "direction": spread_data["direction"]
}

res_resource = requests.post("http://localhost:8001/optimize_resources", json=resource_input)
plan = res_resource.json()
print("Deployment Plan:", plan)


# 4. LLM Recommendation Agent
recommendation_input = {
    "region": "Crete",
    "risk_score": risk_score,
    "spread": spread_data,
    "resources": plan
}

res_recommendation = requests.post("http://localhost:8001/generate_recommendation", json=recommendation_input)
llm_output = res_recommendation.json()
print("\n LLM Recommendation:\n", llm_output["recommendation"])