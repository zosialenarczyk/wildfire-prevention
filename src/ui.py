# Load libraries
import streamlit as st
import datetime
from agent_system import create_wildfire_crew

# Region coordinates
region_coordinates = {
    "Aegean": {"lat": 37.7, "lon": 25.2},
    "Crete": {"lat": 35.2, "lon": 24.9},
    "Peloponnese, Western Greece and Ionian": {"lat": 37.5, "lon": 21.7},
    "Macedonia and Thrace": {"lat": 41.0, "lon": 24.0},
    "Attica": {"lat": 37.9, "lon": 23.7},
    "Thessaly and Central Greece": {"lat": 39.5, "lon": 22.0},
    "Epirus and Western Macedonia": {"lat": 40.0, "lon": 20.7}
}

# Dummy weather database
dummy_weather_data = {
    "Aegean": {
        "2025-04-28": {"temperature": 34, "wind_speed": 25, "humidity": 35},
        "2025-04-29": {"temperature": 30, "wind_speed": 20, "humidity": 45},
    },
    "Crete": {
        "2025-04-28": {"temperature": 32, "wind_speed": 15, "humidity": 40},
        "2025-04-29": {"temperature": 31, "wind_speed": 10, "humidity": 50},
    },
    "Peloponnese, Western Greece and Ionian": {
        "2025-04-28": {"temperature": 36, "wind_speed": 30, "humidity": 20},
        "2025-04-29": {"temperature": 33, "wind_speed": 25, "humidity": 30},
    },
    "Macedonia and Thrace": {
        "2025-04-28": {"temperature": 28, "wind_speed": 20, "humidity": 50},
        "2025-04-29": {"temperature": 27, "wind_speed": 15, "humidity": 55},
    },
    "Attica": {
        "2025-04-28": {"temperature": 35, "wind_speed": 20, "humidity": 25},
        "2025-04-29": {"temperature": 34, "wind_speed": 18, "humidity": 30},
    },
    "Thessaly and Central Greece": {
        "2025-04-28": {"temperature": 30, "wind_speed": 15, "humidity": 45},
        "2025-04-29": {"temperature": 29, "wind_speed": 10, "humidity": 50},
    },
    "Epirus and Western Macedonia": {
        "2025-04-28": {"temperature": 26, "wind_speed": 10, "humidity": 55},
        "2025-04-29": {"temperature": 25, "wind_speed": 8, "humidity": 60},
    }
}

# Helper functions
def fetch_weather(region, date):
    return dummy_weather_data.get(region, {}).get(date, None)

def predict_fire_risk(weather):
    temperature = weather["temperature"]
    wind_speed = weather["wind_speed"]
    humidity = weather["humidity"]
    fire_probability = min(100, (temperature * 1.5) + (wind_speed * 2) - (humidity * 1.2))
    fire_severity = min(100, (temperature * 1.2) + (wind_speed * 1.5) - (humidity * 1))
    return fire_probability, fire_severity

# --- Streamlit App ---
st.set_page_config(page_title="Wildfire Safety Alert", page_icon="🔥")
st.title("Wildfire Safety Alert System - Greece 🇬🇷")
st.image("firefighting_greece.jpeg", caption="Greek Firefighting Efforts", width=600)
st.write("Select your region and date to assess wildfire risk and receive public safety alerts.")

region = st.selectbox("Select Region", list(region_coordinates.keys()))
today = datetime.date.today()
date = st.date_input("Select Date", max_value=today, value=today)

# Show map
if region in region_coordinates:
    location = region_coordinates[region]
    st.map(data={"lat": [location["lat"]], "lon": [location["lon"]]}, zoom=6)

# Main logic
if st.button("Check Fire Risk"):
    weather = fetch_weather(region, str(date))
    if weather:
        st.write(f"🌡️ Temperature: {weather['temperature']}°C")
        st.write(f"💨 Wind Speed: {weather['wind_speed']} km/h")
        st.write(f"💧 Humidity: {weather['humidity']}%")

        fire_probability, fire_severity = predict_fire_risk(weather)
        st.write(f"🔥 Predicted Fire Probability: {fire_probability:.1f}%")
        st.write(f"🔥 Predicted Fire Severity: {fire_severity:.1f}/100")

        st.write("Running CrewAI Agents... 🚒")

        # Run agent system
        crew = create_wildfire_crew(region, fire_probability, fire_severity)
        result = crew.kickoff() 

        # Tone based on risk
        if fire_probability >= 70 or fire_severity >= 60:
            alert_intro = """
🚨 **In light of the current fire danger, immediate action is required**
to protect the safety of residents and visitors. Fire probability is significant,
and firefighting resources will be deployed immediately.
"""
        elif fire_probability >= 40 or fire_severity >= 30:
            alert_intro = """
⚠️ **There is an elevated risk of wildfire activity.**
Residents and tourists are advised to stay cautious, monitor local alerts,
and prepare for possible evacuation if conditions worsen.
"""
        else:
            alert_intro = """
✅ **Fire probability and severity are currently low.**
No immediate action required, but remain vigilant and avoid risky activities in nature.
"""

        # Final Output
        st.success("✅ Public Safety Alert Ready!")
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/47/Fire_icon.svg", width=120)
        st.markdown("## 🚨 PUBLIC SAFETY ALERT")
        st.markdown(alert_intro)
        st.markdown("---")
        st.markdown("### Summary")
        st.markdown(result)
        st.markdown("""
---
### Stay Safe Recommendations 🏕️
- Follow evacuation instructions carefully if issued.
- Avoid activities that could start fires (e.g., grilling, smoking).
- Monitor local emergency broadcasts.
- Report any fires immediately (Call 112).

⚡ _Alert generated by the Wildfire Management AI System_
""")
    else:
        st.error("❌ No weather data available for the selected region and date.")
