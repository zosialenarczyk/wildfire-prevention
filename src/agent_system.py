from crewai import Agent, Task, Crew
import os

# Set OpenAI Key 
os.environ["OPENAI_API_KEY"] = "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"  

# Dictionary of real locations
real_locations = {
    "Attica": ["Penteli", "Nea Makri", "Marathonas", "Pikermi", "Rafina", "Varnavas", "Dionysos"],
    "Crete": ["Chania", "Rethymno", "Heraklion", "Agios Nikolaos", "Ierapetra"],
    "Aegean": ["Mykonos", "Santorini", "Paros", "Naxos", "Syros"],
    "Peloponnese, Western Greece and Ionian": ["Patras", "Pyrgos", "Kalamata", "Tripoli", "Zakynthos", "Kefalonia"],
    "Macedonia and Thrace": ["Kavala", "Xanthi", "Komotini", "Drama", "Alexandroupolis"],
    "Thessaly and Central Greece": ["Larissa", "Volos", "Trikala", "Karditsa", "Lamia"],
    "Epirus and Western Macedonia": ["Ioannina", "Grevena", "Kozani", "Florina", "Konitsa"]
}

# Function to create dynamic evacuation task
def create_evacuate_task(region, fire_probability, fire_severity):
    locations_list = real_locations.get(region, [])
    locations_str = ", ".join(locations_list)

    task_description = (
        f"Given the fire probability of {fire_probability:.1f}% and severity level of {fire_severity:.1f}/100, "
        f"identify which areas should be evacuated in {region}. "
        f"Only choose from the following known locations: {locations_str}. "
        "Prioritize villages or towns with higher exposure to forests, dry grasslands, or strong winds."
        "Collaborate with the Resource Planner to prioritize evacuation zones based on resource availability and fire severity. "
        "Prioritize villages or towns with higher exposure to forests, dry grasslands, or strong winds."
    
    )

    return task_description

# Main function to create Crew
def create_wildfire_crew(region, fire_probability, fire_severity, trucks=3, helicopters=2):
    # Agents
    resource_planning_agent = Agent(
        role='Resource Planner',
        goal='Determine optimal allocation of firefighting resources based on fire probability and available units.',
        backstory='Expert in emergency logistics and firefighting operations.',
        allow_delegation=True,
    )

    evacuation_agent = Agent(
        role='Evacuation Coordinator',
        goal='Identify evacuation zones based on fire probability and spread severity.',
        backstory='Crisis management specialist focusing on human safety.',
        allow_delegation=True,
    )

    tourist_agent = Agent(
        role='Tourism Safety Advisor',
        goal='Evaluate safety of regions for tourists based on fire risk and spread.',
        backstory='Advisor for tourist safety and travel regulations during emergencies.',
        allow_delegation=True,
    )

    public_communication_agent = Agent(
        role='Public Communication Manager',
        goal='Aggregate information from all agents to generate public alerts and recommendations.',
        backstory='Public communication and crisis information specialist.',
        allow_delegation=True,
    )

    # Tasks
    resource_planning_task = Task(
        description=(
            f"Fire probability is estimated at {fire_probability:.1f}%. "
            f"Fire severity is measured at {fire_severity:.1f}/100. "
            f"Available resources include {trucks} firefighting trucks and {helicopters} helicopters. "
            "Work together with the Evacuation Coordinator to plan optimal firefighting resource deployment based on evacuation priorities."
        ),
        expected_output='Resource deployment plan (locations and units)',
        agent=resource_planning_agent
    )

    evacuation_task = Task(
        description=create_evacuate_task(region, fire_probability, fire_severity),
        expected_output='List of prioritized evacuation areas with justification.',
        agent=evacuation_agent
    )

    tourist_task = Task(
        description=(
            f"Based on fire probability {fire_probability:.1f}% and severity {fire_severity:.1f}/100, "
            "assess and recommend tourist safety for the selected region. Coordinate with the Public Communication Manager to ensure consistent advice messaging."
        ),
        expected_output='Tourist safety advisory',
        agent=tourist_agent
    )

    public_communication_task = Task(
        description=(
            "Aggregate outputs from the Resource Planner, Evacuation Coordinator, and Tourism Safety Advisor. "
            "If necessary, request clarifications to ensure accurate and actionable public communication. "
            "Generate a clear public safety alert for residents and tourists."
        ),
        expected_output='Formatted public alert message for public display.',
        agent=public_communication_agent
    )

    # Create Crew
    wildfire_crew = Crew(
        agents=[
            resource_planning_agent,
            evacuation_agent,
            tourist_agent,
            public_communication_agent
        ],
        tasks=[
            resource_planning_task,
            evacuation_task,
            tourist_task,
            public_communication_task
        ],
        verbose=True,
        llm_model="gpt-4.1-nano"
    )

    return wildfire_crew
