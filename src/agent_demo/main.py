# src/agent_demo/main.py

from agent_demo.crew import AgentDemoCrew

def run():
    # Instantiate the Crew class
    crew_instance = AgentDemoCrew()
    
    # Run the crew
    crew_instance.crew().kickoff()
