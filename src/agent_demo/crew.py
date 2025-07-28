from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool, WebsiteSearchTool # Import WebsiteSearchTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

@CrewBase
class AgentDemoCrew:
    """YouTube Discovery Crew - Finding underrated AI coding creators"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def reddit_searcher(self) -> Agent: # First Agent
        return Agent(
            config=self.agents_config['reddit_searcher'],  # from agents.yaml
            verbose=True,
            tools=[SerperDevTool()], # Uses SerperDevTool for web search
            max_iter=3,  # Allow iterations for thorough searching
            memory=True
        )

    @agent
    def content_extractor(self) -> Agent: # Second Agent
        return Agent(
            config=self.agents_config['content_extractor'],
            verbose=True,
            tools=[WebsiteSearchTool()], # Uses WebsiteSearchTool for browsing
            max_iter=5, # More iterations for browsing multiple pages and extracting
            memory=True
        )

    @agent
    def reporting_analyst(self) -> Agent: # Third Agent
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            memory=True
        )

    @task
    def reddit_search_task(self) -> Task: # First Task
        return Task(
            config=self.tasks_config['reddit_search_task'],
            agent=self.reddit_searcher()
        )

    @task
    def extraction_task(self) -> Task: # Second Task
        return Task(
            config=self.tasks_config['extraction_task'],
            agent=self.content_extractor(),
            # The output of reddit_search_task feeds into extraction_task
            context=[self.reddit_search_task()]
        )

    @task
    def reporting_task(self) -> Task: # Third Task
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        return Task(
            config=self.tasks_config['reporting_task'],
            agent=self.reporting_analyst(),
            # The output of extraction_task feeds into reporting_task
            context=[self.extraction_task()],
            output_file="output/underrated_ai_youtubers.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.reddit_searcher(),
                self.content_extractor(),
                self.reporting_analyst()
            ],
            tasks=[
                self.reddit_search_task(),
                self.extraction_task(),
                self.reporting_task()
            ],
            process=Process.sequential, # Sequential processing is crucial for this workflow
            verbose=True
        )