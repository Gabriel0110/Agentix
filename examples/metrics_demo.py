"""
Demo script showing how to use the Agentix metrics system.

This example demonstrates:
1. Basic metrics tracking
2. Custom metrics
3. Automatic timing via decorators
4. Team metrics
5. Research metrics
6. Metrics reporting
"""

import asyncio
import time
from agentix.metrics import (
    BaseWorkflowMetrics,
    AgentTeamMetrics,
    ResearchWorkflowMetrics,
    track_metrics)

class SimpleWorkflow:
    """A basic workflow to demonstrate metrics tracking."""
    
    def __init__(self):
        self.metrics = BaseWorkflowMetrics()
    
    @track_metrics("total_queries")
    async def process_query(self, query: str) -> str:
        """Process a query and track metrics."""
        # Simulate some work
        await asyncio.sleep(0.5)
        self.metrics.increment("total_llm_calls")
        return f"Processed: {query}"
    
    @track_metrics("total_tool_calls")
    def run_tool(self, tool_name: str) -> str:
        """Run a tool and track metrics."""
        # Simulate tool execution
        time.sleep(0.3)
        return f"{tool_name} executed"

class ResearchDemo:
    """Demonstrates research-specific metrics."""
    
    def __init__(self):
        self.metrics = ResearchWorkflowMetrics()
    
    @track_metrics("search_queries")
    async def search_sources(self, topic: str) -> list:
        """Simulate searching for sources."""
        await asyncio.sleep(0.4)
        self.metrics.sources_found += 2
        return ["source1", "source2"]

class TeamDemo:
    """Demonstrates team metrics."""
    
    def __init__(self):
        self.metrics = AgentTeamMetrics(team_size=3)
    
    def simulate_team_work(self):
        """Simulate team contributions."""
        agents = ["Agent1", "Agent2", "Agent3"]
        for agent in agents:
            self.metrics.record_agent_contribution(agent)
            self.metrics.record_agent_contribution(agent)
        self.metrics.parallel_executions += 1

async def main():
    print("🚀 Starting Metrics Demo")
    print("=" * 50)
    
    # 1. Basic Workflow Demo
    print("\n📊 Basic Workflow Metrics")
    print("-" * 30)
    
    workflow = SimpleWorkflow()
    
    # Process some queries
    for i in range(3):
        result = await workflow.process_query(f"Query {i}")
        print(f"Query result: {result}")
    
    # Run some tools
    for i in range(2):
        result = workflow.run_tool(f"Tool {i}")
        print(f"Tool result: {result}")
    
    # Add a custom metric
    workflow.metrics.add_custom_metric("quality_score", 0.95)
    
    print("\nBasic Workflow Metrics Report:")
    print(workflow.metrics.report())
    
    # 2. Research Workflow Demo
    print("\n📚 Research Workflow Metrics")
    print("-" * 30)
    
    research = ResearchDemo()
    
    # Simulate some research
    sources = await research.search_sources("AI")
    research.metrics.facts_extracted = 10
    research.metrics.validation_runs = 2
    
    print(f"\nSource efficiency: {research.metrics.calculate_source_efficiency():.2f}")
    print("\nResearch Metrics Report:")
    print(research.metrics.report())
    
    # 3. Team Workflow Demo
    print("\n👥 Team Workflow Metrics")
    print("-" * 30)
    
    team = TeamDemo()
    team.simulate_team_work()
    
    print("\nAgent Contributions:")
    for agent, percentage in team.metrics.get_contribution_distribution().items():
        print(f"{agent}: {percentage:.1%}")
    
    print("\nTeam Metrics Report:")
    print(team.metrics.report())
    
    print("\n✅ Metrics Demo Complete")

if __name__ == "__main__":
    asyncio.run(main()) 