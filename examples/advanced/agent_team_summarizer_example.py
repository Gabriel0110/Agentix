#!/usr/bin/env python
"""
agent_team_summarizer_example.py

This example demonstrates the use of AgentTeam with a summarizer pattern, where:
1. Three specialized domain agents work on a problem in parallel
2. Their outputs are collected and sent to a summarizer agent
3. The summarizer agent creates a cohesive final response

This pattern is useful for complex queries that benefit from multiple perspectives,
ensuring the user gets a comprehensive, well-structured answer that incorporates
insights from all domain experts.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AgentTeam, TeamHooks
from agentix.memory import ShortTermMemory
from agentix.llms import OpenAIChat

from dotenv import load_dotenv
load_dotenv()

async def main():
    """
    Main function demonstrating an agent team with summarizer pattern.
    
    Args:
        debug: Whether to enable debug logging
    """
    # Create shared model for all agents
    model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Create three specialized domain agents
    finance_agent = Agent.create(
        name="FinanceExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a financial expert specialized in investments, economics, and business strategy.",
            "Provide detailed financial analysis and perspective on topics.",
            "Focus on economic implications, market trends, and financial considerations."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    tech_agent = Agent.create(
        name="TechnologyExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a technology expert with deep knowledge of computers, AI, and digital innovation.",
            "Analyze topics from a technological perspective.",
            "Consider current technology trends, technical feasibility, and future tech developments."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    society_agent = Agent.create(
        name="SocialImpactExpert",
        model=model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are an expert in sociology and social impact analysis.",
            "Focus on how topics affect different populations, social structures, and human behavior.",
            "Consider ethical implications, social trends, and cultural factors."
        ],
        options=AgentOptions(use_reflection=False, max_steps=1)
    )
    
    # Create a summarizer agent with special instructions for synthesis
    summarizer_agent = Agent.create(
        name="SynthesisExpert",
        model=model,
        memory=ShortTermMemory(max_messages=10),
        instructions=[
            "You are a synthesis expert who combines multiple perspectives into cohesive summaries.",
            "When presented with different expert opinions on a topic, your job is to:",
            "1. Identify key insights from each expert's contribution",
            "2. Find areas of agreement and disagreement",
            "3. Synthesize a comprehensive, balanced viewpoint that incorporates all perspectives",
            "4. Structure your response clearly with sections for each major point",
            "5. Conclude with the most important takeaways",
            "Note when experts disagree and explain the different viewpoints fairly."
        ],
        options=AgentOptions(use_reflection=True, max_steps=2)
    )
    
    # Create an agent team with the domain experts
    team = AgentTeam("DomainExpertTeam", [finance_agent, tech_agent, society_agent])
    
    # Create hooks for monitoring team execution
    hooks = TeamHooks(
        on_agent_start=lambda agent_name, query: print(f"\n🚀 Starting {agent_name} with query: '{query}'"),
        on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} completed"),
        on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
        on_final=lambda results: print(f"🏁 All agents completed their analysis")
    )
    
    async def analyze_with_team_and_summarize(query: str) -> str:
        """
        Run the team of experts in parallel and then summarize their outputs.
        
        Args:
            query: The user query to analyze
            
        Returns:
            A synthesized response combining all expert perspectives
        """
        print(f"\n📝 Team analyzing: '{query}'")
        
        # Step 1: Run all domain experts in parallel
        expert_results = await team.run_in_parallel(query, hooks)
        
        # Step 2: Format results with expert names for the summarizer
        formatted_expert_input = ""
        for i, result in enumerate(expert_results):
            agent = team.agents[i]
            formatted_expert_input += f"--- {agent.name} Analysis ---\n{result}\n\n"
        
        # Step 3: Create a summarization request for the synthesis agent
        summarization_request = f"""
        The following are expert analyses on this topic: "{query}"
        
        {formatted_expert_input}
        
        Please synthesize these perspectives into a comprehensive response that 
        incorporates insights from all experts. Identify key points of agreement 
        and disagreement, and structure your response clearly.
        """
        
        print("\n🔄 Generating synthesis of expert perspectives...")
        
        # Step 4: Have the summarizer agent create the final synthesis
        final_summary = await summarizer_agent.run(summarization_request)
        
        return final_summary
    
    # Example queries that benefit from multiple perspectives
    example_queries = [
        "What are the implications of artificial intelligence on the job market?",
        #"How might blockchain technology change the financial industry?",
        #"What are the potential impacts of remote work becoming permanent for many companies?"
    ]
    
    for query in example_queries:
        print("\n" + "=" * 80)
        print(f"USER QUERY: {query}")
        print("=" * 80)
        
        final_answer = await analyze_with_team_and_summarize(query)
        
        print("\n" + "=" * 40 + " FINAL SYNTHESIZED RESPONSE " + "=" * 40)
        print(final_answer)
        print("=" * 100)
        
        # Add a pause between queries for easier reading of output
        if query != example_queries[-1]:
            print("\nMoving to next query in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(main()) 