#!/usr/bin/env python
"""
agent_team_advanced_example.py

Demonstrates advanced synergy:
1) Shared memory among multiple agents
2) Interleaved chat approach for multi-agent analysis
   - Round-robin the agents in a while loop until a convergence condition is met
3) Aggregator agent merges everything into a final answer

Note: This example showcases very advanced usage. Not everything is necessary, but
it demonstrates the full capabilities of the AdvancedAgentTeam class. Pick and choose
the features that are most relevant to your use case.
"""
import os
import asyncio

from agentix.agents import Agent, AgentOptions
from agentix.agents.multi_agent import AdvancedAgentTeam, TeamConfiguration, AgentRole, AdvancedTeamHooks
from agentix.llms import OpenAIChat
from agentix.memory import ShortTermMemory, CompositeMemory

from dotenv import load_dotenv
load_dotenv()

# Different types of convergence checks
def convergence_checks():
    """
    Return different types of convergence check functions.
    """
    return {
        "final_answer": lambda msg: "FINAL ANSWER:" in msg,
        
        "keywords": lambda msg: (
            "conclusion" in msg.lower() or 
            "summary" in msg.lower()
        ),
        
        "comprehensive": lambda msg: all(
            term in msg.lower() for term in ["analysis", "implications", "recommendation"]
        ),
        
        "length": lambda msg: (
            len(msg) > 200 and "conclusion" in msg.lower()
        )
    }


async def main():
    """Main function demonstrating advanced agent team capabilities."""
    # OPTIONAL: Example usage of advanced team hooks for logging
    team_hooks = AdvancedTeamHooks(
        on_agent_start=lambda name, input_str: print(f"\n🤖 {name} starting analysis..."),
        on_agent_end=lambda name, output: print(f"✅ {name} completed analysis"),
        on_round_start=lambda round_num, max_rounds: print(f"\n📍 Starting round {round_num}/{max_rounds}"),
        on_round_end=lambda round_num, contributions: print(f"📍 Round {round_num} complete - {len(contributions)} contributions"),
        on_convergence=lambda agent, content: print(f"🎯 Convergence reached by {agent.name}"),
        on_aggregation=lambda result: print(f"\n🔄 Aggregating team analysis..."),
        on_error=lambda name, error: print(f"❌ Error from {name}: {error}")
    )
    
    # Initialize LLMs
    model1 = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    model2 = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    # Create memories
    agent1_mem = ShortTermMemory(max_messages=5)
    agent2_mem = ShortTermMemory(max_messages=5)
    shared_memory = CompositeMemory(agent1_mem, agent2_mem)
    
    # Create agents for technical analysis
    technical_agent = Agent.create(
        name="TechnicalAgent",
        model=model1,
        memory=agent1_mem,
        instructions=[
            "You are a technical analyst focusing on implementation details.",
            "Provide detailed technical analysis and specific recommendations.",
            "Always consider technical feasibility and best practices.",
            "Conclude with clear technical recommendations."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True,
            usage_limit=10
        )
    )
    
    security_agent = Agent.create(
        name="SecurityAgent",
        model=model2,
        memory=agent2_mem,
        instructions=[
            "You are a security analyst focusing on security implications.",
            "Provide detailed security analysis and specific recommendations.",
            "Always consider security best practices and risk mitigation.",
            "Conclude with clear security recommendations."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True,
            usage_limit=10
        )
    )
    
    # Create technical team configuration
    technical_team_config = TeamConfiguration(
        roles={
            "TechnicalAgent": AgentRole(
                name="Technical Analyst",
                description="Focuses on technical implementation details",
                query_transform=lambda query: f"{query}\nAnalyze from a technical perspective..."
            ),
            "SecurityAgent": AgentRole(
                name="Security Analyst",
                description="Focuses on security implications",
                query_transform=lambda query: f"{query}\nAnalyze from a security perspective..."
            )
        },
        default_role=AgentRole(
            name="General Analyst",
            description="Provides general analysis",
            query_transform=lambda query: query
        )
    )
    
    # Create technical analysis team
    technical_team = AdvancedAgentTeam(
        "TechnicalTeam",
        [technical_agent, security_agent],
        {
            "shared_memory": shared_memory,
            "team_config": technical_team_config,
            "hooks": team_hooks,
            "debug": True
        }
    )
    
    # Enable shared memory
    technical_team.enable_shared_memory()
    
    # Define convergence checks
    checks = convergence_checks()
    
    # Test different scenarios
    print("\n=== Technical Analysis with Final Answer Convergence ===")
    technical_query = """
    Analyze the implementation of a new cloud-based microservices architecture.
    Consider both technical implementation and security aspects.
    Provide specific recommendations.
    """
    
    technical_result = await technical_team.run_interleaved(
        technical_query,
        5,
        checks["final_answer"],
        True  # require all agents
    )
    
    print("\nTechnical Analysis Result:\n", technical_result)
    
    # Create business analysis agents
    market_agent = Agent.create(
        name="MarketAgent",
        model=model1,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a market analyst focusing on market implications.",
            "Provide detailed market analysis and specific recommendations.",
            "Always consider market trends and competitive advantages.",
            "Conclude with clear market-focused recommendations."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    finance_agent = Agent.create(
        name="FinanceAgent",
        model=model2,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are a financial analyst focusing on financial implications.",
            "Provide detailed financial analysis and specific recommendations.",
            "Always consider ROI and financial risks.",
            "Conclude with clear financial recommendations."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    # Create business team configuration
    business_team_config = TeamConfiguration(
        roles={
            "MarketAgent": AgentRole(
                name="Market Analyst",
                description="Analyzes market implications",
                query_transform=lambda query: f"{query}\nAnalyze from a market perspective..."
            ),
            "FinanceAgent": AgentRole(
                name="Financial Analyst",
                description="Analyzes financial implications",
                query_transform=lambda query: f"{query}\nAnalyze from a financial perspective..."
            )
        },
        default_role=AgentRole(
            name="General Analyst",
            description="Provides general analysis",
            query_transform=lambda query: query
        )
    )
    
    # Create business analysis team
    business_team = AdvancedAgentTeam(
        "BusinessTeam",
        [market_agent, finance_agent],
        {
            "shared_memory": CompositeMemory(
                ShortTermMemory(max_messages=5),
                ShortTermMemory(max_messages=5)
            ),
            "team_config": business_team_config,
            "hooks": team_hooks,
            "debug": True
        }
    )
    
    print("\n=== Business Analysis with Comprehensive Convergence ===")
    business_query = """
    Analyze the business implications of expanding into the Asian market.
    Consider both market opportunities and financial implications.
    Provide specific recommendations.
    """
    
    business_result = await business_team.run_interleaved(
        business_query,
        5,
        checks["final_answer"],
        True  # require all agents
    )
    print("\nBusiness Analysis Result:\n", business_result)
    
    # Test with different convergence criteria
    print("\n=== Quick Analysis with Keyword Convergence ===")
    quick_query = "Provide a quick analysis of current cloud computing trends."
    
    quick_result = await technical_team.run_interleaved(
        quick_query,
        3,
        checks["keywords"],
        False  # don't require all agents
    )
    print("\nQuick Analysis Result:\n", quick_result)
    
    # Test with length-based convergence
    print("\n=== Detailed Analysis with Length-based Convergence ===")
    detailed_query = """
    Provide a detailed analysis of blockchain technology adoption in enterprise.
    Include technical, security, market, and financial perspectives.
    """
    
    detailed_result = await business_team.run_interleaved(
        detailed_query,
        4,
        checks["length"],
        True  # require all agents
    )
    print("\nDetailed Analysis Result:\n", detailed_result)
    
    # Aggregator agent merges final answers succinctly
    aggregator_model = OpenAIChat(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    aggregator_agent = Agent.create(
        name="Aggregator",
        model=aggregator_model,
        memory=ShortTermMemory(max_messages=5),
        instructions=[
            "You are an expert aggregator that combines and synthesizes multiple expert perspectives into a comprehensive analysis.",
            "Your role is to create a unified response that:",
            "1. Synthesizes key insights from all experts while eliminating redundancies",
            "2. Maintains the depth and expertise from each perspective",
            "3. Organizes content into clear sections: Market Analysis, Financial Implications, and Recommendations",
            "4. Preserves unique insights from each expert",
            "5. Provides specific, actionable recommendations",
            "\nFormat your response as follows:",
            "### Executive Summary",
            "(Brief overview of key points)",
            "\n### Combined Analysis",
            "(Main insights organized by topic)",
            "\n### Key Recommendations",
            "(Specific, actionable steps)",
            "\n### Risk Considerations",
            "(Important risks and mitigation strategies)",
            "\nEnsure your response is detailed yet concise, and maintain the depth of expertise while eliminating redundancy.",
            "Start with 'FINAL ANSWER:' and then provide your structured response."
        ],
        options=AgentOptions(
            max_steps=3,
            use_reflection=False,
            debug=True
        )
    )
    
    # Format the input for the aggregator
    def format_aggregator_input(team_response: str) -> str:
        return f"""
    Please synthesize and aggregate the following expert analyses into a comprehensive response.
    Each expert has provided valuable insights that need to be combined effectively.

    {team_response}

    Create a unified analysis that combines these perspectives while:
    1. Maintaining the depth of expertise from both market and financial analyses
    2. Eliminating redundant information
    3. Organizing insights logically
    4. Providing clear, actionable recommendations
    5. Highlighting key risks and opportunities

    Format your response according to the structure specified in your instructions.
    """
    
    print("\n--- Aggregator Step ---")
    formatted_input = format_aggregator_input(quick_result)
    aggregator_final = await aggregator_agent.run(formatted_input)
    print("\nAggregated Analysis Result:\n", aggregator_final)


if __name__ == "__main__":
    asyncio.run(main()) 