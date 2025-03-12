#!/usr/bin/env python
"""
startup_research_workflow.py

An advanced production-grade workflow that demonstrates how to build a complex
multi-agent system for startup research and analysis. This example integrates:

1. News Collection Agent: Gathers recent startup news using DuckDuckGo
2. Summarization Agent: Creates concise summaries of the gathered information
3. Trend Analysis Agent: Identifies patterns in startup funding and technology
4. Validation and quality control using task validators
5. Multi-agent coordination via advanced agent team and router

The workflow demonstrates these Agentix features:
- Advanced Agent Team for collaboration
- LLM Convergence Checking for quality assurance
- Task Validation for output verification
- Reflection Memory for agent chain-of-thought
- Composite Memory with Long Term and Summarizing Memory
- Advanced Agent Router for dynamic task assignment
- Planner for workflow orchestration
- Hooks for monitoring and debugging
- Evaluator for output quality assessment

This production-grade workflow handles real-world startup research tasks
and can be extended for additional research domains.
"""
import os
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Agentix Core Components
from agentix.agents import Agent, AgentOptions, AgentHooks
from agentix.planner import SimpleLLMPlanner
from agentix.evaluators import SimpleEvaluator, EvaluationResult

# Agentix Multi-agent Components
from agentix.agents.multi_agent import (
    AdvancedAgentTeam,
    AdvancedTeamOptions,
    AdvancedTeamHooks,
    AgentRole,
    TeamConfiguration,
    LLMConvergenceChecker,
    AdvancedAgentRouter,
    RouterOptions,
    AgentCapability
)

from agentix.agents.multi_agent.llm_convergence_checker import ConvergenceCriteria

# Agentix Memory Components
from agentix.memory import (
    ShortTermMemory,
    LongTermMemory,
    ReflectionMemory,
    SummarizingMemory,
    CompositeMemory
)

# Agentix LLM and Tools
from agentix.llms import OpenAIChat, OpenAIEmbeddings
from agentix.tools import (
    DuckDuckGoToolkit,
    DuckDuckGoNewsSearchTool,
    DuckDuckGoTextSearchTool,
)

from dotenv import load_dotenv
load_dotenv()


@dataclass
class WorkflowMetrics:
    """Tracks performance metrics for the workflow."""
    query_count: int = 0
    agent_calls: int = 0
    search_queries: int = 0
    validation_runs: int = 0
    total_processing_time: float = 0.0
    convergence_attempts: int = 0
    successful_convergence: int = 0
    evaluation_scores: List[float] = None

    def __post_init__(self):
        if self.evaluation_scores is None:
            self.evaluation_scores = []

    def add_evaluation_score(self, score: float):
        self.evaluation_scores.append(score)
        
    def get_average_evaluation_score(self) -> float:
        return sum(self.evaluation_scores) / len(self.evaluation_scores) if self.evaluation_scores else 0.0
    
    def report(self) -> str:
        """Generate a summary report of metrics."""
        return f"""
Workflow Performance Metrics:
----------------------------
Queries Processed: {self.query_count}
Agent Calls: {self.agent_calls}
Search Queries: {self.search_queries}
Validation Runs: {self.validation_runs}
Total Processing Time: {self.total_processing_time:.2f}s
Convergence Rate: {self.successful_convergence}/{self.convergence_attempts} ({(self.successful_convergence/self.convergence_attempts)*100 if self.convergence_attempts else 0:.1f}%)
Average Evaluation Score: {self.get_average_evaluation_score():.2f}/1.0
"""


class StartupResearchWorkflow:
    """
    A comprehensive workflow for startup research and analysis.
    Demonstrates a production-grade setup using multiple integrated agents.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the startup research workflow with all required components.
        
        Args:
            debug: Whether to enable debug logging
        """
        self.debug = debug
        self.metrics = WorkflowMetrics()
        
        # Initialize LLM models with different temperature settings for specialized tasks
        self.main_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.7  # Higher temperature for creative analysis
        )
        
        self.validator_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.1  # Lower temperature for validation
        )
        
        self.router_model = OpenAIChat(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0.2  # Balanced temperature for routing decisions
        )
        
        # Initialize embeddings for long-term memory
        self.embeddings = OpenAIEmbeddings(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        
        # Initialize search tools
        self.search_tools = self._create_search_tools()
        
        # Initialize memories
        self.memories = self._create_memories()
        
        # Initialize specialized agents
        self.agents = self._create_agents()
        
        # Initialize router
        self.router = self._create_router()
        
        # Initialize team
        self.team = self._create_team()
        
        # Initialize planner
        self.planner = SimpleLLMPlanner(self.main_model)
        
        # Initialize evaluator
        self.evaluator = SimpleEvaluator(self.validator_model)
        
        # Initialize convergence checker
        self.convergence_checker = self._create_convergence_checker()
        
        print("✅ Startup Research Workflow initialized successfully")
    
    def _create_search_tools(self) -> Dict[str, Any]:
        """Initialize search tools for the agents."""
        print("🔧 Initializing search tools...")
        
        try:
            # Create DuckDuckGo toolkit with specific tools enabled
            ddg_toolkit = DuckDuckGoToolkit(
                enable_text_search=True,
                enable_news_search=True,
                enable_image_search=False,
                enable_video_search=False,
                enable_chat=False,
                timeout=20  # Increased timeout for more reliable results
            )
            
            # Extract individual tools from the toolkit
            news_search = next((t for t in ddg_toolkit.get_tools() 
                              if isinstance(t, DuckDuckGoNewsSearchTool)), None)
            
            text_search = next((t for t in ddg_toolkit.get_tools() 
                              if isinstance(t, DuckDuckGoTextSearchTool)), None)
            
            # Check if we have the tools we need
            if not news_search or not text_search:
                print("⚠️ Warning: Could not find required DuckDuckGo search tools. Using empty toolkit.")
                return {
                    "toolkit": None,
                    "news_search": None,
                    "text_search": None,
                    "all_tools": []
                }
            
            # Print success message
            print("✅ DuckDuckGo search tools initialized successfully")
            
            return {
                "toolkit": ddg_toolkit,
                "news_search": news_search,
                "text_search": text_search,
                "all_tools": ddg_toolkit.get_tools()
            }
        except Exception as e:
            print(f"⚠️ Error initializing DuckDuckGo tools: {e}")
            print("⚠️ Continuing without search tools. Please ensure duckduckgo-search is installed.")
            print("⚠️ Run: pip install duckduckgo-search")
            return {
                "toolkit": None,
                "news_search": None,
                "text_search": None,
                "all_tools": []
            }
    
    def _create_memories(self) -> Dict[str, Any]:
        """Initialize memory systems for the agents."""
        print("🧠 Initializing memory systems...")
        
        # Create short-term memories for each agent
        news_mem = ShortTermMemory(max_messages=10)
        summary_mem = ShortTermMemory(max_messages=15)
        trends_mem = ShortTermMemory(max_messages=15)
        
        # Create reflection memories for each agent
        news_reflection = ReflectionMemory(include_reflections=False)
        summary_reflection = ReflectionMemory(include_reflections=False)
        trends_reflection = ReflectionMemory(include_reflections=False)
        
        # Create long-term memory for persistent storage
        long_term = LongTermMemory(self.embeddings, max_messages=1000, top_k=5)
        
        # Create summarizing memory for past research
        summarizing = SummarizingMemory(
            summarizer_model=self.validator_model,
            threshold=15,
            max_summary_tokens=200,
            hierarchical=True
        )
        
        # Create composite memories combining different memory types
        news_composite = CompositeMemory(news_mem, news_reflection, long_term)
        summary_composite = CompositeMemory(summary_mem, summary_reflection, summarizing)
        trends_composite = CompositeMemory(trends_mem, trends_reflection, long_term, summarizing)
        
        # Create shared memory for team collaboration
        shared_memory = CompositeMemory(ShortTermMemory(max_messages=30), long_term)
        
        return {
            "news": news_composite,
            "summary": summary_composite,
            "trends": trends_composite,
            "shared": shared_memory,
            "long_term": long_term,
            "summary_memory": summarizing,
            "reflections": {
                "news": news_reflection,
                "summary": summary_reflection,
                "trends": trends_reflection
            }
        }
    
    async def _track_search_query_async(self, name: str, query: str) -> bool:
        """Asynchronous version of search query tracking that returns awaitable bool."""
        self.metrics.search_queries += 1
        return True
    
    async def _log_reflection_async(self, reflection_memory: ReflectionMemory, content: str) -> bool:
        """
        Asynchronous version of reflection logging that returns awaitable bool.
        
        Args:
            reflection_memory: The reflection memory to log to
            content: The reflection content
            
        Returns:
            True to indicate successful logging
        """
        await reflection_memory.add_message({
            "role": "reflection",
            "content": content,
            "metadata": {"timestamp": int(time.time() * 1000)}
        })
        return True
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Initialize specialized agents for the workflow."""
        print("🤖 Creating specialized agents...")
        
        # Check if we have search tools available
        has_tools = len(self.search_tools["all_tools"]) > 0
        if not has_tools:
            print("⚠️ No search tools available. Agents will continue with reasoning only.")
        
        # 1. Create the News Collection Agent
        news_agent = Agent.create(
            name="NewsCollector",
            model=self.main_model,
            memory=self.memories["news"],
            tools=self.search_tools["all_tools"] if has_tools else [],
            instructions=[
                "You are a startup news research specialist focused on gathering accurate information.",
                "Use DuckDuckGo tools to find recent news, funding announcements, and market analysis." if has_tools else "Analyze startup trends and news based on your knowledge.",
                "Focus on factual information like funding amounts, investors, technologies, and market data.",
                "Include timestamps and sources for each piece of information.",
                "Organize findings into clearly structured categories.",
                "Always verify information across multiple sources when possible.",
            ],
            validation_model=self.validator_model,
            task="Gather comprehensive, verified news about startups in specified sectors",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=10,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_tool_call=self._track_search_query_async if has_tools else None,
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["news"],
                    "Collected startup news data. Need to verify: sources cited? recent enough? comprehensive?"
                )
            )
        )
        
        # 2. Create the Summary Generation Agent
        summary_agent = Agent.create(
            name="Summarizer",
            model=self.main_model,
            memory=self.memories["summary"],
            instructions=[
                "You are a startup data summarization specialist.",
                "Create concise, factual summaries of startup information.",
                "Highlight key metrics like funding amounts, growth rates, team size, and technologies.",
                "Structure summaries with clear sections for financials, technology, team, and market.",
                "Remove redundant information and focus on verifiable facts.",
                "Include confidence ratings for information based on source reliability.",
            ],
            validation_model=self.validator_model,
            task="Create accurate, concise summaries of startup information",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=5,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["summary"],
                    "Created summary of startup data. Focus on clarity, accuracy, and insight density."
                )
            )
        )
        
        # 3. Create the Trend Analysis Agent
        trends_agent = Agent.create(
            name="TrendAnalyst",
            model=self.main_model,
            memory=self.memories["trends"],
            tools=[self.search_tools["text_search"]] if has_tools and self.search_tools["text_search"] else [],
            instructions=[
                "You are a startup trend analysis expert.",
                "Identify emerging patterns in startup funding, technology adoption, and market opportunities.",
                "Look for connections across different startups and sectors.",
                "Analyze trends quantitatively when possible (growth rates, funding increases, etc.).",
                "Compare trends against established market contexts.",
                "Distinguish between temporary hype and substantive movements.",
                "Provide actionable insights based on identified trends.",
            ],
            validation_model=self.validator_model,
            task="Identify meaningful patterns and trends in startup data",
            options=AgentOptions(
                validate_output=True,
                use_reflection=True,
                max_steps=7,
                debug=self.debug
            ),
            hooks=AgentHooks(
                on_tool_call=self._track_search_query_async if has_tools else None,
                on_final_answer=lambda ans: self._log_reflection_async(
                    self.memories["reflections"]["trends"],
                    "Analyzed startup trends. Consider: economic context, technological limitations, market adoption curves."
                )
            )
        )
        
        return {
            "news": news_agent,
            "summary": summary_agent, 
            "trends": trends_agent
        }
    
    def _create_router(self) -> AdvancedAgentRouter:
        """Initialize the advanced agent router."""
        print("🔀 Setting up agent router...")
        
        # Define agent capabilities for the router
        capabilities = {
            0: AgentCapability(
                name="Startup News Collection",
                description="Gathers recent news, funding rounds, and market data about startups",
                keywords=["news", "latest", "funding", "investment", "announced", "launched", "raised"],
                examples=[
                    "Find recent news about AI startups in healthcare",
                    "What are the latest funding rounds for fintech startups?",
                    "Research recent developments in clean energy startups"
                ]
            ),
            1: AgentCapability(
                name="Startup Information Summarization",
                description="Creates concise summaries of startup information and market data",
                keywords=["summarize", "summary", "digest", "brief", "overview", "condensed"],
                examples=[
                    "Summarize the funding landscape for AI startups",
                    "Give me a brief overview of the top climate tech startups",
                    "Provide a condensed summary of recent biotech startup innovations"
                ]
            ),
            2: AgentCapability(
                name="Startup Trend Analysis",
                description="Identifies patterns in startup funding, technology, and market opportunities",
                keywords=["trends", "patterns", "analysis", "emerging", "growing", "opportunities", "future"],
                examples=[
                    "What are the emerging trends in AI startup funding?",
                    "Analyze patterns in startup growth across fintech",
                    "Identify future opportunities in the clean energy startup space"
                ]
            )
        }
        
        # Create router options
        router_options = RouterOptions(
            use_llm=True,
            debug=self.debug,
            fallback_index=2,  # Trend analysis as fallback (most general)
            confidence_threshold=0.65,
            router_llm=self.router_model
        )
        
        # Create the router with the agent list
        return AdvancedAgentRouter(
            agents=[self.agents["news"], self.agents["summary"], self.agents["trends"]],
            capabilities=capabilities,
            options=router_options
        )
    
    def _create_team(self) -> AdvancedAgentTeam:
        """Initialize the advanced agent team for data gathering."""
        print("👥 Building agent team...")
        
        # Define specialized query transformations for each role
        def news_collector_transform(query: str) -> str:
            return f"As a startup news specialist, gather comprehensive information about: {query}\n\nFocus on recent funding announcements, technology developments, market entries, and verified facts. Provide your findings in a clear, structured format."
        
        def trend_analyst_transform(query: str) -> str:
            return f"As a startup trend analyst, identify emerging patterns related to: {query}\n\nFocus on funding trends, technology adoption curves, market opportunities, and future projections. Structure your analysis clearly with sections for different trend categories."
        
        # Create team configuration with specialized roles for data gathering
        team_config = TeamConfiguration(
            roles={
                "NewsCollector": AgentRole(
                    name="News Collector",
                    description="Gathers comprehensive startup news and research",
                    query_transform=news_collector_transform
                ),
                "TrendAnalyst": AgentRole(
                    name="Trend Analyst",
                    description="Identifies patterns and emerging trends in startup data",
                    query_transform=trend_analyst_transform
                )
            },
            default_role=None  # No default role; each agent must have a specific role
        )
        
        # Helper function for on_agent_end to avoid issues with boolean logic in lambda
        async def agent_end_handler(agent_name: str, result: str) -> None:
            print(f"✅ {agent_name} completed research")
            self._increment_agent_calls()
        
        # Define team hooks for monitoring and debugging
        team_hooks = AdvancedTeamHooks(
            on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting research..."),
            on_agent_end=agent_end_handler,
            on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
            on_final=lambda results: print(f"🏁 Data gathering completed with {len(results)} contributions"),
            on_round_start=lambda round_num, max_rounds: print(f"\n📊 Starting research round {round_num}/{max_rounds}"),
            on_round_end=lambda round_num, contributions: print(f"📝 Round {round_num} complete with {len(contributions)} contributions")
        )
        
        # Configure the team options
        team_options = AdvancedTeamOptions(
            shared_memory=self.memories["shared"],
            team_config=team_config,
            hooks=team_hooks,
            debug=self.debug
        )
        
        # Create and configure the team with just the data gathering agents
        team = AdvancedAgentTeam(
            name="StartupResearchTeam",
            agents=[self.agents["news"], self.agents["trends"]],  # Only include data gathering agents
            options=team_options
        )
        
        # Enable shared memory
        team.enable_shared_memory()
        
        return team
    
    def _create_convergence_checker(self) -> LLMConvergenceChecker:
        """Initialize the LLM convergence checker for validating individual agent outputs."""
        print("👁️ Setting up convergence checker...")
        
        # Define convergence criteria for individual agent outputs
        criteria = ConvergenceCriteria(
            required_elements=[
                "factual information",
                "data sources",
                "startup details"
            ],
            required_structure=[
                "clear sections",
                "organized data"
            ],
            minimum_length=300,  # Minimum length in words for a complete response
            custom_instructions=[
                "Ensure the response contains specific, verifiable information",
                "Check that all claims are supported by sources",
                "Verify the response is properly structured and organized"
            ]
        )
        
        # Create the convergence checker
        checker = LLMConvergenceChecker(
            model=self.validator_model,
            criteria=criteria,
            debug=self.debug
        )
        
        return checker
    
    async def process_single_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single startup research query using the appropriate agent.
        
        Args:
            query: User's research query
            
        Returns:
            Dictionary with results and performance metrics
        """
        self.metrics.query_count += 1
        print(f"\n🔍 Processing query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Use the router to select the most appropriate agent
            result = await self.router.run(query)
            
            # Validate the output
            self.metrics.validation_runs += 1
            evaluation = await self.evaluate_result(result)
            
            # Store in long-term memory
            await self.memories["long_term"].add_message({
                "role": "assistant",
                "content": result,
                "metadata": {
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "evaluation_score": evaluation.score
                }
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            return {
                "result": result,
                "evaluation": evaluation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            print(f"❌ Error processing query: {str(e)}")
            return {
                "result": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def process_complex_query(self, query: str, max_rounds: int = 3) -> Dict[str, Any]:
        """
        Process a complex startup research query using a two-phase approach:
        1. Parallel data gathering from news and trend agents
        2. Summarization of gathered data
        
        Args:
            query: User's research query
            max_rounds: Maximum number of rounds (used for convergence in data gathering)
            
        Returns:
            Dictionary with results and performance metrics
        """
        self.metrics.query_count += 1
        print(f"\n🔍 Processing complex query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Phase 1: Parallel Data Gathering
            print("\n📊 Phase 1: Gathering Data (News and Trends)")
            
            # Create specialized query transformations for data gathering
            def news_collector_transform(query: str) -> str:
                return f"As a startup news specialist, gather comprehensive information about: {query}\n\nFocus on recent funding announcements, technology developments, market entries, and verified facts. Provide your findings in a clear, structured format."
            
            def trend_analyst_transform(query: str) -> str:
                return f"As a startup trend analyst, identify emerging patterns related to: {query}\n\nFocus on funding trends, technology adoption curves, market opportunities, and future projections. Structure your analysis clearly with sections for different trend categories."
            
            # Create team configuration for data gathering
            team_config = TeamConfiguration(
                roles={
                    "NewsCollector": AgentRole(
                        name="News Collector",
                        description="Gathers comprehensive startup news and research",
                        query_transform=news_collector_transform
                    ),
                    "TrendAnalyst": AgentRole(
                        name="Trend Analyst",
                        description="Identifies patterns and emerging trends in startup data",
                        query_transform=trend_analyst_transform
                    )
                },
                default_role=None
            )
            
            # Define team hooks for monitoring
            team_hooks = AdvancedTeamHooks(
                on_agent_start=lambda agent_name, query: print(f"\n🚀 {agent_name} starting research..."),
                on_agent_end=lambda agent_name, result: print(f"✅ {agent_name} completed research"),
                on_error=lambda agent_name, error: print(f"❌ Error from {agent_name}: {str(error)}"),
                on_final=lambda results: print(f"🏁 Data gathering completed with {len(results)} contributions")
            )
            
            # Create a data gathering team with just news and trend agents
            data_team = AdvancedAgentTeam(
                name="DataGatheringTeam",
                agents=[self.agents["news"], self.agents["trends"]],
                options=AdvancedTeamOptions(
                    shared_memory=self.memories["shared"],
                    team_config=team_config,
                    hooks=team_hooks,
                    debug=self.debug
                )
            )
            
            # Enable shared memory for the data team
            data_team.enable_shared_memory()
            
            # Run news and trend analysis in parallel
            gathering_results = await data_team.run_in_parallel(query)
            news_data = gathering_results[0]
            trend_data = gathering_results[1]
            
            print("\n✅ Data gathering complete")
            print(f"News data length: {len(news_data)} chars")
            print(f"Trend data length: {len(trend_data)} chars")
            
            # Phase 2: Summarization
            print("\n📝 Phase 2: Synthesizing Results")
            
            # Prepare a structured input for the summarizer
            summary_prompt = f"""
Please synthesize and analyze the following startup research data.

GATHERED NEWS:
{news_data}

TREND ANALYSIS:
{trend_data}

Create a comprehensive analysis that:
1. Synthesizes key findings from both news and trend research
2. Identifies the most significant market developments
3. Highlights emerging patterns and opportunities
4. Provides actionable insights and recommendations
5. Notes any potential risks or challenges

Structure your response with clear sections for:
- Executive Summary
- Key Market Developments
- Emerging Trends
- Opportunities and Risks
- Actionable Recommendations
"""
            
            # Get the final synthesis from the summarizer
            final_research = await self.agents["summary"].run(summary_prompt)
            
            print("\n✅ Synthesis complete")
            
            # Validate the output
            self.metrics.validation_runs += 1
            evaluation = await self.evaluate_result(final_research)
            
            # Store in long-term memory
            await self.memories["long_term"].add_message({
                "role": "assistant",
                "content": final_research,
                "metadata": {
                    "query": query,
                    "timestamp": int(time.time() * 1000),
                    "evaluation_score": evaluation.score,
                    "team_research": True,
                    "raw_data": {
                        "news": news_data,
                        "trends": trend_data
                    }
                }
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            return {
                "result": final_research,
                "evaluation": evaluation,
                "processing_time": processing_time,
                "raw_data": {
                    "news": news_data,
                    "trends": trend_data
                }
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            self.metrics.total_processing_time += processing_time
            
            print(f"❌ Error processing complex query: {str(e)}")
            return {
                "result": f"Error: {str(e)}",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def generate_research_plan(self, query: str) -> str:
        """
        Generate a research plan using the SimpleLLMPlanner.
        
        Args:
            query: User's research query
            
        Returns:
            JSON string with research plan steps
        """
        print(f"\n📋 Generating research plan for: '{query}'")
        
        plan = await self.planner.generate_plan(
            query, 
            self.search_tools["all_tools"], 
            self.memories["shared"]
        )
        
        print(f"Plan generated: {len(plan)} characters")
        return plan
    
    async def evaluate_result(self, result: str) -> EvaluationResult:
        """
        Evaluate the quality of a research result.
        
        Args:
            result: The research result to evaluate
            
        Returns:
            Evaluation result with score, feedback, and improvements
        """
        print("\n⭐ Evaluating research quality...")
        
        # Create a message list for the evaluator
        messages = [
            {"role": "system", "content": "Evaluating startup research output quality."},
            {"role": "user", "content": "Provide comprehensive startup research."},
            {"role": "assistant", "content": f"FINAL ANSWER: {result}"}
        ]
        
        # Evaluate the result
        evaluation = await self.evaluator.evaluate(messages)
        
        # Track the evaluation score
        self.metrics.add_evaluation_score(evaluation.score)
        
        print(f"Evaluation complete: Score {evaluation.score:.2f}/1.0")
        return evaluation
    
    def _increment_agent_calls(self) -> None:
        """Increment agent calls counter."""
        self.metrics.agent_calls += 1
    
    def get_metrics_report(self) -> str:
        """Generate a report of performance metrics."""
        return self.metrics.report()


async def main():
    """
    Main function to demonstrate the startup research workflow.
    """
    print("\n" + "=" * 80)
    print("🚀 STARTUP RESEARCH AND ANALYSIS WORKFLOW DEMO")
    print("=" * 80)
    
    # Initialize the workflow
    workflow = StartupResearchWorkflow(debug=True)
    
    complex_queries = [
        "Analyze emerging trends in AI startups focusing on enterprise applications, including recent funding patterns, technology adoption trends, and potential market opportunities",
        #"Research the intersection of blockchain and sustainability startups, including funding rounds, technological innovations, and market challenges"
    ]
    
    # Generate a research plan for a complex query
    plan_query = complex_queries[0]
    research_plan = await workflow.generate_research_plan(plan_query)
    
    print("\n" + "=" * 80)
    print("📋 RESEARCH PLAN")
    print("=" * 80)
    print(f"Query: {plan_query}")
    print("\nPlan:")
    print("-" * 40)
    print(research_plan)
    print("-" * 40)
    
    # Process complex queries using the agent team
    print("\n" + "=" * 80)
    print("👥 PROCESSING MULTI-AGENT TEAM QUERIES")
    print("=" * 80)
    
    for query in complex_queries:
        result = await workflow.process_complex_query(query)
        
        print("\n" + "-" * 60)
        print(f"Query: {query}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        if "evaluation" in result:
            print(f"Evaluation score: {result['evaluation'].score:.2f}/1.0")
            print(f"Feedback: {result['evaluation'].feedback}")
        
        print("\nResult:")
        print("-" * 40)
        print(result["result"])
        print("-" * 60)
    
    # Display metrics report
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE METRICS")
    print("=" * 80)
    print(workflow.get_metrics_report())
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main()) 