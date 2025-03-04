from typing import List, Dict, Any, Optional, Union, Callable, Awaitable
import json
import time

from ...llms import LLM
from ...utils.debug_logger import DebugLogger


class LLMConvergenceChecker:
    """
    Uses an LLM to check if content meets specific criteria
    
    This is useful for determining if agent output meets certain
    quality standards or contains the information we're looking for.
    """
    
    def __init__(
        self,
        llm: LLM,
        criteria: List[str],
        debug: bool = False
    ):
        """
        Initialize a new LLM convergence checker.
        
        Args:
            llm: The LLM to use for checking (OpenAIChat or TogetherChat)
            criteria: List of criteria to check against
            debug: Whether to enable debug logging
        """
        self.llm = llm
        self.criteria = criteria
        self.logger = DebugLogger(debug)
        self.history: List[Dict[str, Any]] = []
    
    def build_convergence_prompt(self, content: str) -> str:
        """
        Build a prompt for checking convergence.
        
        Args:
            content: The content to check
            
        Returns:
            A prompt for the LLM
        """
        criteria_text = "\n".join([
            f"{i+1}. {criterion}" for i, criterion in enumerate(self.criteria)
        ])
        
        return f"""
You are a quality assurance evaluator for an AI assistant. Your job is to determine if the following content meets the specified criteria:

CRITERIA:
{criteria_text}

CONTENT TO EVALUATE:
{content}

Please evaluate the content and determine if it meets ALL of the criteria listed above.
You must respond in JSON format only with the following structure:
{{
  "meets_all_criteria": true|false,
  "explanation": "brief explanation of your reasoning",
  "criteria_assessment": [
    {{ "criterion": "criterion text", "met": true|false, "reason": "reason" }},
    ...
  ]
}}
"""
    
    async def check_convergence(self, content: str) -> bool:
        """
        Check if the content has converged (meets all criteria).
        
        Args:
            content: The content to check
            
        Returns:
            True if the content meets all criteria, False otherwise
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self.build_convergence_prompt(content)
        
        # Call LLM
        try:
            self.logger.log("Checking convergence", {
                "content_length": len(content),
                "criteria_count": len(self.criteria)
            })
            
            llm_response = await self.llm.call([
                {"role": "system", "content": "You are a helpful assistant specializing in content evaluation."},
                {"role": "user", "content": prompt}
            ])
            
            # Parse response
            result = json.loads(llm_response)
            
            # Store in history
            self.history.append({
                "timestamp": int(time.time() * 1000),
                "content_summary": content[:100] + "..." if len(content) > 100 else content,
                "assessment": result,
                "duration_ms": int((time.time() - start_time) * 1000)
            })
            
            # Log result
            meets_criteria = result.get("meets_all_criteria", False)
            self.logger.log("Convergence check result", {
                "meets_all_criteria": meets_criteria,
                "criteria_assessment": result.get("criteria_assessment", [])
            })
            
            return meets_criteria
            
        except Exception as error:
            self.logger.error(f"Error checking convergence: {str(error)}")
            # Store error in history
            self.history.append({
                "timestamp": int(time.time() * 1000),
                "content_summary": content[:100] + "..." if len(content) > 100 else content,
                "error": str(error),
                "duration_ms": int((time.time() - start_time) * 1000)
            })
            return False
    
    def get_criteria_check_function(self) -> Callable[[str], Awaitable[bool]]:
        """
        Get a function that can be used to check if content meets criteria.
        
        Returns:
            A function that takes content and returns a promise that resolves to true/false
        """
        return self.check_convergence
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of convergence checks.
        
        Returns:
            List of convergence check results with timestamps
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear the history of convergence checks."""
        self.logger.log("Clearing convergence check history")
        self.history = [] 