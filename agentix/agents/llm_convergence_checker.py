from dataclasses import dataclass
from typing import List, Optional

from ..llms import LLM
from ..utils.debug_logger import DebugLogger


@dataclass
class ConvergenceCriteria:
    """Criteria for determining content convergence."""
    required_elements: Optional[List[str]] = None  # Required content elements
    required_structure: Optional[List[str]] = None  # Required structural elements
    minimum_length: Optional[int] = None  # Minimum content length
    custom_instructions: Optional[List[str]] = None  # Additional checking instructions


class LLMConvergenceChecker:
    """
    Uses an LLM to determine if content meets specific criteria.
    
    The convergence checker uses an LLM to analyze content against criteria like
    required elements, structure, minimum length, and custom instructions.
    """
    
    def __init__(
        self,
        model: LLM,
        criteria: ConvergenceCriteria,
        debug: bool = False
    ):
        """
        Initialize the convergence checker.
        
        Args:
            model: The LLM to use for checking (OpenAIChat or TogetherChat)
            criteria: Convergence criteria to check against
            debug: Whether to enable debug logging
        """
        self.model = model
        self.criteria = criteria
        self.logger = DebugLogger(debug)
    
    async def has_converged(self, content: str) -> bool:
        """
        Check if content meets convergence criteria using LLM.
        
        Args:
            content: The content to check
            
        Returns:
            True if content has converged, False otherwise
        """
        prompt = self._build_convergence_prompt(content)
        
        try:
            response = await self.model.call([{
                "role": "user",
                "content": prompt
            }])
            decision = self._parse_decision(response)
            
            self.logger.log("Convergence check result", {
                "decision": decision,
                "reasoning": response
            })
            
            return decision
        except Exception as e:
            self.logger.error("Error in convergence check", {"error": str(e)})
            return False  # Default to not converged on error
    
    def _build_convergence_prompt(self, content: str) -> str:
        """
        Build the prompt for convergence checking.
        
        Args:
            content: The content to check
            
        Returns:
            A prompt string for the LLM
        """
        criteria_list = []
        
        if self.criteria.required_elements:
            criteria_list.append(
                f"Content must include these elements: {', '.join(self.criteria.required_elements)}"
            )
        
        if self.criteria.required_structure:
            criteria_list.append(
                f"Content must have these structural elements: {', '.join(self.criteria.required_structure)}"
            )
        
        if self.criteria.minimum_length:
            criteria_list.append(
                f"Content must be at least {self.criteria.minimum_length} characters long"
            )
        
        if self.criteria.custom_instructions:
            criteria_list.extend(self.criteria.custom_instructions)
        
        return f"""
You are a convergence checker that determines if content meets specific criteria.

Criteria for convergence:
{chr(10).join([f"- {c}" for c in criteria_list])}

Content to check:
---
{content}
---

Analyze the content and determine if it meets ALL criteria.
Respond with "YES" or "NO" followed by a brief explanation.
Your response must start with either "YES:" or "NO:" followed by your reasoning.
"""
    
    def _parse_decision(self, response: str) -> bool:
        """
        Parse the LLM's response to get the yes/no decision.
        
        Args:
            response: The LLM response to parse
            
        Returns:
            True if the response indicates convergence, False otherwise
        """
        normalized = response.strip().lower()
        return normalized.startswith("yes:") 