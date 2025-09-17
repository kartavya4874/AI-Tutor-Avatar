# llm_response_module.py
"""
LLM Response Module for AI Virtual Teacher
Handles LLM API integration, query processing, and response refinement
"""

import json
import requests
import time
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List, Union, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import re

# For local LLM support (uncomment in production)
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch

class ResponseType(Enum):
    """Types of responses the system can generate"""
    EXPLANATION = "explanation"
    QUIZ = "quiz"
    EXAMPLE = "example"
    CLARIFICATION = "clarification"
    ENCOURAGEMENT = "encouragement"

@dataclass
class QueryContext:
    """Context information for a user query"""
    user_id: str
    session_id: str
    previous_messages: List[Dict[str, str]]
    learning_level: str = "intermediate"  # beginner, intermediate, advanced
    subject_area: Optional[str] = None
    preferred_explanation_style: str = "detailed"  # brief, detailed, visual

@dataclass
class TeachingResponse:
    """Structured teaching response"""
    content: str
    response_type: ResponseType
    confidence: float
    follow_up_questions: List[str]
    key_concepts: List[str]
    difficulty_level: str
    estimated_reading_time: int  # seconds
    metadata: Dict[str, Any]

class LLMResponseHandler:
    """
    LLM Response Handler for generating and refining educational content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM Response Handler
        
        Args:
            config: Configuration dictionary for LLM settings
        """
        self.config = config or self._default_config()
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
        self.base_url = self.config.get("base_url", "https://api.openai.com/v1")
        
        # Response refinement templates
        self.teaching_prompts = self._load_teaching_prompts()
        
        # Session management
        self.active_sessions = {}
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize local model if specified
        self.local_model = None
        if self.config.get("use_local_model", False):
            self._initialize_local_model()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for LLM handler"""
        return {
            "model_name": "gpt-3.5-turbo",
            "api_key": None,  # Set via environment or config
            "base_url": "https://api.openai.com/v1",
            "max_tokens": 500,
            "temperature": 0.7,
            "use_local_model": False,
            "local_model_path": "microsoft/DialoGPT-medium",
            "response_timeout": 30,
            "retry_attempts": 3,
            "refinement_enabled": True
        }
    
    def _load_teaching_prompts(self) -> Dict[str, str]:
        """Load teaching prompt templates"""
        return {
            "explanation": """
            You are an AI virtual teacher. Explain the following topic in a clear, engaging way suitable for a {level} level student.
            
            Topic: {query}
            
            Structure your response with:
            1. A brief introduction to hook the student's interest
            2. Main explanation with examples
            3. Key takeaways
            4. Connection to real-world applications
            
            Keep the tone conversational and encouraging. Use analogies when helpful.
            """,
            
            "quiz": """
            Create a quiz question based on this topic: {query}
            
            Generate:
            1. One thoughtful question that tests understanding
            2. Multiple choice options (A, B, C, D)
            3. The correct answer with explanation
            4. A hint for struggling students
            
            Difficulty level: {level}
            """,
            
            "clarification": """
            A student needs clarification on: {query}
            
            Previous context: {context}
            
            Provide a clear, focused clarification that:
            1. Addresses the specific confusion
            2. Uses simpler language if needed
            3. Provides a concrete example
            4. Checks for understanding
            """,
            
            "example": """
            Provide a practical example for: {query}
            
            Create an example that:
            1. Is relatable to a {level} student
            2. Shows real-world application
            3. Is step-by-step if applicable
            4. Reinforces key concepts
            """
        }
    
    def _initialize_local_model(self):
        """Initialize local LLM model"""
        try:
            # In production, uncomment these lines:
            # model_path = self.config["local_model_path"]
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # self.local_model = AutoModelForCausalLM.from_pretrained(model_path)
            # self.text_generator = pipeline("text-generation", 
            #                               model=self.local_model,
            #                               tokenizer=self.tokenizer)
            
            # For demo purposes, create mock local model
            self.local_model = self._create_mock_local_model()
            self.logger.info("Local model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local model: {e}")
            self.local_model = None
    
    def _create_mock_local_model(self):
        """Create mock local model for development"""
        class MockLocalModel:
            def generate_response(self, prompt: str) -> str:
                # Simple mock response based on prompt keywords
                if "explain" in prompt.lower():
                    return "This is a detailed explanation of the concept with examples and real-world applications."
                elif "quiz" in prompt.lower():
                    return "Quiz Question: What is the main concept?\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 4\nCorrect Answer: B"
                elif "example" in prompt.lower():
                    return "Here's a practical example that demonstrates the concept step by step."
                else:
                    return "I understand your question. Let me provide a comprehensive response that addresses your learning needs."
        
        return MockLocalModel()
    
    async def generate_response_async(self, query: str, 
                                    context: QueryContext) -> TeachingResponse:
        """
        Generate response asynchronously using LLM API
        
        Args:
            query: User query/question
            context: Query context information
            
        Returns:
            TeachingResponse: Structured teaching response
        """
        try:
            # Determine response type
            response_type = self._classify_query_type(query)
            
            # Build prompt
            prompt = self._build_teaching_prompt(query, context, response_type)
            
            # Generate response
            if self.config.get("use_local_model") and self.local_model:
                raw_response = await self._generate_local_response(prompt)
            else:
                raw_response = await self._generate_api_response(prompt, context)
            
            # Refine response
            if self.config.get("refinement_enabled"):
                refined_response = self._refine_response(raw_response, context)
            else:
                refined_response = raw_response
            
            # Create structured response
            teaching_response = self._create_teaching_response(
                refined_response, response_type, query, context
            )
            
            # Update session context
            self._update_session_context(context.session_id, query, teaching_response)
            
            return teaching_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            return self._create_fallback_response(query, context)
    
    def generate_response(self, query: str, 
                         context: QueryContext) -> TeachingResponse:
        """
        Generate response synchronously
        
        Args:
            query: User query/question
            context: Query context information
            
        Returns:
            TeachingResponse: Structured teaching response
        """
        return asyncio.run(self.generate_response_async(query, context))
    
    def _classify_query_type(self, query: str) -> ResponseType:
        """
        Classify the type of query to determine response format
        
        Args:
            query: User query
            
        Returns:
            ResponseType: Classified query type
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["quiz", "test", "question", "check"]):
            return ResponseType.QUIZ
        elif any(word in query_lower for word in ["example", "show me", "demonstrate"]):
            return ResponseType.EXAMPLE
        elif any(word in query_lower for word in ["clarify", "confused", "don't understand"]):
            return ResponseType.CLARIFICATION
        elif any(word in query_lower for word in ["explain", "what is", "how does", "why"]):
            return ResponseType.EXPLANATION
        else:
            return ResponseType.EXPLANATION  # Default
    
    def _build_teaching_prompt(self, query: str, context: QueryContext, 
                              response_type: ResponseType) -> str:
        """
        Build teaching prompt based on query and context
        
        Args:
            query: User query
            context: Query context
            response_type: Type of response needed
            
        Returns:
            str: Formatted prompt
        """
        template = self.teaching_prompts.get(response_type.value, self.teaching_prompts["explanation"])
        
        prompt = template.format(
            query=query,
            level=context.learning_level,
            context=self._format_conversation_context(context.previous_messages)
        )
        
        return prompt
    
    def _format_conversation_context(self, messages: List[Dict[str, str]]) -> str:
        """Format previous messages for context"""
        if not messages:
            return "No previous context"
        
        context_parts = []
        for msg in messages[-3:]:  # Last 3 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"{role}: {content[:100]}...")
        
        return "\n".join(context_parts)
    
    async def _generate_api_response(self, prompt: str, 
                                   context: QueryContext) -> str:
        """
        Generate response using external API
        
        Args:
            prompt: Formatted prompt
            context: Query context
            
        Returns:
            str: Generated response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an AI virtual teacher specialized in clear, engaging explanations."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"]
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config["retry_attempts"]):
                try:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.config["response_timeout"]
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        else:
                            self.logger.warning(f"API request failed with status {response.status}")
                            
                except Exception as e:
                    self.logger.warning(f"API attempt {attempt + 1} failed: {e}")
                    if attempt < self.config["retry_attempts"] - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts fail, use local fallback
        return await self._generate_local_response(prompt)
    
    async def _generate_local_response(self, prompt: str) -> str:
        """
        Generate response using local model
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            str: Generated response
        """
        if self.local_model:
            return self.local_model.generate_response(prompt)
        else:
            return "I apologize, but I'm currently unable to process your request. Please try again later."
    
    def _refine_response(self, raw_response: str, 
                        context: QueryContext) -> str:
        """
        Refine raw LLM response for better teaching quality
        
        Args:
            raw_response: Raw response from LLM
            context: Query context
            
        Returns:
            str: Refined response
        """
        refined = raw_response
        
        # Add personalized greeting if first message in session
        if len(context.previous_messages) == 0:
            greeting = f"Hello! I'm excited to help you learn about this topic. "
            refined = greeting + refined
        
        # Ensure appropriate complexity level
        if context.learning_level == "beginner":
            refined = self._simplify_language(refined)
        elif context.learning_level == "advanced":
            refined = self._add_technical_details(refined)
        
        # Add encouraging closing
        if not any(phrase in refined.lower() for phrase in ["questions?", "help", "clarify"]):
            refined += "\n\nFeel free to ask if you need any clarification or have follow-up questions!"
        
        return refined
    
    def _simplify_language(self, text: str) -> str:
        """Simplify language for beginner level"""
        # Replace complex terms with simpler alternatives
        replacements = {
            "utilize": "use",
            "consequently": "so",
            "furthermore": "also",
            "however": "but",
            "therefore": "so"
        }
        
        for complex_word, simple_word in replacements.items():
            text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_technical_details(self, text: str) -> str:
        """Add technical details for advanced level"""
        # This would involve more sophisticated analysis in production
        if "concept" in text.lower() and not "algorithm" in text.lower():
            text += "\n\nFor a deeper understanding, you might want to explore the underlying algorithms and mathematical foundations."
        
        return text
    
    def _create_teaching_response(self, content: str, response_type: ResponseType,
                                 query: str, context: QueryContext) -> TeachingResponse:
        """
        Create structured teaching response
        
        Args:
            content: Response content
            response_type: Type of response
            query: Original query
            context: Query context
            
        Returns:
            TeachingResponse: Structured response object
        """
        # Extract key concepts (simple keyword extraction for demo)
        key_concepts = self._extract_key_concepts(content)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(query, response_type)
        
        # Calculate confidence score (mock for demo)
        confidence = 0.85 if len(content) > 100 else 0.70
        
        # Estimate reading time (approximate)
        word_count = len(content.split())
        reading_time = max(10, word_count * 0.5)  # 0.5 seconds per word
        
        return TeachingResponse(
            content=content,
            response_type=response_type,
            confidence=confidence,
            follow_up_questions=follow_up_questions,
            key_concepts=key_concepts,
            difficulty_level=context.learning_level,
            estimated_reading_time=int(reading_time),
            metadata={
                "query": query,
                "timestamp": time.time(),
                "model_used": self.model_name,
                "session_id": context.session_id
            }
        )
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from response content"""
        # Simple keyword extraction (in production, use NLP techniques)
        words = content.lower().split()
        
        # Filter for potential concepts (nouns, important terms)
        important_words = []
        for word in words:
            if (len(word) > 4 and 
                word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']):
                important_words.append(word.strip('.,!?'))
        
        # Return unique words, limited to 5 key concepts
        return list(set(important_words))[:5]
    
    def _generate_follow_up_questions(self, query: str, 
                                    response_type: ResponseType) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []
        
        if response_type == ResponseType.EXPLANATION:
            follow_ups = [
                "Would you like me to explain any specific part in more detail?",
                "Can you think of a real-world example where this applies?",
                "What questions do you have about this concept?"
            ]
        elif response_type == ResponseType.EXAMPLE:
            follow_ups = [
                "Would you like to see another example?",
                "Can you try creating your own example?",
                "What part of this example would you like me to explain further?"
            ]
        elif response_type == ResponseType.QUIZ:
            follow_ups = [
                "Would you like to try another question?",
                "Should I explain the correct answer in more detail?",
                "Want to explore a related topic?"
            ]
        
        return follow_ups[:2]  # Limit to 2 follow-up questions
    
    def _create_fallback_response(self, query: str, 
                                 context: QueryContext) -> TeachingResponse:
        """Create fallback response when generation fails"""
        fallback_content = (
            "I apologize, but I'm having trouble processing your question right now. "
            "Could you please rephrase your question or try asking something more specific? "
            "I'm here to help you learn!"
        )
        
        return TeachingResponse(
            content=fallback_content,
            response_type=ResponseType.CLARIFICATION,
            confidence=0.3,
            follow_up_questions=["Could you rephrase your question?", "What specific topic would you like to explore?"],
            key_concepts=[],
            difficulty_level=context.learning_level,
            estimated_reading_time=10,
            metadata={"fallback": True, "original_query": query}
        )
    
    def _update_session_context(self, session_id: str, query: str, 
                               response: TeachingResponse):
        """Update session context with new interaction"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "messages": [],
                "concepts_covered": set(),
                "start_time": time.time()
            }
        
        session = self.active_sessions[session_id]
        session["messages"].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response.content}
        ])
        session["concepts_covered"].update(response.key_concepts)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of learning session"""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        return {
            "total_interactions": len(session["messages"]) // 2,
            "concepts_covered": list(session["concepts_covered"]),
            "session_duration": time.time() - session["start_time"],
            "last_activity": session["messages"][-1] if session["messages"] else None
        }


# Factory and utility functions
def create_llm_handler(config: Optional[Dict[str, Any]] = None) -> LLMResponseHandler:
    """
    Factory function to create LLMResponseHandler instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        LLMResponseHandler: Configured LLM handler instance
    """
    return LLMResponseHandler(config)

def create_query_context(user_id: str, session_id: str, 
                        learning_level: str = "intermediate",
                        previous_messages: Optional[List[Dict[str, str]]] = None) -> QueryContext:
    """
    Create QueryContext object
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        learning_level: User's learning level
        previous_messages: Previous conversation messages
        
    Returns:
        QueryContext: Query context object
    """
    return QueryContext(
        user_id=user_id,
        session_id=session_id,
        learning_level=learning_level,
        previous_messages=previous_messages or []
    )

def test_llm_response():
    """Test function for LLM response module"""
    print("Testing LLM Response Module...")
    
    # Create LLM handler
    config = {"use_local_model": True, "refinement_enabled": True}
    llm_handler = create_llm_handler(config)
    
    # Create test context
    context = create_query_context(
        user_id="test_user",
        session_id="test_session",
        learning_level="intermediate"
    )
    
    # Test different types of queries
    test_queries = [
        "Explain how photosynthesis works",
        "Can you give me an example of Newton's first law?",
        "I'm confused about quantum mechanics",
        "Create a quiz question about Python programming"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        response = llm_handler.generate_response(query, context)
        print(f"Response type: {response.response_type.value}")
        print(f"Content length: {len(response.content)} characters")
        print(f"Key concepts: {response.key_concepts[:3]}")
        print(f"Confidence: {response.confidence}")
    
    # Test session summary
    summary = llm_handler.get_session_summary("test_session")
    print(f"\nSession summary: {summary}")
    
    print("LLM Response Module test completed!")


if __name__ == "__main__":
    test_llm_response()
