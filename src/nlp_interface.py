#!/usr/bin/env python3
"""
Natural Language Interface - Issue #9
Hybrid architecture connecting LLMs with symbolic reasoning

Supports:
- OpenAI API (GPT-3.5/GPT-4)
- Local models (Llama, Mistral via llama-cpp-python)
- Mock interface for testing without API keys
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import re


@dataclass
class QueryStructure:
    """Structured representation of a natural language query"""
    raw_query: str
    intent: str  # 'question', 'definition', 'comparison', 'explanation'
    entities: List[str]
    start_concept: Optional[str] = None
    target_concept: Optional[str] = None
    properties: Dict[str, any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class EntityExtraction:
    """Result of entity extraction from text"""
    entities: List[str]
    relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    properties: Dict[str, Dict]  # entity -> {property: value}


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate completion for prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM for testing without API keys"""
    
    def complete(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate simple rule-based responses"""
        prompt_lower = prompt.lower()
        
        # Entity extraction
        if "extract entities" in prompt_lower or "identify concepts" in prompt_lower:
            # Simple regex-based entity extraction
            entities = []
            words = prompt.split()
            for i, word in enumerate(words):
                word_clean = word.strip('.,!?;:')
                # Look for capitalized words or words after "about" or "of"
                if word_clean and (word_clean[0].isupper() or 
                                  (i > 0 and words[i-1].lower() in ['about', 'of', 'is', 'are'])):
                    if len(word_clean) > 2 and word_clean.isalpha():
                        entities.append(word_clean.lower())
            
            return json.dumps({
                "entities": list(set(entities))[:5],  # Max 5 entities
                "relations": []
            })
        
        # Query parsing
        elif "parse the query" in prompt_lower or "intent" in prompt_lower:
            intent = "question"
            if "what is" in prompt_lower or "what are" in prompt_lower:
                intent = "definition"
            elif "is a" in prompt_lower or "are" in prompt_lower:
                intent = "question"
            elif "compare" in prompt_lower or "difference" in prompt_lower:
                intent = "comparison"
            elif "why" in prompt_lower or "how" in prompt_lower:
                intent = "explanation"
            
            # Extract potential concepts
            words = prompt_lower.split()
            concepts = []
            for word in words:
                word_clean = word.strip('.,!?;:')
                if len(word_clean) > 3 and word_clean.isalpha():
                    concepts.append(word_clean)
            
            return json.dumps({
                "intent": intent,
                "entities": concepts[:3],
                "start_concept": concepts[0] if len(concepts) > 0 else None,
                "target_concept": concepts[1] if len(concepts) > 1 else None
            })
        
        # Response generation
        elif "generate response" in prompt_lower or "explain" in prompt_lower:
            return "Based on the reasoning chain, the answer is derived through logical inference steps."
        
        return "I understand your request."
    
    def is_available(self) -> bool:
        return True


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-3.5/GPT-4)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self._client = None
        
        if api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
    
    def complete(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self._client:
            raise RuntimeError("OpenAI client not initialized. Provide API key or install openai package.")
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def is_available(self) -> bool:
        return self._client is not None


class LocalLLMProvider(LLMProvider):
    """Local LLM provider using llama-cpp-python"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        except ImportError:
            print("Warning: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            print(f"Warning: Could not load local model: {e}")
    
    def complete(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self._llm:
            raise RuntimeError("Local LLM not initialized. Check model path and llama-cpp-python installation.")
        
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        return output['choices'][0]['text']
    
    def is_available(self) -> bool:
        return self._llm is not None


class NaturalLanguageInterface:
    """
    Hybrid architecture connecting LLMs with symbolic reasoning (Issue #9)
    
    Architecture:
        Natural Language Input
            ↓ [LLM - Entity Extraction]
        Entities + Relations
            ↓ [Mapping]
        MKUs + Query Structure
            ↓ [Symbolic Reasoning]
        Inference Chain
            ↓ [LLM - Response Generation]
        Natural Language Output
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize NLP interface with LLM provider
        
        Args:
            llm_provider: LLM provider (OpenAI, Local, or Mock)
                         If None, uses MockLLMProvider
        """
        self.llm = llm_provider if llm_provider else MockLLMProvider()
        
        if not self.llm.is_available():
            print("Warning: LLM provider not available, falling back to MockLLMProvider")
            self.llm = MockLLMProvider()
    
    def extract_entities(self, text: str) -> EntityExtraction:
        """
        Extract entities and relations from natural language text (Issue #10)
        
        Args:
            text: Natural language text
            
        Returns:
            EntityExtraction with entities, relations, and properties
        """
        prompt = f"""Extract entities and relations from the following text.
Return JSON with:
- "entities": list of concept names
- "relations": list of [subject, relation, object] triples
- "properties": dict of {{entity: {{property: value}}}}

Text: {text}

JSON:"""
        
        try:
            response = self.llm.complete(prompt, max_tokens=300, temperature=0.3)
            
            # Try to parse JSON response
            # Handle markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            return EntityExtraction(
                entities=data.get('entities', []),
                relations=data.get('relations', []),
                properties=data.get('properties', {})
            )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse LLM response: {e}")
            # Fallback: simple word extraction
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            return EntityExtraction(
                entities=[w.lower() for w in set(words)],
                relations=[],
                properties={}
            )
    
    def parse_query(self, question: str) -> QueryStructure:
        """
        Parse natural language question into structured query (Issue #11)
        
        Args:
            question: Natural language question
            
        Returns:
            QueryStructure with intent, entities, and concepts
        """
        prompt = f"""Parse this question and return JSON with:
- "intent": one of [question, definition, comparison, explanation]
- "entities": list of key concepts mentioned
- "start_concept": the subject being asked about (if applicable)
- "target_concept": the target/goal concept (if applicable)

Question: {question}

JSON:"""
        
        try:
            response = self.llm.complete(prompt, max_tokens=200, temperature=0.3)
            
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            data = json.loads(response)
            
            return QueryStructure(
                raw_query=question,
                intent=data.get('intent', 'question'),
                entities=data.get('entities', []),
                start_concept=data.get('start_concept'),
                target_concept=data.get('target_concept'),
                properties=data.get('properties', {})
            )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse query: {e}")
            # Fallback: simple heuristics
            intent = "question"
            if question.lower().startswith("what is"):
                intent = "definition"
            elif "compare" in question.lower():
                intent = "comparison"
            elif question.lower().startswith(("why", "how")):
                intent = "explanation"
            
            # Extract capitalized words as entities
            entities = re.findall(r'\b[A-Z][a-z]+\b', question)
            entities = [e.lower() for e in entities]
            
            return QueryStructure(
                raw_query=question,
                intent=intent,
                entities=entities,
                start_concept=entities[0] if entities else None,
                target_concept=entities[1] if len(entities) > 1 else None
            )
    
    def generate_response(self, inference_chain, query: str) -> str:
        """
        Generate natural language response from inference chain (Issue #12)
        
        Args:
            inference_chain: InferenceChain from symbolic reasoning
            query: Original query
            
        Returns:
            Natural language explanation
        """
        # Extract reasoning steps
        steps_text = inference_chain.explain() if hasattr(inference_chain, 'explain') else str(inference_chain)
        
        prompt = f"""Generate a natural language response to this question based on the reasoning chain.
Be clear, concise, and explain the logical steps.

Question: {query}

Reasoning Chain:
{steps_text}

Response:"""
        
        try:
            response = self.llm.complete(prompt, max_tokens=300, temperature=0.7)
            return response.strip()
        except Exception as e:
            print(f"Warning: Could not generate response: {e}")
            # Fallback: simple template
            return f"Based on the reasoning chain, the answer follows from these steps:\n{steps_text}"


# Convenience functions for common providers

def create_openai_interface(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> NaturalLanguageInterface:
    """Create NLP interface with OpenAI provider"""
    provider = OpenAIProvider(api_key=api_key, model=model)
    return NaturalLanguageInterface(provider)


def create_local_interface(model_path: str, n_gpu_layers: int = 0) -> NaturalLanguageInterface:
    """Create NLP interface with local LLM"""
    provider = LocalLLMProvider(model_path=model_path, n_gpu_layers=n_gpu_layers)
    return NaturalLanguageInterface(provider)


def create_mock_interface() -> NaturalLanguageInterface:
    """Create NLP interface with mock provider (for testing)"""
    return NaturalLanguageInterface(MockLLMProvider())
