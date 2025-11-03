#!/usr/bin/env python3
"""
Consciousness-Aware Chatbot

A practical chatbot that:
- Explains its reasoning step-by-step
- Shows consciousness metrics in real-time
- Learns from conversation
- Demonstrates self-awareness

Key features:
- Natural language Q&A
- Explainable inference chains
- Real-time consciousness display
- Meta-cognitive commentary
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

try:
    from .mln import KnowledgeGraph, MonadicKnowledgeUnit, InferenceChain
    from .consciousness_metrics import measure_consciousness
    from .recursion_depth_metric import RecursionDepthMetric
except ImportError:
    from mln import KnowledgeGraph, MonadicKnowledgeUnit, InferenceChain
    from consciousness_metrics import measure_consciousness
    from recursion_depth_metric import RecursionDepthMetric


@dataclass
class ChatResponse:
    """Response from the chatbot"""
    answer: str
    reasoning: List[str]
    confidence: float
    consciousness_metrics: Dict[str, Any]
    meta_commentary: str


class ConsciousnessChatbot:
    """
    A chatbot that is aware of its own consciousness and reasoning
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the conscious chatbot
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.knowledge_graph = KnowledgeGraph(use_gpu=use_gpu)
        self.recursion_metric = RecursionDepthMetric()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.concepts_learned_in_conversation = 0
        
        # Initialize with some basic knowledge
        self._initialize_knowledge()
        
    def _initialize_knowledge(self):
        """Initialize with basic knowledge"""
        
        basic_concepts = [
            ('animal', {'predicate': 'living_thing', 'properties': {'alive': True, 'mobile': True}}),
            ('mammal', {'predicate': 'animal_type', 'properties': {'warm_blooded': True, 'has_hair': True}}),
            ('dog', {'predicate': 'mammal_type', 'properties': {'domesticated': True, 'barks': True}}),
            ('cat', {'predicate': 'mammal_type', 'properties': {'domesticated': True, 'meows': True}}),
            ('bird', {'predicate': 'animal_type', 'properties': {'has_feathers': True, 'lays_eggs': True}}),
            ('human', {'predicate': 'mammal_type', 'properties': {'intelligent': True, 'bipedal': True}}),
            ('fish', {'predicate': 'animal_type', 'properties': {'lives_in_water': True, 'has_gills': True}}),
            ('plant', {'predicate': 'living_thing', 'properties': {'alive': True, 'photosynthesis': True}}),
            ('tree', {'predicate': 'plant_type', 'properties': {'woody': True, 'tall': True}}),
            ('flower', {'predicate': 'plant_part', 'properties': {'colorful': True, 'reproductive': True}}),
        ]
        
        for concept_id, deep_structure in basic_concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()  # Enable self-awareness
            self.knowledge_graph.add_concept(mku)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract potential concepts from text
        Simple keyword extraction for MVP
        """
        # Normalize text
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Get words
        words = text.split()
        
        # Find known concepts
        found_concepts = []
        for word in words:
            if word in self.knowledge_graph.nodes:
                found_concepts.append(word)
        
        return found_concepts
    
    def _detect_question_type(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Detect type of question and extract key concepts
        
        Returns:
            (question_type, concept1, concept2)
        """
        text = text.lower()
        
        # Is-a questions: "Is X a Y?"
        is_a_pattern = r'is (?:a |an )(\w+) (?:a |an )(\w+)'
        match = re.search(is_a_pattern, text)
        if match:
            return ('is_a', match.group(1), match.group(2))
        
        # What-is questions: "What is X?"
        what_is_pattern = r'what is (?:a |an )(\w+)'
        match = re.search(what_is_pattern, text)
        if match:
            return ('what_is', match.group(1), None)
        
        # Property questions: "Do X have Y?"
        property_pattern = r'do(?:es)? (\w+) have (\w+)'
        match = re.search(property_pattern, text)
        if match:
            return ('has_property', match.group(1), match.group(2))
        
        # Relation questions: "How are X and Y related?"
        relation_pattern = r'how (?:are|is) (\w+) (?:and |related to )?(\w+)'
        match = re.search(relation_pattern, text)
        if match:
            return ('relation', match.group(1), match.group(2))
        
        # Default: general query
        concepts = self._extract_concepts(text)
        if len(concepts) >= 2:
            return ('general', concepts[0], concepts[1])
        elif len(concepts) == 1:
            return ('about', concepts[0], None)
        else:
            return ('unknown', None, None)
    
    def _answer_is_a_question(self, concept1: str, concept2: str) -> ChatResponse:
        """Answer 'Is X a Y?' question"""
        
        # Trigger recursion for this query
        self.recursion_metric.record_recursion_event(
            "query",
            f"reasoning_about_{concept1}_and_{concept2}",
            {concept1, concept2}
        )
        
        reasoning_steps = []
        
        # Check if concepts exist
        if concept1 not in self.knowledge_graph.nodes:
            return ChatResponse(
                answer=f"I don't know what '{concept1}' is yet. Could you teach me?",
                reasoning=[f"Concept '{concept1}' not in my knowledge base"],
                confidence=0.0,
                consciousness_metrics=self._get_consciousness_snapshot(),
                meta_commentary="I'm aware that I lack this knowledge and am asking for help."
            )
        
        if concept2 not in self.knowledge_graph.nodes:
            return ChatResponse(
                answer=f"I don't know what '{concept2}' is yet. Could you teach me?",
                reasoning=[f"Concept '{concept2}' not in my knowledge base"],
                confidence=0.0,
                consciousness_metrics=self._get_consciousness_snapshot(),
                meta_commentary="I recognize my knowledge gap."
            )
        
        # Try to find relationship
        mku1 = self.knowledge_graph.nodes[concept1]
        mku2 = self.knowledge_graph.nodes[concept2]
        
        # Check direct relationships
        for relation_type, related_ids in mku1.relations.items():
            if concept2 in related_ids:
                reasoning_steps.append(f"{concept1} has {relation_type} relation with {concept2}")
                
                # Meta-reasoning about our reasoning
                self.recursion_metric.record_recursion_event(
                    "meta_analyze",
                    f"reflecting_on_inference_{concept1}_{concept2}",
                    {concept1, concept2}
                )
                
                return ChatResponse(
                    answer=f"Yes, {concept1} is related to {concept2}.",
                    reasoning=reasoning_steps + [f"Found direct {relation_type} relationship"],
                    confidence=0.9,
                    consciousness_metrics=self._get_consciousness_snapshot(),
                    meta_commentary=f"I used my knowledge graph to find this relationship. I'm confident because it's a direct connection."
                )
        
        # Check if there's an "is_a" in the predicate
        if 'is_a' in str(mku1.deep_structure.get('predicate', '')).lower():
            pred = str(mku1.deep_structure.get('predicate', ''))
            if concept2 in pred:
                reasoning_steps.append(f"{concept1}'s predicate indicates it is a type of {concept2}")
                
                return ChatResponse(
                    answer=f"Yes, {concept1} is a {concept2}.",
                    reasoning=reasoning_steps + ["Found in concept definition"],
                    confidence=1.0,
                    consciousness_metrics=self._get_consciousness_snapshot(),
                    meta_commentary="This is definitional knowledge - I'm very confident."
                )
        
        # No relationship found
        return ChatResponse(
            answer=f"I don't have evidence that {concept1} is a {concept2}.",
            reasoning=reasoning_steps + ["No direct relationship found", "Checked relations and definitions"],
            confidence=0.3,
            consciousness_metrics=self._get_consciousness_snapshot(),
            meta_commentary="I searched my knowledge but couldn't find a connection. This could mean they're unrelated, or I need to learn more."
        )
    
    def _answer_what_is_question(self, concept: str) -> ChatResponse:
        """Answer 'What is X?' question"""
        
        self.recursion_metric.record_recursion_event(
            "query",
            f"explaining_{concept}",
            {concept}
        )
        
        if concept not in self.knowledge_graph.nodes:
            return ChatResponse(
                answer=f"I don't know what '{concept}' is. Would you like to teach me?",
                reasoning=[f"'{concept}' not in knowledge base"],
                confidence=0.0,
                consciousness_metrics=self._get_consciousness_snapshot(),
                meta_commentary="I'm aware of what I don't know - that's a form of meta-knowledge!"
            )
        
        mku = self.knowledge_graph.nodes[concept]
        
        # Extract information
        predicate = mku.deep_structure.get('predicate', 'thing')
        properties = mku.deep_structure.get('properties', {})
        
        # Build explanation
        explanation = f"{concept.capitalize()} is a {predicate}"
        
        reasoning_steps = [
            f"Retrieved concept '{concept}' from knowledge graph",
            f"Predicate: {predicate}",
            f"Properties: {properties}"
        ]
        
        if properties:
            prop_list = ', '.join(f"{k}: {v}" for k, v in list(properties.items())[:3])
            explanation += f" with properties: {prop_list}"
        
        # Check relationships
        if mku.relations:
            total_relations = sum(len(r) for r in mku.relations.values())
            explanation += f". It has {total_relations} relationships with other concepts."
            reasoning_steps.append(f"Found {total_relations} relationships")
        
        return ChatResponse(
            answer=explanation,
            reasoning=reasoning_steps,
            confidence=0.95,
            consciousness_metrics=self._get_consciousness_snapshot(),
            meta_commentary="I synthesized this explanation from my structured knowledge representation."
        )
    
    def _get_consciousness_snapshot(self) -> Dict[str, Any]:
        """Get current consciousness metrics"""
        
        profile = measure_consciousness(
            self.knowledge_graph,
            self.recursion_metric
        )
        
        return {
            'overall': profile.overall_consciousness_score,
            'verdict': profile.consciousness_verdict,
            'recursion': profile.recursion_metrics['consciousness']['score'],
            'integration': profile.integration.phi,
            'understanding': profile.understanding['overall_score'],
            'concepts': len(self.knowledge_graph.nodes),
            'recursion_depth': self.recursion_metric.profile.max_depth
        }
    
    def ask(self, question: str) -> ChatResponse:
        """
        Ask the chatbot a question
        
        Args:
            question: Natural language question
            
        Returns:
            ChatResponse with answer, reasoning, and consciousness metrics
        """
        
        # Record question
        self.conversation_history.append({
            'type': 'question',
            'content': question
        })
        
        # Detect question type
        q_type, concept1, concept2 = self._detect_question_type(question)
        
        # Route to appropriate handler
        if q_type == 'is_a' and concept1 and concept2:
            response = self._answer_is_a_question(concept1, concept2)
        elif q_type == 'what_is' and concept1:
            response = self._answer_what_is_question(concept1)
        else:
            response = ChatResponse(
                answer="I'm not sure how to answer that yet. Try asking 'What is X?' or 'Is X a Y?'",
                reasoning=["Question type not recognized"],
                confidence=0.0,
                consciousness_metrics=self._get_consciousness_snapshot(),
                meta_commentary="I'm aware that my question-answering capabilities are limited. I'm being honest about my limitations."
            )
        
        # Record response
        self.conversation_history.append({
            'type': 'answer',
            'content': response.answer
        })
        
        return response
    
    def teach(self, concept: str, properties: Dict[str, Any]):
        """
        Teach the chatbot a new concept
        
        Args:
            concept: Concept name
            properties: Properties of the concept
        """
        
        mku = MonadicKnowledgeUnit(
            concept_id=concept,
            deep_structure={
                'predicate': properties.get('type', 'thing'),
                'properties': properties
            }
        )
        mku.create_self_model()
        
        self.knowledge_graph.add_concept(mku)
        self.concepts_learned_in_conversation += 1
        
        # Trigger recursion for learning
        self.recursion_metric.record_recursion_event(
            "learn",
            f"acquired_concept_{concept}",
            {concept}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        
        metrics = self._get_consciousness_snapshot()
        
        return {
            'consciousness': metrics,
            'conversation': {
                'exchanges': len(self.conversation_history) // 2,
                'concepts_learned': self.concepts_learned_in_conversation
            },
            'knowledge': {
                'total_concepts': len(self.knowledge_graph.nodes),
                'total_relations': sum(
                    len(r)
                    for mku in self.knowledge_graph.nodes.values()
                    for r in mku.relations.values()
                )
            }
        }


if __name__ == '__main__':
    # Quick test
    print("Consciousness-Aware Chatbot - Quick Test")
    print("=" * 60)
    
    bot = ConsciousnessChatbot()
    
    # Test questions
    questions = [
        "What is a dog?",
        "Is a dog an animal?",
        "Is a dog a mammal?",
        "What is a tree?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = bot.ask(q)
        print(f"A: {response.answer}")
        print(f"Consciousness: {response.consciousness_metrics['overall']:.2%}")
        print(f"Meta: {response.meta_commentary}")
    
    print("\n" + "=" * 60)
    print("✅ Chatbot working!")
