#!/usr/bin/env python3
"""
Neurosymbolic System - Issues #9-12
Integrates natural language interface with symbolic reasoning

Complete pipeline:
  Natural Language → Entity Extraction → Symbolic Reasoning → Natural Language Response
"""

from typing import Dict, Optional, List
import sys
import os

# Handle both package and standalone imports
try:
    from .mln import HybridIntelligenceSystem, MonadicKnowledgeUnit
    from .nlp_interface import (
        NaturalLanguageInterface,
        QueryStructure,
        EntityExtraction,
        create_mock_interface
    )
    from .ontology_loader import OntologyIntegrator
    from .pattern_learner import ExampleBasedLearner, Example
except ImportError:
    # Standalone execution
    sys.path.insert(0, os.path.dirname(__file__))
    from mln import HybridIntelligenceSystem, MonadicKnowledgeUnit
    from nlp_interface import (
        NaturalLanguageInterface,
        QueryStructure,
        EntityExtraction,
        create_mock_interface
    )
    from ontology_loader import OntologyIntegrator
    from pattern_learner import ExampleBasedLearner, Example


class NeurosymbolicSystem:
    """
    Complete neurosymbolic system combining LLMs with symbolic reasoning
    
    Architecture (Phase 2):
        Natural Language Input
            ↓ [LLM Entity Extraction]
        Entities + Intent
            ↓ [Map to Knowledge Graph]
        MKUs + QueryStructure  
            ↓ [Symbolic Reasoning]
        InferenceChain
            ↓ [LLM Response Generation]
        Natural Language Output + Explanation
    """
    
    def __init__(
        self,
        nlp_interface: Optional[NaturalLanguageInterface] = None,
        use_gpu: bool = False,
        device: str = 'auto'
    ):
        """
        Initialize neurosymbolic system
        
        Args:
            nlp_interface: NLP interface for LLM access (defaults to mock)
            use_gpu: Enable GPU acceleration for symbolic reasoning
            device: GPU device ('cuda', 'mps', 'cpu', or 'auto')
        """
        # Symbolic reasoning engine
        self.symbolic_system = HybridIntelligenceSystem(use_gpu=use_gpu, device=device)
        
        # Natural language interface
        self.nlp = nlp_interface if nlp_interface else create_mock_interface()
        
        # Ontology loader (Issue #13)
        self.ontology = OntologyIntegrator()
        
        # Pattern learner (Issue #14)
        self.learner = ExampleBasedLearner()
        
        print(f"NeurosymbolicSystem initialized")
        print(f"  - Symbolic reasoning: GPU={use_gpu}")
        print(f"  - NLP interface: {type(self.nlp.llm).__name__}")
        print(f"  - Ontology sources: ConceptNet, DBpedia, Wikidata")
        print(f"  - Pattern learning: Enabled")
    
    def add_knowledge_from_text(self, text: str) -> List[str]:
        """
        Extract knowledge from natural language text and add to system (Issue #10)
        
        Args:
            text: Natural language description
            
        Returns:
            List of added concept IDs
        """
        # Extract entities
        extraction = self.nlp.extract_entities(text)
        
        added_concepts = []
        
        # Add entities as concepts
        for entity in extraction.entities:
            properties = extraction.properties.get(entity, {})
            
            self.symbolic_system.add_knowledge(
                entity,
                {
                    'predicate': 'entity',
                    'properties': properties,
                    'source': 'text_extraction'
                }
            )
            added_concepts.append(entity)
        
        # Add relations
        for subject, relation, obj in extraction.relations:
            if subject in self.symbolic_system.kg.nodes:
                subj_node = self.symbolic_system.kg.nodes[subject]
                if relation not in subj_node.relations:
                    subj_node.relations[relation] = set()
                subj_node.relations[relation].add(obj)
        
        return added_concepts
    
    def query_natural_language(self, question: str) -> Dict:
        """
        Answer natural language question using hybrid reasoning (Issues #11, #12)
        
        Complete pipeline:
        1. Parse question → QueryStructure
        2. Map to symbolic concepts
        3. Perform symbolic reasoning
        4. Generate natural language response
        
        Args:
            question: Natural language question
            
        Returns:
            Dict with:
              - question: original question
              - parsed_query: QueryStructure
              - inference_chain: symbolic reasoning
              - answer: natural language response
              - reasoning_trace: detailed reasoning steps
        """
        # Step 1: Parse question (Issue #11)
        parsed = self.nlp.parse_query(question)
        
        print(f"\n[Query Parsing]")
        print(f"  Intent: {parsed.intent}")
        print(f"  Entities: {parsed.entities}")
        print(f"  Start: {parsed.start_concept}, Target: {parsed.target_concept}")
        
        # Step 2: Map to symbolic concepts
        # Ensure entities exist in knowledge graph
        for entity in parsed.entities:
            if entity not in self.symbolic_system.kg.nodes:
                # Add as new concept
                self.symbolic_system.add_knowledge(
                    entity,
                    {
                        'predicate': 'unknown',
                        'properties': {'from_query': True},
                        'source': 'query'
                    }
                )
        
        # Step 3: Symbolic reasoning
        if parsed.start_concept and parsed.target_concept:
            # Relational query
            if (parsed.start_concept in self.symbolic_system.kg.nodes and
                parsed.target_concept in self.symbolic_system.kg.nodes):
                
                print(f"\n[Symbolic Reasoning]")
                print(f"  Querying: {parsed.start_concept} → {parsed.target_concept}")
                
                result = self.symbolic_system.query(
                    question,
                    parsed.start_concept,
                    parsed.target_concept
                )
                
                inference_chain = result['inference_chain']
                
                print(f"  Chain valid: {result['is_valid']}")
                print(f"  Additional inferences: {len(result['additional_inferences'])}")
            else:
                inference_chain = "Concepts not found in knowledge base"
                result = {
                    'inference_chain': inference_chain,
                    'is_valid': False,
                    'additional_inferences': []
                }
        elif parsed.start_concept:
            # Single concept query (definition)
            if parsed.start_concept in self.symbolic_system.kg.nodes:
                concept = self.symbolic_system.kg.nodes[parsed.start_concept]
                inference_chain = f"Concept: {parsed.start_concept}\n"
                inference_chain += f"Properties: {concept.deep_structure.get('properties', {})}\n"
                inference_chain += f"Relations: {list(concept.relations.keys())}"
                result = {
                    'inference_chain': inference_chain,
                    'is_valid': True,
                    'additional_inferences': []
                }
            else:
                inference_chain = f"Concept '{parsed.start_concept}' not found in knowledge base"
                result = {
                    'inference_chain': inference_chain,
                    'is_valid': False,
                    'additional_inferences': []
                }
        else:
            inference_chain = "Could not identify concepts to reason about"
            result = {
                'inference_chain': inference_chain,
                'is_valid': False,
                'additional_inferences': []
            }
        
        # Step 4: Generate natural language response (Issue #12)
        print(f"\n[Response Generation]")
        
        # Create mock inference chain object if needed
        class InferenceChainMock:
            def __init__(self, text):
                self.text = text
            def explain(self):
                return self.text
        
        chain_obj = InferenceChainMock(inference_chain)
        answer = self.nlp.generate_response(chain_obj, question)
        
        print(f"  Generated answer: {answer[:100]}...")
        
        return {
            'question': question,
            'parsed_query': parsed,
            'inference_chain': inference_chain,
            'answer': answer,
            'is_valid': result.get('is_valid', False),
            'additional_inferences': result.get('additional_inferences', []),
            'meta_analysis': result.get('meta_analysis', {})
        }
    
    def add_knowledge(self, concept_id: str, deep_structure: Dict):
        """
        Add knowledge directly (bypassing NLP) for programmatic use
        
        Args:
            concept_id: Concept identifier
            deep_structure: Deep structure dict
        """
        self.symbolic_system.add_knowledge(concept_id, deep_structure)
    
    def load_from_ontology(
        self,
        concept: str,
        sources: List[str] = ['conceptnet'],
        limit: int = 50
    ) -> int:
        """
        Load a concept and related concepts from ontologies (Issue #13)
        
        Args:
            concept: Concept to load
            sources: Ontology sources ('conceptnet', 'dbpedia', 'wikidata')
            limit: Maximum facts per source
            
        Returns:
            Number of concepts added
        """
        print(f"\nLoading '{concept}' from ontologies...")
        
        # Load triples from ontology
        triples = self.ontology.load_concept(concept, sources, limit)
        
        # Convert to MKU format
        mku_data = self.ontology.triples_to_mku_data(triples)
        
        # Add to knowledge graph
        count = 0
        for concept_id, deep_structure in mku_data.items():
            # Convert relation lists back to sets
            relations = {}
            for rel_type, targets in deep_structure.get('relations', {}).items():
                relations[rel_type] = set(targets) if isinstance(targets, list) else targets
            
            # Add concept
            self.symbolic_system.add_knowledge(concept_id, {
                'predicate': deep_structure.get('predicate', 'concept'),
                'properties': deep_structure.get('properties', {}),
                'source': deep_structure.get('source', 'ontology')
            })
            
            # Add relations
            if concept_id in self.symbolic_system.kg.nodes:
                node = self.symbolic_system.kg.nodes[concept_id]
                node.relations.update(relations)
            
            count += 1
        
        print(f"✓ Added {count} concepts from ontologies")
        return count
    
    def learn_from_examples(self, examples: List[Example]) -> int:
        """
        Learn patterns from examples (Issue #14)
        
        Args:
            examples: List of Example objects
            
        Returns:
            Number of patterns learned
        """
        patterns = self.learner.learn_from_examples(examples)
        
        # Store patterns for later use
        print(f"\nLearned {len(patterns)} reusable patterns")
        
        return len(patterns)
    
    def apply_learned_pattern(self, pattern_index: int, variables: Dict, concept_id: str) -> bool:
        """
        Apply a learned pattern to create new concept
        
        Args:
            pattern_index: Index of pattern to apply (0-based)
            variables: Variable values for pattern
            concept_id: ID for new concept
            
        Returns:
            True if successful
        """
        patterns = self.learner.extractor.get_patterns()
        
        if pattern_index >= len(patterns):
            print(f"Pattern {pattern_index} not found")
            return False
        
        pattern = patterns[pattern_index]
        deep_structure = self.learner.apply_pattern(pattern, variables, concept_id)
        
        # Add to knowledge graph
        self.add_knowledge(concept_id, deep_structure)
        
        print(f"✓ Created '{concept_id}' from pattern {pattern.pattern_id}")
        return True
    
    def explain_reasoning(self) -> str:
        """Get explanation of system's reasoning"""
        return self.symbolic_system.explain_reasoning()
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            'total_concepts': len(self.symbolic_system.kg.nodes),
            'total_rules': len(self.symbolic_system.kg.inference_rules),
            'nlp_provider': type(self.nlp.llm).__name__,
            'gpu_enabled': self.symbolic_system.kg.use_gpu
        }


def demo_neurosymbolic():
    """Demonstrate neurosymbolic system"""
    print("=" * 70)
    print("NEUROSYMBOLIC SYSTEM DEMO")
    print("Natural Language + Symbolic Reasoning")
    print("=" * 70)
    print()
    
    # Create system with mock LLM (no API key needed)
    system = NeurosymbolicSystem(use_gpu=False)
    
    print("\n" + "=" * 70)
    print("1. Loading from Ontologies (Issue #13)")
    print("=" * 70)
    
    # Load from ConceptNet
    count = system.load_from_ontology('dog', sources=['conceptnet'], limit=20)
    print(f"\nLoaded {count} concepts about 'dog'")
    
    print("\n" + "=" * 70)
    print("2. Adding Knowledge from Text")
    print("=" * 70)
    
    # Add knowledge from natural language
    text = """
    Dogs are mammals. Mammals are animals. 
    Dogs have four legs and are domesticated.
    Cats are also mammals and are domesticated.
    """
    
    concepts = system.add_knowledge_from_text(text)
    print(f"Extracted and added concepts: {concepts}")
    
    # Add explicit relations
    system.add_knowledge('dog', {
        'predicate': 'is_a',
        'properties': {'legs': 4, 'domesticated': True}
    })
    system.symbolic_system.kg.nodes['dog'].relations['subtype'] = {'mammal'}
    
    system.add_knowledge('mammal', {
        'predicate': 'is_a',
        'properties': {'warm_blooded': True}
    })
    system.symbolic_system.kg.nodes['mammal'].relations['subtype'] = {'animal'}
    
    system.add_knowledge('animal', {
        'predicate': 'living_thing',
        'properties': {'alive': True, 'mobile': True}
    })
    
    print("\n" + "=" * 70)
    print("3. Natural Language Query")
    print("=" * 70)
    
    # Query in natural language
    question = "Is a dog an animal?"
    result = system.query_natural_language(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Valid reasoning: {result['is_valid']}")
    
    print("\n" + "=" * 70)
    print("4. System Statistics")
    print("=" * 70)
    stats = system.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Neurosymbolic demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_neurosymbolic()
