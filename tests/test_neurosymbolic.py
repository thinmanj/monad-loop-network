#!/usr/bin/env python3
"""
Tests for Neurosymbolic System - Phase 2 (Issues #9-12)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nlp_interface import (
    MockLLMProvider,
    NaturalLanguageInterface,
    QueryStructure,
    EntityExtraction,
    create_mock_interface
)
from src.neurosymbolic import NeurosymbolicSystem


def test_mock_llm_provider():
    """Test MockLLMProvider basic functionality"""
    print("Testing MockLLMProvider...")
    
    provider = MockLLMProvider()
    assert provider.is_available()
    
    response = provider.complete("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
    
    print("✓ MockLLMProvider works")


def test_entity_extraction():
    """Test entity extraction from text (Issue #10)"""
    print("\nTesting entity extraction...")
    
    nlp = create_mock_interface()
    
    text = "Dogs are mammals. Cats are also mammals."
    extraction = nlp.extract_entities(text)
    
    assert isinstance(extraction, EntityExtraction)
    assert isinstance(extraction.entities, list)
    assert len(extraction.entities) > 0
    
    print(f"  Extracted entities: {extraction.entities}")
    print("✓ Entity extraction works")


def test_query_parsing():
    """Test query parsing (Issue #11)"""
    print("\nTesting query parsing...")
    
    nlp = create_mock_interface()
    
    question = "Is a Dog an Animal?"
    parsed = nlp.parse_query(question)
    
    assert isinstance(parsed, QueryStructure)
    assert parsed.raw_query == question
    assert parsed.intent in ['question', 'definition', 'comparison', 'explanation']
    assert isinstance(parsed.entities, list)
    
    print(f"  Intent: {parsed.intent}")
    print(f"  Entities: {parsed.entities}")
    print(f"  Start: {parsed.start_concept}, Target: {parsed.target_concept}")
    print("✓ Query parsing works")


def test_response_generation():
    """Test response generation (Issue #12)"""
    print("\nTesting response generation...")
    
    nlp = create_mock_interface()
    
    class MockChain:
        def explain(self):
            return "Step 1: A→B\nStep 2: B→C\nTherefore: A→C"
    
    chain = MockChain()
    response = nlp.generate_response(chain, "Is A related to C?")
    
    assert isinstance(response, str)
    assert len(response) > 0
    
    print(f"  Response: {response[:100]}...")
    print("✓ Response generation works")


def test_neurosymbolic_system_init():
    """Test neurosymbolic system initialization (Issue #9)"""
    print("\nTesting neurosymbolic system...")
    
    system = NeurosymbolicSystem(use_gpu=False)
    
    assert system.symbolic_system is not None
    assert system.nlp is not None
    
    stats = system.get_statistics()
    assert 'total_concepts' in stats
    assert 'total_rules' in stats
    assert 'nlp_provider' in stats
    
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total rules: {stats['total_rules']}")
    print("✓ Neurosymbolic system initializes")


def test_add_knowledge_from_text():
    """Test adding knowledge from natural language"""
    print("\nTesting add_knowledge_from_text...")
    
    system = NeurosymbolicSystem(use_gpu=False)
    
    text = "Dogs are mammals. Cats are mammals."
    concepts = system.add_knowledge_from_text(text)
    
    assert isinstance(concepts, list)
    assert len(concepts) > 0
    
    # Check concepts were added
    for concept in concepts:
        assert concept in system.symbolic_system.kg.nodes
    
    print(f"  Added {len(concepts)} concepts: {concepts}")
    print("✓ Knowledge from text works")


def test_natural_language_query():
    """Test natural language query processing"""
    print("\nTesting natural language query...")
    
    system = NeurosymbolicSystem(use_gpu=False)
    
    # Add some knowledge
    system.add_knowledge('dog', {
        'predicate': 'is_a',
        'properties': {'legs': 4}
    })
    system.add_knowledge('animal', {
        'predicate': 'living_thing',
        'properties': {'alive': True}
    })
    system.symbolic_system.kg.nodes['dog'].relations['subtype'] = {'animal'}
    
    # Query
    question = "Is a dog an animal?"
    result = system.query_natural_language(question)
    
    assert isinstance(result, dict)
    assert 'question' in result
    assert 'answer' in result
    assert 'parsed_query' in result
    assert 'inference_chain' in result
    
    assert result['question'] == question
    assert isinstance(result['answer'], str)
    assert len(result['answer']) > 0
    
    print(f"  Question: {result['question']}")
    print(f"  Answer: {result['answer'][:80]}...")
    print("✓ Natural language query works")


def test_integration_pipeline():
    """Test complete neurosymbolic pipeline"""
    print("\nTesting complete pipeline...")
    
    system = NeurosymbolicSystem(use_gpu=False)
    
    # 1. Add knowledge from text
    text = "A dog is a mammal. A mammal is an animal."
    system.add_knowledge_from_text(text)
    
    # 2. Add structured knowledge
    system.add_knowledge('dog', {'predicate': 'animal_type'})
    system.add_knowledge('mammal', {'predicate': 'animal_class'})
    system.add_knowledge('animal', {'predicate': 'living'})
    
    # Add relations
    if 'dog' in system.symbolic_system.kg.nodes:
        system.symbolic_system.kg.nodes['dog'].relations['subtype'] = {'mammal'}
    if 'mammal' in system.symbolic_system.kg.nodes:
        system.symbolic_system.kg.nodes['mammal'].relations['subtype'] = {'animal'}
    
    # 3. Query in natural language
    result = system.query_natural_language("Is a Dog related to an Animal?")
    
    # 4. Verify result
    assert result is not None
    assert result['is_valid'] or len(result['inference_chain']) > 0
    
    print("  ✓ Text extraction → Knowledge graph → Query → Response")
    print("✓ Integration pipeline works")


def run_all_tests():
    """Run all neurosymbolic tests"""
    print("=" * 70)
    print("Phase 2: Neurosymbolic System Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_mock_llm_provider,
        test_entity_extraction,
        test_query_parsing,
        test_response_generation,
        test_neurosymbolic_system_init,
        test_add_knowledge_from_text,
        test_natural_language_query,
        test_integration_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    print()
    print("Phase 2 Summary:")
    print("  ✓ Issue #9:  Hybrid architecture designed")
    print("  ✓ Issue #10: Entity extraction from text")
    print("  ✓ Issue #11: Natural language query parsing")
    print("  ✓ Issue #12: Response generation from reasoning")
    print("  ✓ Complete NLP → Symbolic → NLP pipeline")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
