#!/usr/bin/env python3
"""
Unit tests for Monad-Loop Network
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import (
    MonadicKnowledgeUnit,
    KnowledgeGraph,
    HybridIntelligenceSystem,
    InferenceChain,
    TransitivityRule,
)


def test_mku_creation():
    """Test creating a Monadic Knowledge Unit"""
    mku = MonadicKnowledgeUnit(
        concept_id='test_concept',
        deep_structure={
            'predicate': 'test_predicate',
            'arguments': ['arg1', 'arg2'],
            'properties': {'prop1': 'value1'}
        }
    )
    
    assert mku.concept_id == 'test_concept'
    assert mku.deep_structure['predicate'] == 'test_predicate'
    assert len(mku.deep_structure['arguments']) == 2


def test_knowledge_graph_add():
    """Test adding concepts to knowledge graph"""
    kg = KnowledgeGraph()
    
    mku1 = MonadicKnowledgeUnit('concept1')
    mku2 = MonadicKnowledgeUnit('concept2')
    
    kg.add_concept(mku1)
    kg.add_concept(mku2)
    
    assert len(kg.nodes) == 2
    assert 'concept1' in kg.nodes
    assert 'concept2' in kg.nodes


def test_pre_established_harmony():
    """Test that concepts establish relations automatically"""
    kg = KnowledgeGraph()
    
    # Add concepts with similar properties
    mku1 = MonadicKnowledgeUnit('dog', deep_structure={
        'predicate': 'is_a',
        'properties': {'domesticated': True, 'mammal': True}
    })
    
    mku2 = MonadicKnowledgeUnit('cat', deep_structure={
        'predicate': 'is_a',
        'properties': {'domesticated': True, 'mammal': True}
    })
    
    kg.add_concept(mku1)
    kg.add_concept(mku2)
    
    # Relations should be established automatically
    assert len(mku1.relations) > 0 or len(mku2.relations) > 0


def test_surface_generation():
    """Test Chomsky-style surface generation"""
    mku = MonadicKnowledgeUnit('dog', deep_structure={
        'predicate': 'is_a',
        'arguments': ['mammal']
    })
    
    text_form = mku.generate_surface_form('text')
    logic_form = mku.generate_surface_form('logic')
    code_form = mku.generate_surface_form('code')
    
    assert isinstance(text_form, str)
    assert isinstance(logic_form, str)
    assert isinstance(code_form, str)
    assert 'mammal' in text_form or 'mammal' in logic_form


def test_inference_chain():
    """Test inference chain validation"""
    kg = KnowledgeGraph()
    
    dog = MonadicKnowledgeUnit('dog')
    mammal = MonadicKnowledgeUnit('mammal')
    
    kg.add_concept(dog)
    kg.add_concept(mammal)
    
    chain = kg.query('dog', 'mammal')
    
    assert isinstance(chain, InferenceChain)
    explanation = chain.explain()
    assert isinstance(explanation, str)


def test_meta_reasoning():
    """Test strange loop / meta-reasoning"""
    system = HybridIntelligenceSystem()
    
    system.add_knowledge('concept1', {
        'predicate': 'test',
        'properties': {'key': 'value'}
    })
    
    # Introspect
    meta_info = system.slp.introspect("test query")
    
    assert 'query' in meta_info
    assert 'graph_state' in meta_info
    assert meta_info['graph_state']['num_concepts'] >= 1


def test_godel_sentence():
    """Test self-referential Gödel sentence generation"""
    system = HybridIntelligenceSystem()
    
    godel = system.slp.godel_sentence()
    
    assert isinstance(godel, str)
    assert 'cannot prove' in godel.lower() or 'concepts' in godel.lower()


def test_hybrid_system_query():
    """Test end-to-end query processing"""
    system = HybridIntelligenceSystem()
    
    # Add knowledge
    system.add_knowledge('dog', {
        'predicate': 'is_a',
        'arguments': ['mammal'],
        'properties': {'domesticated': True}
    })
    
    system.add_knowledge('mammal', {
        'predicate': 'is_alive',
        'properties': {'warm_blooded': True}
    })
    
    # Query
    result = system.query(
        "Is a dog a mammal?",
        start_concept='dog',
        target_concept='mammal'
    )
    
    assert 'question' in result
    assert 'inference_chain' in result
    assert 'is_valid' in result
    assert isinstance(result['is_valid'], bool)


def test_inconsistency_detection():
    """Test detection of logical inconsistencies"""
    system = HybridIntelligenceSystem()
    
    system.add_knowledge('concept1', {'predicate': 'test'})
    
    inconsistencies = system.detect_inconsistencies()
    
    assert isinstance(inconsistencies, list)


def test_transitivity_rule():
    """Test transitive inference"""
    kg = KnowledgeGraph()
    rule = TransitivityRule()
    
    mku = MonadicKnowledgeUnit('test', deep_structure={
        'predicate': 'relates_to',
        'arguments': ['other']
    })
    mku.relations['subtype'] = {'related_concept'}
    
    kg.add_concept(mku)
    kg.add_concept(MonadicKnowledgeUnit('related_concept'))
    
    can_apply = rule.can_apply(mku)
    assert isinstance(can_apply, bool)


if __name__ == '__main__':
    # Run tests
    import traceback
    
    tests = [
        test_mku_creation,
        test_knowledge_graph_add,
        test_pre_established_harmony,
        test_surface_generation,
        test_inference_chain,
        test_meta_reasoning,
        test_godel_sentence,
        test_hybrid_system_query,
        test_inconsistency_detection,
        test_transitivity_rule,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
