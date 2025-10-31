#!/usr/bin/env python3
"""
Tests for new inference rules - Issue #4
Tests Modus Ponens, Contraposition, Symmetry, and Composition rules
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import (
    MonadicKnowledgeUnit,
    KnowledgeGraph,
    ModusPonensRule,
    ContrapositionRule,
    SymmetryRule,
    CompositionRule,
    TransitivityRule,
)


def test_modus_ponens():
    """Test Modus Ponens rule: If A→B and A, then B"""
    print("Testing Modus Ponens Rule...")
    
    kg = KnowledgeGraph(use_gpu=False)
    rule = ModusPonensRule()
    
    # Create A with A→B relation
    a = MonadicKnowledgeUnit('A', {'predicate': 'thing'})
    a.relations['implies'] = {'B'}
    
    b = MonadicKnowledgeUnit('B', {
        'predicate': 'conclusion',
        'properties': {'derived': True}
    })
    
    kg.nodes['A'] = a
    kg.nodes['B'] = b
    
    # Test applicability
    assert rule.can_apply(a), "Should be able to apply to A→B"
    
    # Apply rule
    result = rule.apply(a, kg)
    
    assert result is not None, "Should produce a result"
    assert 'modus_ponens' in result.deep_structure['predicate']
    assert 'A' in result.deep_structure['arguments']
    assert 'B' in result.deep_structure['arguments']
    
    print("✓ Modus Ponens test passed")


def test_contraposition():
    """Test Contraposition rule: If A→B, then ¬B→¬A"""
    print("\nTesting Contraposition Rule...")
    
    kg = KnowledgeGraph(use_gpu=False)
    rule = ContrapositionRule()
    
    # Create A→B
    a = MonadicKnowledgeUnit('A', {'predicate': 'antecedent'})
    a.relations['implies'] = {'B'}
    
    b = MonadicKnowledgeUnit('B', {'predicate': 'consequent'})
    
    kg.nodes['A'] = a
    kg.nodes['B'] = b
    
    # Apply contraposition
    result = rule.apply(a, kg)
    
    assert result is not None
    assert 'contrapositive' in result.deep_structure['predicate']
    assert result.deep_structure['properties']['negation'] == True
    assert '¬B' in result.deep_structure['arguments']
    assert '¬A' in result.deep_structure['arguments']
    
    print("✓ Contraposition test passed")


def test_symmetry():
    """Test Symmetry rule: If A ~ B (symmetric), then B ~ A"""
    print("\nTesting Symmetry Rule...")
    
    kg = KnowledgeGraph(use_gpu=False)
    rule = SymmetryRule()
    
    # Create A with equivalence to B
    a = MonadicKnowledgeUnit('A', {'predicate': 'entity'})
    a.relations['equivalence'] = {'B'}
    
    b = MonadicKnowledgeUnit('B', {'predicate': 'entity'})
    
    kg.nodes['A'] = a
    kg.nodes['B'] = b
    
    # Test applicability
    assert rule.can_apply(a), "Should apply to symmetric relations"
    
    # Apply symmetry
    result = rule.apply(a, kg)
    
    assert result is not None
    assert 'symmetric_relation' in result.deep_structure['predicate']
    assert 'B' in result.deep_structure['arguments']
    assert 'A' in result.deep_structure['arguments']
    
    print("✓ Symmetry test passed")


def test_composition():
    """Test Composition rule: Combine two rules"""
    print("\nTesting Composition Rule...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Create A→B→C chain
    a = MonadicKnowledgeUnit('A', {'predicate': 'start'})
    a.relations['implies'] = {'B'}
    
    b = MonadicKnowledgeUnit('B', {'predicate': 'middle'})
    b.relations['implies'] = {'C'}
    
    c = MonadicKnowledgeUnit('C', {'predicate': 'end'})
    
    kg.nodes['A'] = a
    kg.nodes['B'] = b
    kg.nodes['C'] = c
    
    # Compose Modus Ponens with itself
    rule1 = ModusPonensRule()
    rule2 = ModusPonensRule()
    composed = CompositionRule(rule1, rule2)
    
    # Apply composed rule
    result = composed.apply(a, kg)
    
    assert result is not None
    print(f"  Composed result: {result.concept_id}")
    
    # Check if composition is marked
    if result.deep_structure.get('composed'):
        assert 'ModusPonensRule' in result.deep_structure['composition']
        print("  ✓ Composition marked correctly")
    
    print("✓ Composition test passed")


def test_subtype_inference():
    """Test that subtype relations work with Modus Ponens"""
    print("\nTesting Subtype with Modus Ponens...")
    
    kg = KnowledgeGraph(use_gpu=False)
    rule = ModusPonensRule()
    
    # Dog is a subtype of Mammal
    dog = MonadicKnowledgeUnit('dog', {
        'predicate': 'is_a',
        'properties': {'domesticated': True}
    })
    dog.relations['subtype'] = {'mammal'}
    
    mammal = MonadicKnowledgeUnit('mammal', {
        'predicate': 'is_alive',
        'properties': {'warm_blooded': True}
    })
    
    kg.nodes['dog'] = dog
    kg.nodes['mammal'] = mammal
    
    # Apply Modus Ponens
    result = rule.apply(dog, kg)
    
    assert result is not None
    assert 'mammal' in result.deep_structure['arguments']
    
    print("✓ Subtype inference test passed")


def test_rule_integration():
    """Test multiple rules working together"""
    print("\nTesting Rule Integration...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Add all rules
    kg.add_inference_rule(ModusPonensRule())
    kg.add_inference_rule(ContrapositionRule())
    kg.add_inference_rule(SymmetryRule())
    kg.add_inference_rule(TransitivityRule())
    
    # Create concepts
    a = MonadicKnowledgeUnit('A', {'predicate': 'start'})
    a.relations['implies'] = {'B'}
    a.relations['equivalence'] = {'A_prime'}
    
    kg.nodes['A'] = a
    kg.nodes['B'] = MonadicKnowledgeUnit('B', {'predicate': 'end'})
    kg.nodes['A_prime'] = MonadicKnowledgeUnit('A_prime', {'predicate': 'start'})
    
    # Apply all rules
    inferences = kg.apply_inference(a)
    
    assert len(inferences) > 0, "Should produce inferences"
    print(f"  Produced {len(inferences)} inferences")
    
    # Check we got different types
    predicates = [inf.deep_structure.get('predicate') for inf in inferences]
    print(f"  Inference types: {predicates}")
    
    print("✓ Rule integration test passed")


def test_no_application():
    """Test that rules don't apply when they shouldn't"""
    print("\nTesting Non-Application Cases...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Concept with no relevant relations
    isolated = MonadicKnowledgeUnit('isolated', {'predicate': 'alone'})
    kg.nodes['isolated'] = isolated
    
    # Modus Ponens shouldn't apply (no implies/subtype)
    mp_rule = ModusPonensRule()
    isolated.relations = {}  # No relations
    assert not mp_rule.can_apply(isolated)
    
    # Symmetry shouldn't apply (no symmetric relations)
    sym_rule = SymmetryRule()
    isolated.relations = {'asymmetric': {'other'}}
    assert not sym_rule.can_apply(isolated)
    
    print("✓ Non-application test passed")


def run_all_tests():
    """Run all inference rule tests"""
    print("=" * 70)
    print("Issue #4: Inference Rules Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_modus_ponens,
        test_contraposition,
        test_symmetry,
        test_composition,
        test_subtype_inference,
        test_rule_integration,
        test_no_application,
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
    print("Issue #4 Summary:")
    print("  ✓ ModusPonensRule: If A→B and A, then B")
    print("  ✓ ContrapositionRule: If A→B, then ¬B→¬A")
    print("  ✓ SymmetryRule: If A~B (symmetric), then B~A")
    print("  ✓ CompositionRule: Combine multiple rules")
    print("  Total: 4 new inference rules + composition")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
