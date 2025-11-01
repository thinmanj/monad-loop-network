#!/usr/bin/env python3
"""
Tests for Issue #5: Rule Priorities
Tests that inference rules are applied in priority order
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
    TransitivityRule,
    SubstitutionRule,
    CompositionRule,
)


def test_rule_priority_assignment():
    """Test that rules have correct priorities"""
    print("Testing rule priority assignment...")
    
    mp = ModusPonensRule()
    trans = TransitivityRule()
    sym = SymmetryRule()
    contra = ContrapositionRule()
    subst = SubstitutionRule()
    
    assert mp.priority == 0, "Modus Ponens should be highest priority (0)"
    assert trans.priority == 1, "Transitivity should be high priority (1)"
    assert subst.priority == 1, "Substitution should be high priority (1)"
    assert sym.priority == 2, "Symmetry should be medium priority (2)"
    assert contra.priority == 3, "Contraposition should be low priority (3)"
    
    print("✓ Rule priorities assigned correctly")


def test_rule_sorting():
    """Test that rules are sorted by priority when added"""
    print("\nTesting rule sorting...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Add rules in random order
    kg.add_inference_rule(ContrapositionRule())  # Priority 3
    kg.add_inference_rule(SymmetryRule())        # Priority 2
    kg.add_inference_rule(ModusPonensRule())     # Priority 0
    kg.add_inference_rule(TransitivityRule())    # Priority 1
    
    # Check they're sorted by priority
    priorities = [rule.priority for rule in kg.inference_rules]
    assert priorities == [0, 1, 2, 3], f"Rules should be sorted: [0,1,2,3], got {priorities}"
    
    print(f"  Rules sorted by priority: {priorities}")
    print("✓ Rules sorted correctly")


def test_priority_execution_order():
    """Test that high-priority rules execute before low-priority rules"""
    print("\nTesting execution order...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Add rules in reverse priority order
    kg.add_inference_rule(ContrapositionRule())  # Priority 3 - should execute last
    kg.add_inference_rule(SymmetryRule())        # Priority 2
    kg.add_inference_rule(TransitivityRule())    # Priority 1
    kg.add_inference_rule(ModusPonensRule())     # Priority 0 - should execute first
    
    # Create concept with multiple applicable rules
    a = MonadicKnowledgeUnit('A', {'predicate': 'test'})
    a.relations['implies'] = {'B'}      # Triggers ModusPonens + Contraposition
    a.relations['equivalence'] = {'C'}  # Triggers Symmetry (symmetric relation)
    
    kg.add_concept(a)
    kg.add_concept(MonadicKnowledgeUnit('B', {'predicate': 'test'}))
    kg.add_concept(MonadicKnowledgeUnit('C', {'predicate': 'test'}))
    
    # Apply inference - should be in priority order
    inferences = kg.apply_inference(a)
    
    # First inference should be from ModusPonens (priority 0)
    assert len(inferences) > 0, "Should produce inferences"
    assert 'modus_ponens' in inferences[0].deep_structure.get('predicate', '')
    
    print(f"  First inference (highest priority): {inferences[0].deep_structure['predicate']}")
    print(f"  Total inferences: {len(inferences)}")
    print("✓ High-priority rules execute first")


def test_composition_priority():
    """Test that CompositionRule has lower priority than its components"""
    print("\nTesting composition priority...")
    
    mp = ModusPonensRule()      # Priority 0
    trans = TransitivityRule()  # Priority 1
    
    # Composition should have priority max(0,1) + 1 = 2
    comp = CompositionRule(mp, trans)
    
    assert comp.priority == 2, f"Composition should have priority 2, got {comp.priority}"
    assert comp.priority > mp.priority, "Composition should have lower priority than components"
    assert comp.priority > trans.priority, "Composition should have lower priority than components"
    
    print(f"  ModusPonens: priority {mp.priority}")
    print(f"  Transitivity: priority {trans.priority}")
    print(f"  Composition: priority {comp.priority}")
    print("✓ Composition has correct priority")


def test_custom_priority_rule():
    """Test creating a rule with custom priority"""
    print("\nTesting custom priority...")
    
    from src.mln import InferenceRule, Optional
    
    class CustomRule(InferenceRule):
        """Custom rule with priority 5"""
        def __init__(self):
            super().__init__(priority=5)
        
        def can_apply(self, premise):
            return True
        
        def apply(self, premise, kg):
            return None
    
    custom = CustomRule()
    assert custom.priority == 5, "Custom rule should have priority 5"
    
    kg = KnowledgeGraph(use_gpu=False)
    kg.add_inference_rule(ModusPonensRule())  # Priority 0
    kg.add_inference_rule(custom)              # Priority 5
    
    priorities = [r.priority for r in kg.inference_rules]
    assert priorities == [0, 5], f"Should be sorted [0,5], got {priorities}"
    
    print(f"  Custom rule priority: {custom.priority}")
    print(f"  Sorted priorities: {priorities}")
    print("✓ Custom priorities work correctly")


def test_priority_comparison():
    """Test rule comparison operators"""
    print("\nTesting rule comparison...")
    
    high = ModusPonensRule()     # Priority 0
    medium = SymmetryRule()      # Priority 2
    low = ContrapositionRule()   # Priority 3
    
    # Test __lt__ operator
    assert high < medium, "High priority should be < medium priority"
    assert medium < low, "Medium priority should be < low priority"
    assert high < low, "High priority should be < low priority"
    
    # Test sorting
    rules = [low, high, medium]
    rules.sort()
    assert rules == [high, medium, low], "Rules should sort by priority"
    
    print("  ✓ High priority (0) < Medium priority (2)")
    print("  ✓ Medium priority (2) < Low priority (3)")
    print("  ✓ Sorting works correctly")
    print("✓ Rule comparison operators work")


def run_all_tests():
    """Run all priority tests"""
    print("=" * 70)
    print("Issue #5: Rule Priorities Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_rule_priority_assignment,
        test_rule_sorting,
        test_priority_execution_order,
        test_composition_priority,
        test_custom_priority_rule,
        test_priority_comparison,
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
    print("Issue #5 Summary:")
    print("  ✓ Priority levels: 0 (Highest) → 4+ (Lowest)")
    print("  ✓ ModusPonens: Priority 0 (Direct logical inference)")
    print("  ✓ Transitivity/Substitution: Priority 1 (Structural reasoning)")
    print("  ✓ Symmetry: Priority 2 (Bidirectional relations)")
    print("  ✓ Contraposition: Priority 3 (Derived implications)")
    print("  ✓ Composition: Priority = max(components) + 1")
    print("  ✓ Rules automatically sorted on add")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
