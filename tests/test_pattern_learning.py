#!/usr/bin/env python3
"""Tests for pattern learning - Issue #14"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pattern_learner import ExampleBasedLearner, Example


def test_pattern_extraction():
    """Test basic pattern extraction"""
    print("Testing pattern extraction...")
    
    learner = ExampleBasedLearner()
    
    examples = [
        Example("dog", "dog", {'predicate': 'animal', 'legs': 4}, {}),
        Example("cat", "cat", {'predicate': 'animal', 'legs': 4}, {}),
        Example("horse", "horse", {'predicate': 'animal', 'legs': 4}, {}),
    ]
    
    patterns = learner.learn_from_examples(examples)
    
    assert len(patterns) > 0
    assert patterns[0].frequency == 3
    
    print("✓ Pattern extraction works")


def test_pattern_application():
    """Test applying learned patterns"""
    print("\nTesting pattern application...")
    
    learner = ExampleBasedLearner()
    
    examples = [
        Example("dog", "dog", {'predicate': 'animal', 'type': 'mammal'}, {}),
        Example("cat", "cat", {'predicate': 'animal', 'type': 'mammal'}, {}),
    ]
    
    patterns = learner.learn_from_examples(examples)
    
    if patterns:
        new_concept = learner.apply_pattern(
            patterns[0],
            {'type': 'mammal'},
            'cow'
        )
        
        assert 'predicate' in new_concept
        assert new_concept['predicate'] == 'animal'
    
    print("✓ Pattern application works")


def run_tests():
    print("=" * 70)
    print("Pattern Learning Tests - Issue #14")
    print("=" * 70)
    print()
    
    test_pattern_extraction()
    test_pattern_application()
    
    print("\n" + "=" * 70)
    print("✓ All pattern learning tests passed")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
