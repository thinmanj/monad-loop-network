#!/usr/bin/env python3
"""
Pattern Learner - Issue #14
Learn from examples and generalize patterns into new MKUs

Extracts deep structures from examples and creates reusable patterns
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class Example:
    """A single training example"""
    input_text: str
    concept_id: str
    deep_structure: Dict
    relations: Dict[str, Set[str]]


@dataclass
class Pattern:
    """An extracted pattern that can be generalized"""
    pattern_id: str
    template: Dict  # Template with placeholders
    variables: List[str]  # Variable names
    examples: List[Example]
    frequency: int = 1
    confidence: float = 1.0


class PatternExtractor:
    """
    Extract patterns from examples
    Uses structural similarity to find common patterns
    """
    
    def __init__(self, min_examples: int = 2, min_confidence: float = 0.6):
        """
        Args:
            min_examples: Minimum examples needed to form a pattern
            min_confidence: Minimum confidence for pattern extraction
        """
        self.min_examples = min_examples
        self.min_confidence = min_confidence
        self.patterns: Dict[str, Pattern] = {}
    
    def add_example(self, example: Example):
        """Add a training example"""
        # Find similar patterns
        matching_pattern = self._find_matching_pattern(example)
        
        if matching_pattern:
            # Add to existing pattern
            matching_pattern.examples.append(example)
            matching_pattern.frequency += 1
            matching_pattern.confidence = self._calculate_confidence(matching_pattern)
        else:
            # Create new pattern
            pattern = self._create_pattern_from_example(example)
            self.patterns[pattern.pattern_id] = pattern
    
    def _find_matching_pattern(self, example: Example) -> Optional[Pattern]:
        """Find a pattern that matches this example"""
        for pattern in self.patterns.values():
            if self._matches_pattern(example, pattern):
                return pattern
        return None
    
    def _matches_pattern(self, example: Example, pattern: Pattern) -> bool:
        """Check if example matches pattern template"""
        # Compare predicates
        if example.deep_structure.get('predicate') != pattern.template.get('predicate'):
            return False
        
        # Compare structure (flexible matching)
        example_keys = set(example.deep_structure.keys())
        pattern_keys = set(pattern.template.keys())
        
        # At least 70% key overlap
        overlap = len(example_keys & pattern_keys) / len(example_keys | pattern_keys)
        return overlap >= 0.7
    
    def _create_pattern_from_example(self, example: Example) -> Pattern:
        """Create a new pattern from an example"""
        pattern_id = f"pattern_{len(self.patterns) + 1}_{example.concept_id}"
        
        # Create template (copy structure, identify variables later)
        template = example.deep_structure.copy()
        
        return Pattern(
            pattern_id=pattern_id,
            template=template,
            variables=[],
            examples=[example],
            frequency=1,
            confidence=1.0
        )
    
    def _calculate_confidence(self, pattern: Pattern) -> float:
        """Calculate confidence based on example frequency and consistency"""
        # More examples = higher confidence
        frequency_score = min(pattern.frequency / 10.0, 1.0)
        
        # Check consistency across examples
        consistency = self._check_consistency(pattern)
        
        return (frequency_score + consistency) / 2.0
    
    def _check_consistency(self, pattern: Pattern) -> float:
        """Check how consistent examples are with the pattern"""
        if len(pattern.examples) < 2:
            return 1.0
        
        # Count matching keys across examples
        all_keys = []
        for example in pattern.examples:
            all_keys.extend(example.deep_structure.keys())
        
        key_counts = Counter(all_keys)
        avg_consistency = sum(count / len(pattern.examples) for count in key_counts.values()) / len(key_counts)
        
        return min(avg_consistency, 1.0)
    
    def get_patterns(self, min_frequency: int = None) -> List[Pattern]:
        """Get learned patterns"""
        patterns = list(self.patterns.values())
        
        if min_frequency:
            patterns = [p for p in patterns if p.frequency >= min_frequency]
        
        # Sort by frequency and confidence
        patterns.sort(key=lambda p: (p.frequency, p.confidence), reverse=True)
        return patterns
    
    def generalize_pattern(self, pattern: Pattern) -> Dict:
        """
        Generalize a pattern by identifying variables
        
        Returns:
            Generalized template with variables marked as {var_name}
        """
        if len(pattern.examples) < 2:
            return pattern.template
        
        # Find varying properties across examples
        varying_props = self._find_varying_properties(pattern)
        
        # Create generalized template
        generalized = pattern.template.copy()
        
        for prop in varying_props:
            if prop in generalized:
                generalized[prop] = f"{{var_{prop}}}"
        
        return generalized
    
    def _find_varying_properties(self, pattern: Pattern) -> Set[str]:
        """Find properties that vary across examples"""
        if len(pattern.examples) < 2:
            return set()
        
        varying = set()
        
        # Compare first example with others
        first_example = pattern.examples[0].deep_structure
        
        for key in first_example.keys():
            values = [ex.deep_structure.get(key) for ex in pattern.examples if key in ex.deep_structure]
            
            # If values differ, it's a variable
            if len(set(str(v) for v in values)) > 1:
                varying.add(key)
        
        return varying


class ExampleBasedLearner:
    """
    Learn new concepts from examples
    Combines pattern extraction with concept generation
    """
    
    def __init__(self):
        self.extractor = PatternExtractor()
        self.learned_concepts: Dict[str, Dict] = {}
    
    def learn_from_examples(self, examples: List[Example]) -> List[Pattern]:
        """
        Learn patterns from a list of examples
        
        Args:
            examples: List of training examples
            
        Returns:
            List of learned patterns
        """
        print(f"Learning from {len(examples)} examples...")
        
        # Add all examples to pattern extractor
        for example in examples:
            self.extractor.add_example(example)
        
        # Get learned patterns
        patterns = self.extractor.get_patterns(min_frequency=2)
        
        print(f"Discovered {len(patterns)} patterns")
        
        return patterns
    
    def apply_pattern(
        self,
        pattern: Pattern,
        variables: Dict[str, any],
        concept_id: str
    ) -> Dict:
        """
        Apply a learned pattern with specific variables to create new concept
        
        Args:
            pattern: Learned pattern
            variables: Variable values {var_name: value}
            concept_id: ID for new concept
            
        Returns:
            Deep structure for new concept
        """
        # Get generalized template
        template = self.extractor.generalize_pattern(pattern)
        
        # Substitute variables
        deep_structure = {}
        for key, value in template.items():
            if isinstance(value, str) and value.startswith('{var_') and value.endswith('}'):
                # Extract variable name
                var_name = value[5:-1]  # Remove {var_ and }
                deep_structure[key] = variables.get(var_name, value)
            else:
                deep_structure[key] = value
        
        # Add metadata
        deep_structure['learned_from'] = pattern.pattern_id
        deep_structure['confidence'] = pattern.confidence
        
        return deep_structure
    
    def suggest_similar_concepts(
        self,
        concept: Dict,
        num_suggestions: int = 5
    ) -> List[Tuple[str, Dict, float]]:
        """
        Suggest similar concepts based on learned patterns
        
        Args:
            concept: Concept to base suggestions on
            num_suggestions: Number of suggestions to generate
            
        Returns:
            List of (concept_id, deep_structure, confidence) tuples
        """
        suggestions = []
        
        # Find matching patterns
        for pattern in self.extractor.patterns.values():
            # Check if concept matches pattern
            predicate = concept.get('predicate')
            if predicate == pattern.template.get('predicate'):
                # Generate variations
                for i in range(min(num_suggestions, 3)):
                    new_id = f"suggested_{pattern.pattern_id}_{i}"
                    # Use pattern template as basis
                    deep_structure = pattern.template.copy()
                    deep_structure['suggested'] = True
                    
                    suggestions.append((new_id, deep_structure, pattern.confidence))
        
        return suggestions[:num_suggestions]


def demo_pattern_learning():
    """Demonstrate pattern learning"""
    print("=" * 70)
    print("PATTERN LEARNING DEMO - Issue #14")
    print("=" * 70)
    print()
    
    learner = ExampleBasedLearner()
    
    # Create training examples
    print("1. Creating training examples")
    print("-" * 70)
    
    examples = [
        Example(
            input_text="A dog is a mammal",
            concept_id="dog",
            deep_structure={
                'predicate': 'is_a',
                'category': 'mammal',
                'properties': {'legs': 4, 'domesticated': True}
            },
            relations={'subtype': {'mammal'}}
        ),
        Example(
            input_text="A cat is a mammal",
            concept_id="cat",
            deep_structure={
                'predicate': 'is_a',
                'category': 'mammal',
                'properties': {'legs': 4, 'domesticated': True}
            },
            relations={'subtype': {'mammal'}}
        ),
        Example(
            input_text="A horse is a mammal",
            concept_id="horse",
            deep_structure={
                'predicate': 'is_a',
                'category': 'mammal',
                'properties': {'legs': 4, 'domesticated': True}
            },
            relations={'subtype': {'mammal'}}
        ),
        Example(
            input_text="A bird is an animal",
            concept_id="bird",
            deep_structure={
                'predicate': 'is_a',
                'category': 'animal',
                'properties': {'legs': 2, 'can_fly': True}
            },
            relations={'subtype': {'animal'}}
        ),
    ]
    
    print(f"Created {len(examples)} examples")
    
    # Learn patterns
    print("\n2. Learning patterns from examples")
    print("-" * 70)
    patterns = learner.learn_from_examples(examples)
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\nPattern {i}: {pattern.pattern_id}")
        print(f"  Frequency: {pattern.frequency}")
        print(f"  Confidence: {pattern.confidence:.2f}")
        print(f"  Template predicate: {pattern.template.get('predicate')}")
        print(f"  Example concepts: {[ex.concept_id for ex in pattern.examples[:3]]}")
    
    # Generalize patterns
    print("\n3. Generalizing patterns")
    print("-" * 70)
    if patterns:
        pattern = patterns[0]
        generalized = learner.extractor.generalize_pattern(pattern)
        print(f"\nGeneralized template for {pattern.pattern_id}:")
        for key, value in generalized.items():
            print(f"  {key}: {value}")
    
    # Apply pattern to create new concept
    print("\n4. Applying pattern to create new concept")
    print("-" * 70)
    if patterns:
        new_concept = learner.apply_pattern(
            patterns[0],
            variables={'category': 'mammal', 'legs': 4},
            concept_id='cow'
        )
        print("\nNew concept 'cow' created from pattern:")
        for key, value in new_concept.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Pattern learning demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_pattern_learning()
