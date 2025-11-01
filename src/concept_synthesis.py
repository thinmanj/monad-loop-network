#!/usr/bin/env python3
"""
Concept Synthesis - Issue #21
Synthesizes NEW concepts from examples using abductive learning

Phase 4: Self-Improvement - Creative capability (CRITICAL)
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import hashlib


@dataclass
class ConceptExample:
    """
    An example instance of a concept
    Used for learning generalizations
    """
    example_id: str
    properties: Dict[str, Any]
    relations: Dict[str, Set[str]] = field(default_factory=dict)
    positive: bool = True  # Is this a positive or negative example?
    
    def get_property_keys(self) -> Set[str]:
        """Get all property keys"""
        return set(self.properties.keys())
    
    def get_relation_types(self) -> Set[str]:
        """Get all relation types"""
        return set(self.relations.keys())


@dataclass
class SynthesizedConcept:
    """
    A newly synthesized concept created from examples
    Ready to be converted into a MonadicKnowledgeUnit
    """
    concept_id: str
    confidence: float  # 0.0 to 1.0
    
    # Generalized structure
    common_properties: Dict[str, Any]  # Properties shared by all examples
    typical_properties: Dict[str, Any]  # Properties in >50% of examples
    common_relations: Dict[str, Set[str]]  # Relations shared by examples
    
    # Meta-information
    source_examples: List[ConceptExample] = field(default_factory=list)
    abstraction_level: int = 0  # Higher = more abstract
    parent_concepts: Set[str] = field(default_factory=set)
    
    def to_mku_structure(self) -> Dict[str, Any]:
        """Convert to MonadicKnowledgeUnit structure"""
        return {
            'concept_id': self.concept_id,
            'deep_structure': {
                'predicate': self._infer_predicate(),
                'properties': self.common_properties,
                'typical_properties': self.typical_properties,
                'constraints': self._extract_constraints()
            },
            'relations': self.common_relations,
            'confidence': self.confidence
        }
    
    def _infer_predicate(self) -> str:
        """Infer predicate from concept structure"""
        # Simple heuristic: use concept_id as predicate
        return self.concept_id.replace('_', ' ')
    
    def _extract_constraints(self) -> List[str]:
        """Extract constraints from properties"""
        constraints = []
        
        # Type constraints
        for prop, value in self.common_properties.items():
            if isinstance(value, bool):
                constraints.append(f"{prop} is boolean")
            elif isinstance(value, (int, float)):
                constraints.append(f"{prop} is numeric")
            elif isinstance(value, str):
                constraints.append(f"{prop} is string")
        
        return constraints


class ConceptSynthesizer:
    """
    Synthesizes new concepts from examples (Issue #21)
    Implements abductive learning
    """
    
    def __init__(
        self,
        min_examples: int = 3,
        min_confidence: float = 0.6,
        commonality_threshold: float = 0.7
    ):
        """
        Args:
            min_examples: Minimum examples needed to synthesize
            min_confidence: Minimum confidence to accept synthesis
            commonality_threshold: Properties must appear in this % of examples
        """
        self.min_examples = min_examples
        self.min_confidence = min_confidence
        self.commonality_threshold = commonality_threshold
    
    def synthesize_concept(
        self,
        examples: List[ConceptExample],
        concept_name: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Optional[SynthesizedConcept]:
        """
        Synthesize a new concept from examples
        
        Args:
            examples: List of example instances
            concept_name: Optional name for the concept
            context: Optional knowledge graph context
            
        Returns:
            SynthesizedConcept if successful, None otherwise
        """
        # Validate input
        if len(examples) < self.min_examples:
            return None
        
        # Separate positive and negative examples
        positive_examples = [e for e in examples if e.positive]
        negative_examples = [e for e in examples if not e.positive]
        
        if not positive_examples:
            return None
        
        # Extract common structure
        common_props = self._extract_common_properties(positive_examples)
        typical_props = self._extract_typical_properties(positive_examples)
        common_rels = self._extract_common_relations(positive_examples)
        
        # Filter out negative example properties
        if negative_examples:
            common_props = self._filter_negative_properties(
                common_props, negative_examples
            )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            positive_examples, negative_examples,
            common_props, typical_props
        )
        
        if confidence < self.min_confidence:
            return None
        
        # Generate concept name if not provided
        if concept_name is None:
            concept_name = self._generate_concept_name(common_props, typical_props)
        
        # Infer parent concepts
        parents = self._infer_parent_concepts(common_props, context)
        
        # Create synthesized concept
        concept = SynthesizedConcept(
            concept_id=concept_name,
            confidence=confidence,
            common_properties=common_props,
            typical_properties=typical_props,
            common_relations=common_rels,
            source_examples=examples,
            parent_concepts=parents
        )
        
        return concept
    
    def _extract_common_properties(
        self,
        examples: List[ConceptExample]
    ) -> Dict[str, Any]:
        """Extract properties common to ALL examples"""
        if not examples:
            return {}
        
        # Start with first example's properties
        common = dict(examples[0].properties)
        
        # Keep only properties shared by all examples
        for example in examples[1:]:
            # Find intersection
            common_keys = set(common.keys()) & set(example.properties.keys())
            
            # Keep only matching values
            filtered = {}
            for key in common_keys:
                if common[key] == example.properties[key]:
                    filtered[key] = common[key]
            
            common = filtered
        
        return common
    
    def _extract_typical_properties(
        self,
        examples: List[ConceptExample]
    ) -> Dict[str, Any]:
        """Extract properties present in >threshold% of examples"""
        if not examples:
            return {}
        
        # Count property occurrences
        property_counts = defaultdict(lambda: defaultdict(int))
        
        for example in examples:
            for key, value in example.properties.items():
                property_counts[key][value] += 1
        
        # Keep properties above threshold
        typical = {}
        threshold_count = len(examples) * self.commonality_threshold
        
        for key, value_counts in property_counts.items():
            # Find most common value
            most_common_value = max(value_counts.items(), key=lambda x: x[1])
            value, count = most_common_value
            
            if count >= threshold_count:
                typical[key] = value
        
        return typical
    
    def _extract_common_relations(
        self,
        examples: List[ConceptExample]
    ) -> Dict[str, Set[str]]:
        """Extract relations common to examples"""
        if not examples:
            return {}
        
        # Count relation occurrences
        relation_counts = defaultdict(lambda: Counter())
        
        for example in examples:
            for rel_type, targets in example.relations.items():
                for target in targets:
                    relation_counts[rel_type][target] += 1
        
        # Keep relations above threshold
        common = {}
        threshold_count = len(examples) * self.commonality_threshold
        
        for rel_type, target_counts in relation_counts.items():
            common_targets = {
                target for target, count in target_counts.items()
                if count >= threshold_count
            }
            if common_targets:
                common[rel_type] = common_targets
        
        return common
    
    def _filter_negative_properties(
        self,
        properties: Dict[str, Any],
        negative_examples: List[ConceptExample]
    ) -> Dict[str, Any]:
        """Remove properties that appear in negative examples"""
        filtered = {}
        
        for key, value in properties.items():
            # Check if this property appears in negative examples
            appears_in_negative = False
            
            for neg_example in negative_examples:
                if key in neg_example.properties:
                    if neg_example.properties[key] == value:
                        appears_in_negative = True
                        break
            
            if not appears_in_negative:
                filtered[key] = value
        
        return filtered
    
    def _calculate_confidence(
        self,
        positive_examples: List[ConceptExample],
        negative_examples: List[ConceptExample],
        common_props: Dict[str, Any],
        typical_props: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence in the synthesized concept
        
        Based on:
        - Number of examples
        - Consistency of properties
        - Distinctiveness from negative examples
        """
        # Base confidence from number of examples
        example_confidence = min(len(positive_examples) / 10.0, 1.0)
        
        # Consistency: how many properties are common vs typical
        if typical_props:
            consistency = len(common_props) / len(typical_props)
        else:
            consistency = 1.0 if common_props else 0.0
        
        # Distinctiveness from negative examples
        if negative_examples:
            # Check how different we are from negative examples
            distinctiveness = 1.0  # Default if we filtered well
        else:
            distinctiveness = 0.8  # Lower if no negative examples to compare
        
        # Weighted combination
        confidence = (
            0.4 * example_confidence +
            0.4 * consistency +
            0.2 * distinctiveness
        )
        
        return confidence
    
    def _generate_concept_name(
        self,
        common_props: Dict[str, Any],
        typical_props: Dict[str, Any]
    ) -> str:
        """Generate a descriptive name for the concept"""
        # Use hash of properties for unique ID
        prop_str = str(sorted(common_props.items()))
        hash_suffix = hashlib.md5(prop_str.encode()).hexdigest()[:8]
        
        # Try to create descriptive name from properties
        if 'type' in common_props:
            return f"{common_props['type']}_{hash_suffix}"
        elif 'category' in common_props:
            return f"{common_props['category']}_{hash_suffix}"
        else:
            return f"concept_{hash_suffix}"
    
    def _infer_parent_concepts(
        self,
        properties: Dict[str, Any],
        context: Optional[Dict]
    ) -> Set[str]:
        """Infer parent concepts from properties"""
        parents = set()
        
        # Simple heuristics
        if 'is_a' in properties:
            parents.add(properties['is_a'])
        
        if 'type' in properties:
            parents.add(properties['type'])
        
        if 'category' in properties:
            parents.add(properties['category'])
        
        return parents
    
    def refine_concept(
        self,
        concept: SynthesizedConcept,
        new_examples: List[ConceptExample]
    ) -> SynthesizedConcept:
        """
        Refine an existing concept with new examples
        Allows incremental learning
        """
        # Combine old and new examples
        all_examples = concept.source_examples + new_examples
        
        # Re-synthesize with all examples
        refined = self.synthesize_concept(
            all_examples,
            concept_name=concept.concept_id
        )
        
        return refined if refined else concept
    
    def merge_concepts(
        self,
        concept1: SynthesizedConcept,
        concept2: SynthesizedConcept
    ) -> Optional[SynthesizedConcept]:
        """
        Merge two similar concepts into a more general one
        """
        # Combine examples
        all_examples = concept1.source_examples + concept2.source_examples
        
        # Synthesize merged concept
        merged = self.synthesize_concept(all_examples)
        
        if merged:
            # Increase abstraction level
            merged.abstraction_level = max(
                concept1.abstraction_level,
                concept2.abstraction_level
            ) + 1
            
            # Combine parent concepts
            merged.parent_concepts = concept1.parent_concepts | concept2.parent_concepts
        
        return merged


def demo_concept_synthesis():
    """Demonstrate concept synthesis capabilities"""
    print("=" * 70)
    print("CONCEPT SYNTHESIS DEMO - Issue #21")
    print("Creating new concepts from examples (Abductive Learning)")
    print("=" * 70)
    print()
    
    # Example 1: Synthesize "bird" concept from examples
    print("1. Synthesizing 'bird' concept from examples")
    print("-" * 70)
    
    bird_examples = [
        ConceptExample(
            example_id="eagle",
            properties={
                'has_wings': True,
                'can_fly': True,
                'has_feathers': True,
                'lays_eggs': True,
                'warm_blooded': True,
                'type': 'animal'
            },
            relations={'is_a': {'animal'}}
        ),
        ConceptExample(
            example_id="sparrow",
            properties={
                'has_wings': True,
                'can_fly': True,
                'has_feathers': True,
                'lays_eggs': True,
                'warm_blooded': True,
                'type': 'animal',
                'size': 'small'
            },
            relations={'is_a': {'animal'}}
        ),
        ConceptExample(
            example_id="penguin",
            properties={
                'has_wings': True,
                'can_fly': False,  # Different!
                'has_feathers': True,
                'lays_eggs': True,
                'warm_blooded': True,
                'type': 'animal'
            },
            relations={'is_a': {'animal'}}
        ),
        ConceptExample(
            example_id="ostrich",
            properties={
                'has_wings': True,
                'can_fly': False,
                'has_feathers': True,
                'lays_eggs': True,
                'warm_blooded': True,
                'type': 'animal',
                'size': 'large'
            },
            relations={'is_a': {'animal'}}
        ),
    ]
    
    # Add negative example (bat - has wings but not a bird)
    bat_example = ConceptExample(
        example_id="bat",
        properties={
            'has_wings': True,
            'can_fly': True,
            'has_feathers': False,  # Key difference!
            'lays_eggs': False,
            'warm_blooded': True,
            'type': 'animal'
        },
        relations={'is_a': {'mammal'}},
        positive=False
    )
    
    synthesizer = ConceptSynthesizer(min_examples=3, min_confidence=0.6)
    
    bird_concept = synthesizer.synthesize_concept(
        bird_examples + [bat_example],
        concept_name='bird'
    )
    
    if bird_concept:
        print(f"\n✓ Concept synthesized: {bird_concept.concept_id}")
        print(f"  Confidence: {bird_concept.confidence:.2%}")
        print(f"  Source examples: {len(bird_concept.source_examples)}")
        print(f"\n  Common properties (in ALL examples):")
        for key, value in bird_concept.common_properties.items():
            print(f"    {key}: {value}")
        print(f"\n  Typical properties (in >70% of examples):")
        for key, value in bird_concept.typical_properties.items():
            print(f"    {key}: {value}")
        print(f"\n  Relations:")
        for rel, targets in bird_concept.common_relations.items():
            print(f"    {rel}: {targets}")
    
    # Example 2: Synthesize "mammal" concept
    print("\n\n2. Synthesizing 'mammal' concept from examples")
    print("-" * 70)
    
    mammal_examples = [
        ConceptExample(
            example_id="dog",
            properties={
                'has_fur': True,
                'gives_birth': True,
                'warm_blooded': True,
                'produces_milk': True,
                'type': 'animal'
            }
        ),
        ConceptExample(
            example_id="cat",
            properties={
                'has_fur': True,
                'gives_birth': True,
                'warm_blooded': True,
                'produces_milk': True,
                'type': 'animal'
            }
        ),
        ConceptExample(
            example_id="whale",
            properties={
                'has_fur': False,  # Whales don't have fur
                'gives_birth': True,
                'warm_blooded': True,
                'produces_milk': True,
                'type': 'animal',
                'lives_in': 'water'
            }
        ),
    ]
    
    mammal_concept = synthesizer.synthesize_concept(
        mammal_examples,
        concept_name='mammal'
    )
    
    if mammal_concept:
        print(f"\n✓ Concept synthesized: {mammal_concept.concept_id}")
        print(f"  Confidence: {mammal_concept.confidence:.2%}")
        print(f"\n  Common properties:")
        for key, value in mammal_concept.common_properties.items():
            print(f"    {key}: {value}")
    
    # Example 3: Convert to MKU structure
    print("\n\n3. Converting to MonadicKnowledgeUnit structure")
    print("-" * 70)
    
    if bird_concept:
        mku_structure = bird_concept.to_mku_structure()
        print(f"\nMKU structure for '{bird_concept.concept_id}':")
        print(f"  concept_id: {mku_structure['concept_id']}")
        print(f"  predicate: {mku_structure['deep_structure']['predicate']}")
        print(f"  properties: {len(mku_structure['deep_structure']['properties'])} common")
        print(f"  constraints: {mku_structure['deep_structure']['constraints']}")
        print(f"  confidence: {mku_structure['confidence']:.2%}")
    
    # Example 4: Concept refinement
    print("\n\n4. Refining concept with new example")
    print("-" * 70)
    
    if bird_concept:
        new_example = ConceptExample(
            example_id="hummingbird",
            properties={
                'has_wings': True,
                'can_fly': True,
                'has_feathers': True,
                'lays_eggs': True,
                'warm_blooded': True,
                'type': 'animal',
                'size': 'tiny'
            }
        )
        
        refined_bird = synthesizer.refine_concept(bird_concept, [new_example])
        print(f"\n✓ Concept refined with new example")
        print(f"  Old confidence: {bird_concept.confidence:.2%}")
        print(f"  New confidence: {refined_bird.confidence:.2%}")
        print(f"  Total examples: {len(refined_bird.source_examples)}")
    
    print("\n" + "=" * 70)
    print("✓ Concept synthesis complete!")
    print("=" * 70)
    print()
    print("Key Capabilities:")
    print("  ✓ Synthesizes concepts from 3+ examples")
    print("  ✓ Extracts common properties (100% of examples)")
    print("  ✓ Identifies typical properties (>70% of examples)")
    print("  ✓ Uses negative examples for discrimination")
    print("  ✓ Calculates confidence scores")
    print("  ✓ Infers parent concepts and relations")
    print("  ✓ Refines concepts with new examples")
    print("  ✓ Generates MKU-compatible structures")
    print("  ✓ System can now CREATE new knowledge!")
    print("=" * 70)


if __name__ == '__main__':
    demo_concept_synthesis()
