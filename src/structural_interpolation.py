#!/usr/bin/env python3
"""
Structural Interpolation - Issue #22
Synthesizes intermediate concepts to fill gaps between existing concepts

Phase 4: Self-Improvement - Conceptual hierarchy building
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class ConceptDistance:
    """
    Represents the distance/relationship between two concepts
    """
    concept_a: str
    concept_b: str
    distance: float  # 0.0 = identical, 1.0 = completely different
    path: List[str] = field(default_factory=list)  # Path through graph
    common_properties: Set[str] = field(default_factory=set)
    differing_properties: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    
    def needs_interpolation(self, threshold: float = 0.5) -> bool:
        """Check if gap is large enough to need interpolation"""
        # Need interpolation if distance is high, regardless of path length
        # Path length 2 means direct connection, still might need intermediate
        return self.distance > threshold


@dataclass
class InterpolatedConcept:
    """
    A concept synthesized to fill a gap between two concepts
    """
    concept_id: str
    confidence: float
    
    # Structural properties (blend of A and C)
    properties: Dict[str, Any]
    relations: Dict[str, Set[str]]
    
    # Position in hierarchy
    distance_from_a: float  # 0.0 to 1.0
    distance_from_c: float
    
    # Provenance
    source_concept_a: str
    source_concept_c: str
    interpolation_strategy: str
    
    def to_mku_structure(self) -> Dict[str, Any]:
        """Convert to MonadicKnowledgeUnit structure"""
        return {
            'concept_id': self.concept_id,
            'deep_structure': {
                'predicate': f"interpolation({self.source_concept_a}, {self.source_concept_c})",
                'properties': self.properties,
                'constraints': [
                    f"more_general_than({self.source_concept_a})",
                    f"more_specific_than({self.source_concept_c})"
                ]
            },
            'relations': self.relations,
            'confidence': self.confidence,
            'meta': {
                'interpolated': True,
                'sources': [self.source_concept_a, self.source_concept_c]
            }
        }


class StructuralInterpolator:
    """
    Synthesizes intermediate concepts between existing concepts (Issue #22)
    
    Given concepts A (specific) and C (general), finds or creates B
    such that: A → B → C
    
    Example: dog (A) → mammal (B) → animal (C)
    """
    
    def __init__(
        self,
        interpolation_threshold: float = 0.5,
        min_confidence: float = 0.6
    ):
        """
        Args:
            interpolation_threshold: Minimum distance to warrant interpolation
            min_confidence: Minimum confidence for synthesized concepts
        """
        self.interpolation_threshold = interpolation_threshold
        self.min_confidence = min_confidence
    
    def interpolate_between(
        self,
        concept_a: str,
        concept_c: str,
        knowledge_graph: Dict,
        num_intermediates: int = 1
    ) -> List[InterpolatedConcept]:
        """
        Synthesize intermediate concept(s) between A and C
        
        Args:
            concept_a: Specific concept (e.g., 'dog')
            concept_c: General concept (e.g., 'animal')
            knowledge_graph: Current knowledge graph
            num_intermediates: Number of intermediate concepts to create
            
        Returns:
            List of interpolated concepts
        """
        # Analyze distance and relationship
        distance = self._compute_distance(concept_a, concept_c, knowledge_graph)
        
        # Check if interpolation is needed
        if distance.distance <= self.interpolation_threshold:
            return []  # Too close, no need for interpolation
        
        # Strategy selection based on available information
        if distance.common_properties and distance.differing_properties:
            strategy = "property_blending"
        elif distance.path:
            strategy = "path_interpolation"
        else:
            strategy = "structural_analysis"
        
        # Synthesize intermediate concepts
        interpolated = []
        
        for i in range(num_intermediates):
            # Calculate position along spectrum (0.0 = A, 1.0 = C)
            position = (i + 1) / (num_intermediates + 1)
            
            concept = self._synthesize_intermediate(
                concept_a, concept_c,
                distance, position,
                strategy, knowledge_graph
            )
            
            if concept and concept.confidence >= self.min_confidence:
                interpolated.append(concept)
        
        return interpolated
    
    def _compute_distance(
        self,
        concept_a: str,
        concept_c: str,
        knowledge_graph: Dict
    ) -> ConceptDistance:
        """
        Compute semantic/structural distance between concepts
        """
        # Get concepts from graph
        mku_a = knowledge_graph.get(concept_a)
        mku_c = knowledge_graph.get(concept_c)
        
        if not mku_a or not mku_c:
            return ConceptDistance(
                concept_a=concept_a,
                concept_b=concept_c,
                distance=1.0  # Maximum distance if not found
            )
        
        # Extract properties
        props_a = set()
        props_c = set()
        
        if hasattr(mku_a, 'deep_structure'):
            props_a = set(mku_a.deep_structure.get('properties', {}).keys())
        
        if hasattr(mku_c, 'deep_structure'):
            props_c = set(mku_c.deep_structure.get('properties', {}).keys())
        
        # Compute property-based distance
        if props_a or props_c:
            common = props_a & props_c
            total = props_a | props_c
            property_similarity = len(common) / len(total) if total else 0.0
            distance = 1.0 - property_similarity
        else:
            distance = 0.5  # Unknown
        
        # Find path between concepts
        path = self._find_path(concept_a, concept_c, knowledge_graph)
        
        # Adjust distance based on path length
        if path:
            path_distance = (len(path) - 1) / 10.0  # Normalize
            distance = max(distance, path_distance)
        
        # Identify common and differing properties
        common_props = props_a & props_c
        differing = {}
        
        for prop in props_a - props_c:
            if hasattr(mku_a, 'deep_structure'):
                val_a = mku_a.deep_structure.get('properties', {}).get(prop)
                differing[prop] = (val_a, None)
        
        for prop in props_c - props_a:
            if hasattr(mku_c, 'deep_structure'):
                val_c = mku_c.deep_structure.get('properties', {}).get(prop)
                differing[prop] = (None, val_c)
        
        return ConceptDistance(
            concept_a=concept_a,
            concept_b=concept_c,
            distance=distance,
            path=path,
            common_properties=common_props,
            differing_properties=differing
        )
    
    def _find_path(
        self,
        start: str,
        goal: str,
        knowledge_graph: Dict,
        max_depth: int = 5
    ) -> List[str]:
        """Find shortest path between concepts (BFS)"""
        if start not in knowledge_graph or goal not in knowledge_graph:
            return []
        
        if start == goal:
            return [start]
        
        # BFS
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            mku = knowledge_graph[current]
            
            # Check all relations
            if hasattr(mku, 'relations'):
                for rel_type, targets in mku.relations.items():
                    for target in targets:
                        if target == goal:
                            return path + [target]
                        
                        if target not in visited and target in knowledge_graph:
                            visited.add(target)
                            queue.append((target, path + [target]))
        
        return []
    
    def _synthesize_intermediate(
        self,
        concept_a: str,
        concept_c: str,
        distance: ConceptDistance,
        position: float,
        strategy: str,
        knowledge_graph: Dict
    ) -> Optional[InterpolatedConcept]:
        """
        Synthesize a single intermediate concept
        
        position: 0.0 = closer to A, 1.0 = closer to C
        """
        if strategy == "property_blending":
            return self._blend_properties(
                concept_a, concept_c, distance, position, knowledge_graph
            )
        elif strategy == "path_interpolation":
            return self._interpolate_along_path(
                concept_a, concept_c, distance, position, knowledge_graph
            )
        else:
            return self._structural_interpolation(
                concept_a, concept_c, distance, position, knowledge_graph
            )
    
    def _blend_properties(
        self,
        concept_a: str,
        concept_c: str,
        distance: ConceptDistance,
        position: float,
        knowledge_graph: Dict
    ) -> Optional[InterpolatedConcept]:
        """
        Create intermediate by blending properties
        
        Position determines which properties to include:
        - position = 0.3 → more A properties, few C properties
        - position = 0.7 → more C properties, few A properties
        """
        mku_a = knowledge_graph[concept_a]
        mku_c = knowledge_graph[concept_c]
        
        # Start with common properties (always included)
        blended_props = {}
        
        for prop in distance.common_properties:
            if hasattr(mku_a, 'deep_structure'):
                val = mku_a.deep_structure.get('properties', {}).get(prop)
                if val is not None:
                    blended_props[prop] = val
        
        # Add specific properties based on position
        # Closer to A (position < 0.5) → include more A-specific properties
        # Closer to C (position > 0.5) → include more C-specific properties
        
        if hasattr(mku_a, 'deep_structure'):
            props_a = mku_a.deep_structure.get('properties', {})
            for prop, val in props_a.items():
                if prop not in distance.common_properties:
                    # Include A-specific property with probability based on distance from C
                    if (1.0 - position) > 0.5:  # Closer to A
                        blended_props[prop] = val
        
        if hasattr(mku_c, 'deep_structure'):
            props_c = mku_c.deep_structure.get('properties', {})
            for prop, val in props_c.items():
                if prop not in distance.common_properties:
                    # Include C-specific property with probability based on distance from A
                    if position > 0.5:  # Closer to C
                        blended_props[prop] = val
        
        # Generate concept name
        concept_name = self._generate_intermediate_name(concept_a, concept_c, position)
        
        # Build relations
        relations = {
            'specializes': {concept_c},  # Intermediate is more specific than C
            'generalizes': {concept_a}   # Intermediate is more general than A
        }
        
        # Calculate confidence
        confidence = self._calculate_interpolation_confidence(
            distance, len(blended_props), position
        )
        
        return InterpolatedConcept(
            concept_id=concept_name,
            confidence=confidence,
            properties=blended_props,
            relations=relations,
            distance_from_a=position,
            distance_from_c=1.0 - position,
            source_concept_a=concept_a,
            source_concept_c=concept_c,
            interpolation_strategy="property_blending"
        )
    
    def _interpolate_along_path(
        self,
        concept_a: str,
        concept_c: str,
        distance: ConceptDistance,
        position: float,
        knowledge_graph: Dict
    ) -> Optional[InterpolatedConcept]:
        """
        If path exists, identify concept at position along path
        """
        if not distance.path or len(distance.path) < 3:
            return None
        
        # Find concept closest to position in path
        path_length = len(distance.path) - 1
        target_index = int(position * path_length)
        
        if 0 < target_index < len(distance.path):
            # Concept already exists in path
            existing_concept = distance.path[target_index]
            
            # Return as interpolated concept (already exists)
            return InterpolatedConcept(
                concept_id=existing_concept,
                confidence=1.0,  # High confidence - already exists
                properties={},  # Would need to extract from graph
                relations={'is_a': {concept_c}},
                distance_from_a=target_index / path_length,
                distance_from_c=1.0 - (target_index / path_length),
                source_concept_a=concept_a,
                source_concept_c=concept_c,
                interpolation_strategy="path_existing"
            )
        
        return None
    
    def _structural_interpolation(
        self,
        concept_a: str,
        concept_c: str,
        distance: ConceptDistance,
        position: float,
        knowledge_graph: Dict
    ) -> Optional[InterpolatedConcept]:
        """
        Fallback: structural analysis to create intermediate
        """
        # Use property blending as fallback
        return self._blend_properties(
            concept_a, concept_c, distance, position, knowledge_graph
        )
    
    def _generate_intermediate_name(
        self,
        concept_a: str,
        concept_c: str,
        position: float
    ) -> str:
        """Generate descriptive name for intermediate concept"""
        # Create hash for uniqueness
        hash_input = f"{concept_a}_{concept_c}_{position}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        
        # Descriptive name based on position
        if position < 0.5:
            return f"{concept_a}_type_{hash_suffix}"
        else:
            return f"specific_{concept_c}_{hash_suffix}"
    
    def _calculate_interpolation_confidence(
        self,
        distance: ConceptDistance,
        num_properties: int,
        position: float
    ) -> float:
        """
        Calculate confidence in interpolated concept
        
        Based on:
        - Distance between concepts (closer = higher confidence)
        - Number of properties (more = higher confidence)
        - Position (middle = more confident than extremes)
        """
        # Distance confidence: High distance means gap exists (good for interpolation!)
        # Invert the logic - larger gaps NEED interpolation
        distance_confidence = min(distance.distance, 1.0) * 0.5 + 0.3  # Range: 0.3 to 0.8
        
        # Property confidence
        property_confidence = min(num_properties / 3.0, 1.0)  # Lower threshold
        
        # Position confidence (middle is best)
        position_confidence = 1.0 - abs(0.5 - position) * 2.0
        
        # Weighted combination
        confidence = (
            0.3 * distance_confidence +
            0.4 * property_confidence +
            0.3 * position_confidence
        )
        
        return confidence
    
    def find_gaps(
        self,
        knowledge_graph: Dict,
        min_distance: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Identify gaps in knowledge graph that need interpolation
        
        Returns:
            List of (concept_a, concept_c, distance) tuples
        """
        gaps = []
        concepts = list(knowledge_graph.keys())
        
        # Check all pairs
        for i, concept_a in enumerate(concepts):
            for concept_c in concepts[i+1:]:
                distance = self._compute_distance(concept_a, concept_c, knowledge_graph)
                
                if distance.needs_interpolation(min_distance):
                    gaps.append((concept_a, concept_c, distance.distance))
        
        # Sort by distance (largest gaps first)
        gaps.sort(key=lambda x: x[2], reverse=True)
        
        return gaps


def demo_structural_interpolation():
    """Demonstrate structural interpolation capabilities"""
    print("=" * 70)
    print("STRUCTURAL INTERPOLATION DEMO - Issue #22")
    print("Filling conceptual gaps: A → B → C")
    print("=" * 70)
    print()
    
    # Create mock knowledge graph with gap
    from dataclasses import dataclass
    from typing import Dict, Any
    
    @dataclass
    class MockMKU:
        concept_id: str
        deep_structure: Dict[str, Any]
        relations: Dict[str, set]
    
    kg = {
        'dog': MockMKU(
            'dog',
            {
                'predicate': 'species',
                'properties': {
                    'has_fur': True,
                    'barks': True,
                    'domesticated': True,
                    'has_backbone': True,
                    'warm_blooded': True,
                    'four_legs': True
                }
            },
            {'is_a': {'animal'}}  # Gap! No mammal
        ),
        'cat': MockMKU(
            'cat',
            {
                'predicate': 'species',
                'properties': {
                    'has_fur': True,
                    'meows': True,
                    'domesticated': True,
                    'has_backbone': True,
                    'warm_blooded': True,
                    'four_legs': True
                }
            },
            {'is_a': {'animal'}}  # Gap! No mammal
        ),
        'animal': MockMKU(
            'animal',
            {
                'predicate': 'kingdom',
                'properties': {
                    'moves': True,
                    'eats': True,
                    'has_backbone': True
                }
            },
            {}
        ),
    }
    
    print("Knowledge graph has gap:")
    print("  dog → [???] → animal")
    print("  cat → [???] → animal")
    print()
    
    interpolator = StructuralInterpolator(
        interpolation_threshold=0.3,  # Lower threshold to detect the gap
        min_confidence=0.5
    )
    
    # Example 1: Compute distance first
    print("1. Computing distance between 'dog' and 'animal'")
    print("-" * 70)
    
    distance = interpolator._compute_distance('dog', 'animal', kg)
    print(f"\nDistance analysis:")
    print(f"  Overall distance: {distance.distance:.2f}")
    print(f"  Common properties: {distance.common_properties}")
    print(f"  Path exists: {bool(distance.path)}")
    print(f"  Needs interpolation: {distance.needs_interpolation(0.3)}")
    
    # Example 2: Find all gaps
    print("\n\n2. Identifying gaps in knowledge graph")
    print("-" * 70)
    
    gaps = interpolator.find_gaps(kg, min_distance=0.2)
    print(f"\nFound {len(gaps)} gaps:")
    for concept_a, concept_c, dist in gaps[:5]:
        print(f"  • {concept_a} → {concept_c} (distance: {dist:.2f})")
    
    # Example 3: Interpolate between dog and animal
    print("\n\n3. Interpolating between 'dog' and 'animal'")
    print("-" * 70)
    
    print("\nAnalyzing relationship...")
    print(f"  Distance: {distance.distance:.2f} (threshold: 0.3)")
    print(f"  Should interpolate: {distance.distance > 0.3}")
    
    intermediates = interpolator.interpolate_between('dog', 'animal', kg, num_intermediates=1)
    print(f"  Intermediates created: {len(intermediates)}")
    
    if intermediates:
        for intermediate in intermediates:
            print(f"\n✓ INTERPOLATED CONCEPT CREATED:")
            print(f"  ID: {intermediate.concept_id}")
            print(f"  Confidence: {intermediate.confidence:.2%}")
            print(f"  Strategy: {intermediate.interpolation_strategy}")
            print(f"\n  Position in hierarchy:")
            print(f"    Distance from 'dog': {intermediate.distance_from_a:.2f}")
            print(f"    Distance from 'animal': {intermediate.distance_from_c:.2f}")
            print(f"\n  Properties:")
            for key, value in intermediate.properties.items():
                print(f"    • {key}: {value}")
            print(f"\n  Relations:")
            for rel, targets in intermediate.relations.items():
                print(f"    • {rel}: {targets}")
            
            # Show hierarchy
            print(f"\n  Resulting hierarchy:")
            print(f"    animal (general)")
            print(f"      ↑")
            print(f"    {intermediate.concept_id} ← INTERPOLATED")
            print(f"      ↑")
            print(f"    dog (specific)")
    else:
        print("\n✗ No interpolation needed or possible")
    
    # Example 4: Show MKU structure
    if intermediates:
        print("\n\n4. Converting to MKU structure")
        print("-" * 70)
        
        intermediate = intermediates[0]
        mku_structure = intermediate.to_mku_structure()
        
        print(f"\nMKU structure:")
        print(f"  concept_id: {mku_structure['concept_id']}")
        print(f"  predicate: {mku_structure['deep_structure']['predicate']}")
        print(f"  properties: {len(mku_structure['deep_structure']['properties'])} properties")
        print(f"  confidence: {mku_structure['confidence']:.2%}")
        print(f"  interpolated: {mku_structure['meta']['interpolated']}")
        print(f"  sources: {mku_structure['meta']['sources']}")
    
    print("\n" + "=" * 70)
    print("✓ Structural interpolation complete!")
    print("=" * 70)
    print()
    print("Key Capabilities:")
    print("  ✓ Identifies gaps in knowledge graph")
    print("  ✓ Computes semantic distance between concepts")
    print("  ✓ Synthesizes intermediate concepts via property blending")
    print("  ✓ Positions intermediates in conceptual hierarchy")
    print("  ✓ Generates MKU-compatible structures")
    print("  ✓ System can now fill conceptual gaps automatically!")
    print()
    print("Example: Given 'dog' and 'animal', system synthesizes:")
    print("  → A mammal-like concept with shared properties")
    print("  → Positioned between specific (dog) and general (animal)")
    print("=" * 70)


if __name__ == '__main__':
    demo_structural_interpolation()
