#!/usr/bin/env python3
"""
Analogical Reasoning Engine - Issues #15-18
Implements Hofstadter's approach to finding and transferring structural analogies

Phase 3: Extract abstract structure, find isomorphisms, transfer knowledge
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class AbstractStructure:
    """
    Abstract relational structure (Issue #15)
    Represents the pattern of relationships without specific content
    """
    nodes: Set[str]  # Abstract node labels (A, B, C...)
    edges: List[Tuple[str, str, str]]  # (source, relation, target)
    properties: Dict[str, Set[str]]  # node -> {property_types}
    signature: str = ""  # Hash signature for quick comparison
    
    def __post_init__(self):
        if not self.signature:
            self.signature = self._compute_signature()
    
    def _compute_signature(self) -> str:
        """Compute structural signature for comparison"""
        # Sort everything for deterministic hash
        nodes_str = '|'.join(sorted(self.nodes))
        edges_str = '|'.join(sorted(f"{s}-{r}->{t}" for s, r, t in self.edges))
        props_str = '|'.join(sorted(f"{n}:{','.join(sorted(p))}" 
                                   for n, p in self.properties.items()))
        
        combined = f"{nodes_str}::{edges_str}::{props_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def similarity(self, other: 'AbstractStructure') -> float:
        """
        Calculate structural similarity with another structure
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Quick check: same signature = identical structure
        if self.signature == other.signature:
            return 1.0
        
        # Node count similarity
        node_sim = len(self.nodes & other.nodes) / max(len(self.nodes | other.nodes), 1)
        
        # Edge pattern similarity
        self_edge_patterns = {(r, ) for _, r, _ in self.edges}
        other_edge_patterns = {(r, ) for _, r, _ in other.edges}
        edge_sim = len(self_edge_patterns & other_edge_patterns) / max(
            len(self_edge_patterns | other_edge_patterns), 1
        )
        
        # Property similarity
        all_nodes = self.nodes | other.nodes
        prop_matches = 0
        prop_total = 0
        for node in all_nodes:
            self_props = self.properties.get(node, set())
            other_props = other.properties.get(node, set())
            if self_props or other_props:
                prop_matches += len(self_props & other_props)
                prop_total += len(self_props | other_props)
        
        prop_sim = prop_matches / max(prop_total, 1)
        
        # Weighted combination
        return 0.4 * node_sim + 0.4 * edge_sim + 0.2 * prop_sim


class StructureExtractor:
    """
    Extract abstract structure from concepts (Issue #15)
    """
    
    def __init__(self, max_depth: int = 3):
        """
        Args:
            max_depth: Maximum depth for structure extraction
        """
        self.max_depth = max_depth
    
    def extract_structure(
        self,
        concept_id: str,
        knowledge_graph: Dict,
        abstract_labels: bool = True
    ) -> AbstractStructure:
        """
        Extract abstract relational structure from a concept
        
        Args:
            concept_id: Starting concept
            knowledge_graph: Knowledge graph (dict of concept_id -> MKU)
            abstract_labels: Use abstract labels (A, B, C) instead of actual IDs
            
        Returns:
            AbstractStructure representing the pattern
        """
        if concept_id not in knowledge_graph:
            return AbstractStructure(nodes=set(), edges=[], properties={})
        
        # Extract local structure
        visited = set()
        nodes = set()
        edges = []
        properties = defaultdict(set)
        
        # BFS to extract structure
        queue = [(concept_id, 0, None)]  # (node, depth, label)
        label_map = {}
        label_counter = 0
        
        while queue:
            current_id, depth, parent_label = queue.pop(0)
            
            if depth > self.max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Assign abstract label
            if abstract_labels:
                if current_id not in label_map:
                    label_map[current_id] = chr(65 + label_counter)  # A, B, C...
                    label_counter += 1
                current_label = label_map[current_id]
            else:
                current_label = current_id
            
            nodes.add(current_label)
            
            # Extract node properties (types, not values)
            if current_id in knowledge_graph:
                concept = knowledge_graph[current_id]
                
                # Add property types (not values)
                if hasattr(concept, 'deep_structure'):
                    ds = concept.deep_structure
                    if 'predicate' in ds:
                        properties[current_label].add(f"pred:{ds['predicate']}")
                    if 'properties' in ds and isinstance(ds['properties'], dict):
                        for prop_name in ds['properties'].keys():
                            properties[current_label].add(f"prop:{prop_name}")
                
                # Extract relations
                if hasattr(concept, 'relations'):
                    for rel_type, targets in concept.relations.items():
                        for target_id in targets:
                            if target_id not in label_map and abstract_labels:
                                label_map[target_id] = chr(65 + label_counter)
                                label_counter += 1
                            
                            target_label = label_map.get(target_id, target_id) if abstract_labels else target_id
                            edges.append((current_label, rel_type, target_label))
                            
                            # Add to queue for further exploration
                            if depth < self.max_depth:
                                queue.append((target_id, depth + 1, current_label))
        
        return AbstractStructure(
            nodes=nodes,
            edges=edges,
            properties=dict(properties)
        )
    
    def extract_multiple(
        self,
        concept_ids: List[str],
        knowledge_graph: Dict
    ) -> Dict[str, AbstractStructure]:
        """
        Extract structures for multiple concepts
        
        Returns:
            Dict mapping concept_id -> AbstractStructure
        """
        structures = {}
        for concept_id in concept_ids:
            structures[concept_id] = self.extract_structure(
                concept_id,
                knowledge_graph,
                abstract_labels=True
            )
        return structures
    
    def find_similar_structures(
        self,
        query_structure: AbstractStructure,
        structure_library: Dict[str, AbstractStructure],
        min_similarity: float = 0.5,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find concepts with similar structure
        
        Args:
            query_structure: Structure to match
            structure_library: Dict of concept_id -> AbstractStructure
            min_similarity: Minimum similarity threshold
            top_k: Number of results to return
            
        Returns:
            List of (concept_id, similarity_score) sorted by similarity
        """
        matches = []
        
        for concept_id, structure in structure_library.items():
            similarity = query_structure.similarity(structure)
            
            if similarity >= min_similarity:
                matches.append((concept_id, similarity))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]


def demo_structure_extraction():
    """Demonstrate structure extraction"""
    print("=" * 70)
    print("STRUCTURE EXTRACTION DEMO - Issue #15")
    print("Hofstadter's Analogical Reasoning")
    print("=" * 70)
    print()
    
    # Create mock knowledge graph
    from dataclasses import dataclass as dc
    from typing import Dict as D, Set as S
    
    @dc
    class MockMKU:
        concept_id: str
        deep_structure: D
        relations: D[str, S[str]]
    
    kg = {
        'sun': MockMKU(
            'sun',
            {'predicate': 'star', 'properties': {'hot': True, 'bright': True}},
            {'orbited_by': {'earth', 'mars', 'venus'}}
        ),
        'earth': MockMKU(
            'earth',
            {'predicate': 'planet', 'properties': {'orbits': True}},
            {'orbits': {'sun'}}
        ),
        'mars': MockMKU(
            'mars',
            {'predicate': 'planet', 'properties': {'orbits': True}},
            {'orbits': {'sun'}}
        ),
        'nucleus': MockMKU(
            'nucleus',
            {'predicate': 'particle', 'properties': {'positive': True, 'heavy': True}},
            {'orbited_by': {'electron1', 'electron2'}}
        ),
        'electron1': MockMKU(
            'electron1',
            {'predicate': 'particle', 'properties': {'orbits': True}},
            {'orbits': {'nucleus'}}
        ),
    }
    
    print("1. Extracting structure from 'sun' (solar system)")
    print("-" * 70)
    
    extractor = StructureExtractor(max_depth=2)
    solar_structure = extractor.extract_structure('sun', kg)
    
    print(f"\nAbstract structure:")
    print(f"  Nodes: {sorted(solar_structure.nodes)}")
    print(f"  Edges: {len(solar_structure.edges)}")
    print(f"  Signature: {solar_structure.signature}")
    print(f"\nRelations:")
    for s, r, t in solar_structure.edges[:5]:
        print(f"  {s} --[{r}]--> {t}")
    
    print("\n2. Extracting structure from 'nucleus' (atomic structure)")
    print("-" * 70)
    
    atom_structure = extractor.extract_structure('nucleus', kg)
    
    print(f"\nAbstract structure:")
    print(f"  Nodes: {sorted(atom_structure.nodes)}")
    print(f"  Edges: {len(atom_structure.edges)}")
    print(f"  Signature: {atom_structure.signature}")
    print(f"\nRelations:")
    for s, r, t in atom_structure.edges[:5]:
        print(f"  {s} --[{r}]--> {t}")
    
    print("\n3. Comparing structural similarity")
    print("-" * 70)
    
    similarity = solar_structure.similarity(atom_structure)
    print(f"\nSimilarity between solar system and atom: {similarity:.2f}")
    print(f"Interpretation: {'HIGH - Strong analogy!' if similarity > 0.6 else 'MODERATE - Some similarity' if similarity > 0.3 else 'LOW - Different structures'}")
    
    print("\n4. Finding similar structures")
    print("-" * 70)
    
    # Build structure library
    structures = extractor.extract_multiple(['sun', 'nucleus', 'earth'], kg)
    
    # Find matches for solar system
    matches = extractor.find_similar_structures(
        solar_structure,
        structures,
        min_similarity=0.3
    )
    
    print(f"\nConcepts with similar structure to 'sun':")
    for concept_id, sim in matches:
        print(f"  {concept_id}: {sim:.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Structure extraction complete!")
    print("=" * 70)
    print()
    print("Key Insight:")
    print("  The solar system (sun + planets) has similar STRUCTURE")
    print("  to an atom (nucleus + electrons), even though the")
    print("  content is completely different!")
    print("=" * 70)


if __name__ == '__main__':
    demo_structure_extraction()
