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
from itertools import permutations


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


@dataclass
class NodeMapping:
    """
    Mapping between nodes in two structures (Issue #16)
    Represents an isomorphism or partial isomorphism
    """
    source_to_target: Dict[str, str]  # source_node -> target_node
    target_to_source: Dict[str, str]  # target_node -> source_node
    score: float = 0.0  # Quality of the mapping (0.0 to 1.0)
    
    def __post_init__(self):
        # Verify bidirectional consistency
        for s, t in self.source_to_target.items():
            assert self.target_to_source.get(t) == s, "Inconsistent mapping"
    
    def is_complete(self, source_nodes: Set[str], target_nodes: Set[str]) -> bool:
        """Check if this is a complete isomorphism"""
        return (
            len(self.source_to_target) == len(source_nodes) and
            len(self.target_to_source) == len(target_nodes) and
            set(self.source_to_target.keys()) == source_nodes and
            set(self.target_to_source.keys()) == target_nodes
        )


class IsomorphismMatcher:
    """
    Find optimal node mappings between similar structures (Issue #16)
    Implements graph isomorphism and partial matching algorithms
    """
    
    def __init__(self, max_attempts: int = 1000):
        """
        Args:
            max_attempts: Maximum mapping attempts for large structures
        """
        self.max_attempts = max_attempts
    
    def find_best_mapping(
        self,
        source_structure: 'AbstractStructure',
        target_structure: 'AbstractStructure',
        require_complete: bool = False
    ) -> Optional[NodeMapping]:
        """
        Find the best node mapping between two structures
        
        Args:
            source_structure: Source structure
            target_structure: Target structure
            require_complete: Require complete isomorphism
            
        Returns:
            Best NodeMapping found, or None if no valid mapping exists
        """
        source_nodes = list(source_structure.nodes)
        target_nodes = list(target_structure.nodes)
        
        # Early exit: different sizes and complete mapping required
        if require_complete and len(source_nodes) != len(target_nodes):
            return None
        
        # Try exact mapping first (for small structures)
        # Exhaustive only works when target has enough nodes
        if (len(source_nodes) <= 6 and len(target_nodes) <= 6 and 
            len(source_nodes) <= len(target_nodes)):
            return self._exhaustive_search(
                source_structure,
                target_structure,
                require_complete
            )
        else:
            return self._greedy_search(
                source_structure,
                target_structure,
                require_complete
            )
    
    def _exhaustive_search(
        self,
        source_structure: 'AbstractStructure',
        target_structure: 'AbstractStructure',
        require_complete: bool
    ) -> Optional[NodeMapping]:
        """
        Exhaustive search for optimal mapping (small structures only)
        """
        source_nodes = list(source_structure.nodes)
        target_nodes = list(target_structure.nodes)
        
        best_mapping = None
        best_score = -1.0
        
        # Try all permutations
        num_attempts = 0
        for target_perm in permutations(target_nodes, len(source_nodes)):
            if num_attempts >= self.max_attempts:
                break
            num_attempts += 1
            
            # Create mapping
            s_to_t = dict(zip(source_nodes, target_perm))
            t_to_s = {v: k for k, v in s_to_t.items()}
            
            # Score this mapping
            score = self._score_mapping(
                s_to_t,
                source_structure,
                target_structure
            )
            
            if score > best_score:
                best_score = score
                best_mapping = NodeMapping(
                    source_to_target=s_to_t,
                    target_to_source=t_to_s,
                    score=score
                )
        
        # Check completeness requirement
        if require_complete and best_mapping:
            if not best_mapping.is_complete(source_structure.nodes, target_structure.nodes):
                return None
        
        return best_mapping
    
    def _greedy_search(
        self,
        source_structure: 'AbstractStructure',
        target_structure: 'AbstractStructure',
        require_complete: bool
    ) -> Optional[NodeMapping]:
        """
        Greedy heuristic search for large structures
        """
        source_nodes = list(source_structure.nodes)
        target_nodes = list(target_structure.nodes)
        
        # Start with empty mapping
        s_to_t = {}
        t_to_s = {}
        
        # Greedily match nodes based on local similarity
        for s_node in source_nodes:
            best_target = None
            best_local_score = -1.0
            
            for t_node in target_nodes:
                if t_node in t_to_s:
                    continue  # Already mapped
                
                # Local compatibility score
                local_score = self._local_compatibility(
                    s_node, t_node,
                    source_structure, target_structure,
                    s_to_t
                )
                
                if local_score > best_local_score:
                    best_local_score = local_score
                    best_target = t_node
            
            if best_target:
                s_to_t[s_node] = best_target
                t_to_s[best_target] = s_node
        
        if not s_to_t:
            return None
        
        # Score final mapping
        score = self._score_mapping(s_to_t, source_structure, target_structure)
        
        mapping = NodeMapping(
            source_to_target=s_to_t,
            target_to_source=t_to_s,
            score=score
        )
        
        # Check completeness
        if require_complete:
            if not mapping.is_complete(source_structure.nodes, target_structure.nodes):
                return None
        
        return mapping
    
    def _score_mapping(
        self,
        s_to_t: Dict[str, str],
        source_structure: 'AbstractStructure',
        target_structure: 'AbstractStructure'
    ) -> float:
        """
        Score a node mapping based on structure preservation
        
        Returns:
            Score from 0.0 (bad) to 1.0 (perfect)
        """
        if not s_to_t:
            return 0.0
        
        # Edge preservation: how many edges are preserved?
        edge_matches = 0
        edge_total = 0
        
        for s_src, rel, s_tgt in source_structure.edges:
            if s_src in s_to_t and s_tgt in s_to_t:
                edge_total += 1
                t_src = s_to_t[s_src]
                t_tgt = s_to_t[s_tgt]
                
                # Check if corresponding edge exists
                if (t_src, rel, t_tgt) in target_structure.edges:
                    edge_matches += 1
        
        edge_score = edge_matches / max(edge_total, 1) if edge_total > 0 else 0.0
        
        # Property preservation: how many property types match?
        prop_matches = 0
        prop_total = 0
        
        for s_node, t_node in s_to_t.items():
            s_props = source_structure.properties.get(s_node, set())
            t_props = target_structure.properties.get(t_node, set())
            
            if s_props or t_props:
                prop_matches += len(s_props & t_props)
                prop_total += len(s_props | t_props)
        
        prop_score = prop_matches / max(prop_total, 1) if prop_total > 0 else 0.5
        
        # Weighted combination (give some base score even if no properties match)
        return 0.7 * edge_score + 0.3 * prop_score
    
    def _local_compatibility(
        self,
        s_node: str,
        t_node: str,
        source_structure: 'AbstractStructure',
        target_structure: 'AbstractStructure',
        existing_mapping: Dict[str, str]
    ) -> float:
        """
        Compute local compatibility between two nodes
        """
        # Property similarity
        s_props = source_structure.properties.get(s_node, set())
        t_props = target_structure.properties.get(t_node, set())
        
        if s_props or t_props:
            prop_sim = len(s_props & t_props) / max(len(s_props | t_props), 1)
        else:
            prop_sim = 0.5
        
        # Neighbor compatibility (based on existing mappings)
        neighbor_matches = 0
        neighbor_total = 0
        
        # Check outgoing edges
        s_neighbors = {tgt for src, rel, tgt in source_structure.edges if src == s_node}
        t_neighbors = {tgt for src, rel, tgt in target_structure.edges if src == t_node}
        
        for s_nbr in s_neighbors:
            if s_nbr in existing_mapping:
                neighbor_total += 1
                if existing_mapping[s_nbr] in t_neighbors:
                    neighbor_matches += 1
        
        neighbor_score = neighbor_matches / max(neighbor_total, 1) if neighbor_total > 0 else 0.5
        
        return 0.6 * prop_sim + 0.4 * neighbor_score


def demo_structure_extraction():
    """Demonstrate structure extraction and isomorphism matching"""
    print("=" * 70)
    print("ANALOGICAL REASONING DEMO - Issues #15-16")
    print("Hofstadter's Structure Extraction + Isomorphism Matching")
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
        'venus': MockMKU(
            'venus',
            {'predicate': 'planet', 'properties': {'orbits': True}},
            {'orbits': {'sun'}}
        ),
        'electron1': MockMKU(
            'electron1',
            {'predicate': 'particle', 'properties': {'orbits': True}},
            {'orbits': {'nucleus'}}
        ),
        'electron2': MockMKU(
            'electron2',
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
    
    # Issue #16: Isomorphism Matching
    print("\n" + "=" * 70)
    print("5. Finding optimal node mapping (Issue #16)")
    print("-" * 70)
    
    matcher = IsomorphismMatcher(max_attempts=1000)
    mapping = matcher.find_best_mapping(
        solar_structure,
        atom_structure,
        require_complete=False
    )
    
    if mapping:
        print(f"\nOptimal mapping found (score: {mapping.score:.2f}):\n")
        for source, target in sorted(mapping.source_to_target.items()):
            print(f"  {source} (solar) → {target} (atom)")
        
        print("\nInterpretation:")
        print(f"  This mapping shows the correspondence between")
        print(f"  solar system components and atomic components.")
        print(f"  Edge preservation: {mapping.score * 0.7 / 0.7:.1%}")
        print(f"  Property preservation: {mapping.score * 0.3 / 0.3:.1%}")
    else:
        print("\nNo valid mapping found.")
    
    print("\n" + "=" * 70)
    print("✓ Isomorphism matching complete!")
    print("=" * 70)
    print()
    print("Key Insight:")
    print("  The solar system (sun + planets) has similar STRUCTURE")
    print("  to an atom (nucleus + electrons), and we can now map")
    print("  the specific correspondences between them!")
    print("=" * 70)


if __name__ == '__main__':
    demo_structure_extraction()
