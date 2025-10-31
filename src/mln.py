#!/usr/bin/env python3
"""
Monad-Loop Network (MLN): Proof of Concept
A self-referential knowledge system combining GEB, Chomsky, and Leibniz principles
"""

from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

# Optional GPU acceleration
try:
    from .gpu_similarity import GPUStructuralSimilarity
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ============================================================================
# 1. MONADIC KNOWLEDGE UNITS (Leibniz)
# ============================================================================

@dataclass
class MonadicKnowledgeUnit:
    """
    A self-contained concept that reflects the universe from its perspective.
    Unlike embeddings, MKUs have operational semantics.
    """
    concept_id: str
    deep_structure: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Callable] = field(default_factory=list)
    relations: Dict[str, Set[str]] = field(default_factory=dict)
    meta_model: Optional['MetaRepresentation'] = None
    
    def __post_init__(self):
        # Initialize default deep structure
        if not self.deep_structure:
            self.deep_structure = {
                'predicate': None,
                'arguments': [],
                'properties': {},
                'constraints': []
            }
    
    def reflect_universe(self, knowledge_graph: 'KnowledgeGraph'):
        """
        Leibniz's pre-established harmony: 
        Each monad reflects the entire universe from its perspective
        """
        # Establish relations based on structural similarity
        for other_id, other_mku in knowledge_graph.nodes.items():
            if other_id == self.concept_id:
                continue
            
            similarity = self._structural_similarity(other_mku)
            if similarity > 0.3:  # Threshold for establishing relation
                relation_type = self._infer_relation_type(other_mku)
                if relation_type not in self.relations:
                    self.relations[relation_type] = set()
                self.relations[relation_type].add(other_id)
    
    def _structural_similarity(self, other: 'MonadicKnowledgeUnit') -> float:
        """Compute structural similarity (simple version)"""
        # Compare deep structure properties
        my_props = set(self.deep_structure.get('properties', {}).keys())
        other_props = set(other.deep_structure.get('properties', {}).keys())
        
        if not my_props and not other_props:
            return 0.0
        
        intersection = my_props.intersection(other_props)
        union = my_props.union(other_props)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _infer_relation_type(self, other: 'MonadicKnowledgeUnit') -> str:
        """Infer type of relationship based on structure"""
        # Simple heuristic: check predicate compatibility
        my_pred = self.deep_structure.get('predicate')
        other_pred = other.deep_structure.get('predicate')
        
        if my_pred and other_pred:
            if 'is_a' in str(my_pred).lower():
                return 'subtype'
            elif 'has' in str(my_pred).lower():
                return 'composition'
            else:
                return 'association'
        return 'related_to'
    
    def generate_surface_form(self, modality: str = 'text') -> str:
        """
        Chomsky transformation: Deep structure → Surface structure
        Same meaning, different surface realizations
        """
        if modality == 'text':
            return self._generate_text()
        elif modality == 'logic':
            return self._generate_logic()
        elif modality == 'code':
            return self._generate_code()
        else:
            return str(self.deep_structure)
    
    def _generate_text(self) -> str:
        pred = self.deep_structure.get('predicate', self.concept_id)
        args = self.deep_structure.get('arguments', [])
        
        if args:
            return f"{pred}({', '.join(map(str, args))})"
        return str(pred)
    
    def _generate_logic(self) -> str:
        pred = self.deep_structure.get('predicate', self.concept_id)
        args = self.deep_structure.get('arguments', [])
        return f"∀x: {pred}(x) → {' ∧ '.join(map(str, args))}"
    
    def _generate_code(self) -> str:
        return f"class {self.concept_id}:\n    pass  # {self.deep_structure}"
    
    def create_self_model(self) -> 'MetaRepresentation':
        """GEB: Create representation of itself for meta-reasoning"""
        self.meta_model = MetaRepresentation(self)
        return self.meta_model


@dataclass
class MetaRepresentation:
    """A model of an MKU (for self-reference)"""
    target_mku: MonadicKnowledgeUnit
    
    def introspect_structure(self) -> Dict:
        """Examine own structure"""
        return {
            'concept_id': self.target_mku.concept_id,
            'deep_structure': self.target_mku.deep_structure,
            'relation_count': sum(len(v) for v in self.target_mku.relations.values()),
            'has_meta_model': self.target_mku.meta_model is not None
        }
    
    def introspect_capabilities(self) -> List[str]:
        """What can this MKU do?"""
        capabilities = []
        if self.target_mku.transformations:
            capabilities.append('transform')
        if self.target_mku.relations:
            capabilities.append('relate')
        if self.target_mku.deep_structure.get('predicate'):
            capabilities.append('predicate')
        return capabilities


# ============================================================================
# 2. KNOWLEDGE GRAPH WITH OPERATIONAL SEMANTICS
# ============================================================================

class KnowledgeGraph:
    """
    Graph where nodes are MKUs (operational) not just embeddings (static)
    
    Now with GPU acceleration for pre-established harmony (Issue #1)
    """
    def __init__(self, use_gpu: bool = True, device: str = 'auto'):
        self.nodes: Dict[str, MonadicKnowledgeUnit] = {}
        self.inference_rules: List['InferenceRule'] = []
        
        # GPU acceleration for similarity computation
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            self.gpu_similarity = GPUStructuralSimilarity(device=device)
            print(f"KnowledgeGraph using GPU acceleration: {self.gpu_similarity.device}")
        else:
            self.gpu_similarity = None
            if use_gpu and not GPU_AVAILABLE:
                print("Warning: GPU requested but not available, using CPU")
    
    def add_concept(self, mku: MonadicKnowledgeUnit):
        """Add concept and establish pre-established harmony (GPU-accelerated)"""
        self.nodes[mku.concept_id] = mku
        
        if self.use_gpu and len(self.nodes) > 1:
            # GPU-accelerated similarity computation
            self._gpu_reflect_universe(mku)
        else:
            # Original CPU method
            mku.reflect_universe(self)
            
            # Update other nodes' perspectives
            for other_mku in self.nodes.values():
                if other_mku.concept_id != mku.concept_id:
                    other_mku.reflect_universe(self)
    
    def _gpu_reflect_universe(self, mku: MonadicKnowledgeUnit):
        """GPU-accelerated pre-established harmony (Issue #1 optimization)"""
        # Get all other concepts
        other_concepts = [m for m in self.nodes.values() if m.concept_id != mku.concept_id]
        if not other_concepts:
            return
        
        # Batch similarity computation on GPU
        other_structures = [m.deep_structure for m in other_concepts]
        similarities = self.gpu_similarity.batch_similarity(
            mku.deep_structure,
            other_structures
        )
        
        # Establish relations based on similarity
        threshold = 0.3
        for other_mku, similarity in zip(other_concepts, similarities):
            if similarity > threshold:
                relation_type = mku._infer_relation_type(other_mku)
                if relation_type not in mku.relations:
                    mku.relations[relation_type] = set()
                mku.relations[relation_type].add(other_mku.concept_id)
                
                # Bidirectional relation
                if relation_type not in other_mku.relations:
                    other_mku.relations[relation_type] = set()
                other_mku.relations[relation_type].add(mku.concept_id)
    
    def query(self, start_id: str, target_id: str) -> 'InferenceChain':
        """
        Not just path-finding, but valid inference chain
        """
        if start_id not in self.nodes or target_id not in self.nodes:
            return InferenceChain([])
        
        # BFS with inference rules
        queue = [(self.nodes[start_id], [])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            
            if current.concept_id == target_id:
                return InferenceChain(path + [current])
            
            if current.concept_id in visited:
                continue
            visited.add(current.concept_id)
            
            # Follow relations
            for relation_type, related_ids in current.relations.items():
                for related_id in related_ids:
                    if related_id not in visited and related_id in self.nodes:
                        next_node = self.nodes[related_id]
                        queue.append((next_node, path + [current]))
        
        return InferenceChain([])  # No path found
    
    def add_inference_rule(self, rule: 'InferenceRule'):
        """Add a transformation rule for reasoning"""
        self.inference_rules.append(rule)
    
    def apply_inference(self, premise: MonadicKnowledgeUnit) -> List[MonadicKnowledgeUnit]:
        """Apply inference rules to derive new knowledge"""
        conclusions = []
        for rule in self.inference_rules:
            if rule.can_apply(premise):
                conclusion = rule.apply(premise, self)
                if conclusion:
                    conclusions.append(conclusion)
        return conclusions


@dataclass
class InferenceChain:
    """A chain of reasoning (not just an answer)"""
    steps: List[MonadicKnowledgeUnit]
    
    def explain(self) -> str:
        """Generate explanation of reasoning"""
        if not self.steps:
            return "No inference path found"
        
        explanation = "Reasoning chain:\n"
        for i, step in enumerate(self.steps, 1):
            explanation += f"  {i}. {step.concept_id}\n"
        return explanation
    
    def is_valid(self) -> bool:
        """Meta-reasoning: is this chain valid?"""
        # Check each step is connected
        for i in range(len(self.steps) - 1):
            current = self.steps[i]
            next_step = self.steps[i + 1]
            
            # Check if there's a relation
            related_ids = set()
            for rel_set in current.relations.values():
                related_ids.update(rel_set)
            
            if next_step.concept_id not in related_ids:
                return False
        
        return True


# ============================================================================
# 3. INFERENCE RULES (Chomsky-style transformations)
# ============================================================================

class InferenceRule(ABC):
    """Abstract inference rule"""
    @abstractmethod
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        pass
    
    @abstractmethod
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        pass


class TransitivityRule(InferenceRule):
    """If A→B and B→C, then A→C"""
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        return bool(premise.relations)
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        # Find transitive closure
        for rel_type, related_ids in premise.relations.items():
            for related_id in related_ids:
                if related_id in kg.nodes:
                    related = kg.nodes[related_id]
                    # Check if related has further relations
                    if related.relations:
                        # Create new inference
                        new_mku = MonadicKnowledgeUnit(
                            concept_id=f"{premise.concept_id}_implies_{list(related.relations.values())[0]}",
                            deep_structure={
                                'predicate': 'transitive_closure',
                                'arguments': [premise.concept_id, related_id]
                            }
                        )
                        return new_mku
        return None


class SubstitutionRule(InferenceRule):
    """Substitute equivalent concepts"""
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        return 'equivalence' in premise.relations
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        if 'equivalence' not in premise.relations:
            return None
        
        equiv_ids = premise.relations['equivalence']
        if equiv_ids:
            equiv_id = list(equiv_ids)[0]
            # Create substituted version
            new_mku = MonadicKnowledgeUnit(
                concept_id=f"substituted_{premise.concept_id}",
                deep_structure=kg.nodes[equiv_id].deep_structure.copy()
            )
            return new_mku
        return None


class ModusPonensRule(InferenceRule):
    """
    Modus Ponens: If A→B and A, then B
    If we have an implication and its antecedent, infer the consequent
    
    Issue #4
    """
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        # Check if premise has 'implies' relation
        return 'implies' in premise.relations or 'subtype' in premise.relations
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        # Look for implies or subtype relations
        for rel_type in ['implies', 'subtype']:
            if rel_type not in premise.relations:
                continue
            
            related_ids = premise.relations[rel_type]
            if related_ids:
                # Get the first consequent
                consequent_id = list(related_ids)[0]
                if consequent_id in kg.nodes:
                    consequent = kg.nodes[consequent_id]
                    # Create new inferred concept
                    new_mku = MonadicKnowledgeUnit(
                        concept_id=f"inferred_{consequent_id}_from_{premise.concept_id}",
                        deep_structure={
                            'predicate': 'modus_ponens',
                            'arguments': [premise.concept_id, consequent_id],
                            'properties': consequent.deep_structure.get('properties', {})
                        }
                    )
                    return new_mku
        return None


class ContrapositionRule(InferenceRule):
    """
    Contraposition: If A→B, then ¬B→¬A
    From an implication, derive its contrapositive
    
    Issue #4
    """
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        return 'implies' in premise.relations or 'subtype' in premise.relations
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        for rel_type in ['implies', 'subtype']:
            if rel_type not in premise.relations:
                continue
            
            related_ids = premise.relations[rel_type]
            if related_ids:
                consequent_id = list(related_ids)[0]
                # Create contrapositive: not B implies not A
                new_mku = MonadicKnowledgeUnit(
                    concept_id=f"not_{consequent_id}_implies_not_{premise.concept_id}",
                    deep_structure={
                        'predicate': 'contrapositive',
                        'arguments': [f'¬{consequent_id}', f'¬{premise.concept_id}'],
                        'properties': {'negation': True}
                    }
                )
                return new_mku
        return None


class SymmetryRule(InferenceRule):
    """
    Symmetry: If A relates B (symmetric relation), then B relates A
    For relations that are inherently symmetric (like 'sibling', 'equivalent')
    
    Issue #4
    """
    # Define which relations are symmetric
    SYMMETRIC_RELATIONS = {'equivalence', 'sibling', 'peer', 'similar_to'}
    
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        # Check if premise has any symmetric relations
        return any(rel in self.SYMMETRIC_RELATIONS for rel in premise.relations.keys())
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        for rel_type in self.SYMMETRIC_RELATIONS:
            if rel_type not in premise.relations:
                continue
            
            related_ids = premise.relations[rel_type]
            if related_ids:
                # Get the first related concept
                related_id = list(related_ids)[0]
                if related_id in kg.nodes:
                    related = kg.nodes[related_id]
                    
                    # Check if reverse relation already exists
                    if rel_type in related.relations and premise.concept_id in related.relations[rel_type]:
                        continue  # Already symmetric
                    
                    # Create symmetric inference
                    new_mku = MonadicKnowledgeUnit(
                        concept_id=f"symmetric_{related_id}_{premise.concept_id}",
                        deep_structure={
                            'predicate': 'symmetric_relation',
                            'arguments': [related_id, premise.concept_id],
                            'properties': {'relation_type': rel_type}
                        }
                    )
                    return new_mku
        return None


class CompositionRule(InferenceRule):
    """
    Composition: Combine multiple inference rules
    If we can apply two rules in sequence, compose them
    
    Issue #4
    """
    def __init__(self, rule1: InferenceRule, rule2: InferenceRule):
        self.rule1 = rule1
        self.rule2 = rule2
    
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        # Can apply if first rule is applicable
        return self.rule1.can_apply(premise)
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph) -> Optional[MonadicKnowledgeUnit]:
        # Apply first rule
        intermediate = self.rule1.apply(premise, kg)
        if not intermediate:
            return None
        
        # Try to apply second rule to intermediate result
        if self.rule2.can_apply(intermediate):
            final = self.rule2.apply(intermediate, kg)
            if final:
                # Mark as composed inference
                final.deep_structure['composed'] = True
                final.deep_structure['composition'] = [
                    self.rule1.__class__.__name__,
                    self.rule2.__class__.__name__
                ]
                return final
        
        return intermediate  # Return intermediate if second rule doesn't apply


# ============================================================================
# 4. STRANGE LOOP PROCESSOR (GEB Self-Reference)
# ============================================================================

class StrangeLoopProcessor:
    """
    Implements self-reference and meta-reasoning
    The system can reason about its own reasoning
    """
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.meta_kg: Optional['MetaKnowledgeGraph'] = None
        self.reasoning_trace: List[str] = []
    
    def create_strange_loop(self):
        """Create self-referential structure"""
        self.meta_kg = MetaKnowledgeGraph(self.kg)
        
        # The loop: KG can query its own structure
        meta_mku = MonadicKnowledgeUnit(
            concept_id='self',
            deep_structure={
                'predicate': 'knowledge_graph',
                'arguments': list(self.kg.nodes.keys())
            }
        )
        meta_mku.create_self_model()
        self.kg.add_concept(meta_mku)
    
    def introspect(self, query: str) -> Dict:
        """
        Meta-reasoning: Examine own reasoning process
        "Why did I answer X?"
        """
        result = {
            'query': query,
            'reasoning_trace': self.reasoning_trace.copy(),
            'graph_state': {
                'num_concepts': len(self.kg.nodes),
                'num_rules': len(self.kg.inference_rules)
            }
        }
        
        if self.meta_kg:
            result['meta_analysis'] = self.meta_kg.analyze()
        
        return result
    
    def detect_inconsistency(self) -> List[str]:
        """Meta-reasoning: Find inconsistencies in knowledge"""
        inconsistencies = []
        
        # Check for cycles in "subtype" relations (would be inconsistent)
        for node_id, node in self.kg.nodes.items():
            if 'subtype' in node.relations:
                # Simple cycle detection
                visited = {node_id}
                current = node
                
                while 'subtype' in current.relations:
                    subtypes = current.relations['subtype']
                    if not subtypes:
                        break
                    
                    next_id = list(subtypes)[0]
                    if next_id in visited:
                        inconsistencies.append(f"Cycle detected: {' → '.join(visited)} → {next_id}")
                        break
                    
                    visited.add(next_id)
                    if next_id not in self.kg.nodes:
                        break
                    current = self.kg.nodes[next_id]
        
        return inconsistencies
    
    def godel_sentence(self) -> str:
        """
        Construct self-referential statement
        "This statement cannot be proven in this system"
        """
        return f"GODEL: The knowledge graph with {len(self.kg.nodes)} concepts cannot prove this statement"


class MetaKnowledgeGraph:
    """A model of the knowledge graph (for meta-reasoning)"""
    def __init__(self, target_kg: KnowledgeGraph):
        self.target = target_kg
    
    def analyze(self) -> Dict:
        """Analyze the structure of the knowledge graph"""
        analysis = {
            'total_concepts': len(self.target.nodes),
            'total_relations': sum(
                sum(len(v) for v in node.relations.values())
                for node in self.target.nodes.values()
            ),
            'avg_relations_per_concept': 0,
            'most_connected': None,
            'isolated_concepts': []
        }
        
        if self.target.nodes:
            relation_counts = {}
            for node_id, node in self.target.nodes.items():
                count = sum(len(v) for v in node.relations.values())
                relation_counts[node_id] = count
                
                if count == 0:
                    analysis['isolated_concepts'].append(node_id)
            
            if relation_counts:
                analysis['avg_relations_per_concept'] = sum(relation_counts.values()) / len(relation_counts)
                analysis['most_connected'] = max(relation_counts.items(), key=lambda x: x[1])[0]
        
        return analysis


# ============================================================================
# 5. HYBRID SYSTEM (LLM + Symbolic)
# ============================================================================

class HybridIntelligenceSystem:
    """
    Combines statistical pattern recognition with symbolic reasoning
    
    Now with GPU acceleration by default (Issue #1)
    """
    def __init__(self, use_gpu: bool = True, device: str = 'auto'):
        self.kg = KnowledgeGraph(use_gpu=use_gpu, device=device)
        self.slp = StrangeLoopProcessor(self.kg)
        self.slp.create_strange_loop()
        
        # Add inference rules (Issue #4 complete)
        self.kg.add_inference_rule(TransitivityRule())
        self.kg.add_inference_rule(SubstitutionRule())
        self.kg.add_inference_rule(ModusPonensRule())
        self.kg.add_inference_rule(ContrapositionRule())
        self.kg.add_inference_rule(SymmetryRule())
        # Note: CompositionRule requires two rules, can be added as needed
    
    def add_knowledge(self, concept_id: str, deep_structure: Dict):
        """Add new knowledge to the system"""
        mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
        self.kg.add_concept(mku)
    
    def query(self, question: str, start_concept: str, target_concept: str) -> Dict:
        """
        Process query with explainable reasoning
        """
        self.slp.reasoning_trace.append(f"Query: {question}")
        
        # Perform inference
        chain = self.kg.query(start_concept, target_concept)
        
        # Validate with meta-reasoning
        is_valid = chain.is_valid()
        
        # Apply inference rules
        if chain.steps:
            inferred = self.kg.apply_inference(chain.steps[0])
        else:
            inferred = []
        
        result = {
            'question': question,
            'inference_chain': chain.explain(),
            'is_valid': is_valid,
            'additional_inferences': [i.concept_id for i in inferred],
            'meta_analysis': self.slp.introspect(question)
        }
        
        return result
    
    def explain_reasoning(self) -> str:
        """Generate human-readable explanation"""
        trace = self.slp.introspect("explain")
        return json.dumps(trace, indent=2)
    
    def detect_inconsistencies(self) -> List[str]:
        """Find logical inconsistencies"""
        return self.slp.detect_inconsistency()


# ============================================================================
# 6. DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate the self-referential knowledge system"""
    print("=" * 70)
    print("SELF-REFERENTIAL KNOWLEDGE SYSTEM DEMO")
    print("Combining GEB (strange loops) + Chomsky (deep structure) + Leibniz (monads)")
    print("=" * 70)
    print()
    
    # Create system
    system = HybridIntelligenceSystem()
    
    # Add knowledge about mammals
    print("1. Adding knowledge: Mammals")
    system.add_knowledge('mammal', {
        'predicate': 'is_alive',
        'arguments': ['warm_blooded', 'vertebrate'],
        'properties': {'breathes': 'air', 'has': 'hair'}
    })
    
    system.add_knowledge('dog', {
        'predicate': 'is_a',
        'arguments': ['mammal'],
        'properties': {'domesticated': True, 'barks': True}
    })
    
    system.add_knowledge('cat', {
        'predicate': 'is_a',
        'arguments': ['mammal'],
        'properties': {'domesticated': True, 'meows': True}
    })
    
    system.add_knowledge('animal', {
        'predicate': 'living_thing',
        'arguments': ['mobile', 'consumes_food'],
        'properties': {'alive': True}
    })
    
    # Query the system
    print("\n2. Querying: Is a dog related to an animal?")
    result = system.query(
        "Is a dog an animal?",
        start_concept='dog',
        target_concept='animal'
    )
    print(json.dumps(result, indent=2))
    
    # Show surface realizations (Chomsky)
    print("\n3. Surface realizations (same deep structure, different forms):")
    dog_mku = system.kg.nodes['dog']
    print(f"   Text:  {dog_mku.generate_surface_form('text')}")
    print(f"   Logic: {dog_mku.generate_surface_form('logic')}")
    print(f"   Code:  {dog_mku.generate_surface_form('code')}")
    
    # Meta-reasoning (GEB strange loop)
    print("\n4. Meta-reasoning (strange loop - system examining itself):")
    print(f"   Gödel sentence: {system.slp.godel_sentence()}")
    
    if system.slp.meta_kg:
        analysis = system.slp.meta_kg.analyze()
        print(f"\n   Graph analysis:")
        print(f"   - Total concepts: {analysis['total_concepts']}")
        print(f"   - Avg relations: {analysis['avg_relations_per_concept']:.2f}")
        print(f"   - Most connected: {analysis['most_connected']}")
    
    # Self-reference (monad reflecting universe)
    print("\n5. Monadic reflection (each concept's view of the universe):")
    dog_mku.create_self_model()
    if dog_mku.meta_model:
        introspection = dog_mku.meta_model.introspect_structure()
        print(f"   Dog's self-model: {json.dumps(introspection, indent=4)}")
    
    # Inconsistency detection
    print("\n6. Inconsistency detection:")
    inconsistencies = system.detect_inconsistencies()
    if inconsistencies:
        print(f"   Found: {inconsistencies}")
    else:
        print("   No inconsistencies detected")
    
    print("\n" + "=" * 70)
    print("Key differences from statistical LLMs:")
    print("  ✓ Explicit reasoning chains (not black box)")
    print("  ✓ Meta-reasoning (system examines its own inference)")
    print("  ✓ Structural knowledge (not just correlations)")
    print("  ✓ Multiple surface realizations from same deep structure")
    print("  ✓ Self-reference and introspection")
    print("=" * 70)


if __name__ == '__main__':
    demo()
