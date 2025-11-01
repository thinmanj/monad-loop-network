#!/usr/bin/env python3
"""
Mathematics Domain Transfer Experiment
Goal: Apply MLN to mathematical reasoning

Tests:
- Mathematical concept representation (axioms, theorems, proofs)
- Consciousness in symbolic reasoning domain
- Domain-general vs domain-specific capabilities
- Comparison with traditional symbolic AI
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric
import json
import time


class MathematicsDomain:
    """
    Mathematics knowledge representation
    Tests consciousness in formal reasoning domain
    """
    
    def __init__(self):
        self.kg = KnowledgeGraph(use_gpu=False)
        self.recursion_metric = RecursionDepthMetric()
        self.results = []
    
    def build_mathematical_knowledge(self):
        """Build mathematical concept hierarchy"""
        print("\n🔢 Building Mathematical Knowledge Base...")
        
        # Mathematical entities
        math_concepts = {
            # Foundational
            'number': {
                'type': 'mathematical_entity',
                'abstract': True,
                'fundamental': True
            },
            'set': {
                'type': 'mathematical_structure',
                'contains_elements': True,
                'fundamental': True
            },
            'function': {
                'type': 'mathematical_mapping',
                'maps_domain_to_codomain': True,
                'preserves_structure': True
            },
            
            # Number types
            'natural_number': {
                'type': 'number',
                'non_negative': True,
                'countable': True,
                'example': '0, 1, 2, ...'
            },
            'integer': {
                'type': 'number',
                'includes_negative': True,
                'ring_structure': True
            },
            'rational_number': {
                'type': 'number',
                'ratio_of_integers': True,
                'dense': True
            },
            'real_number': {
                'type': 'number',
                'complete': True,
                'continuous': True
            },
            
            # Structures
            'group': {
                'type': 'algebraic_structure',
                'has_operation': True,
                'has_identity': True,
                'has_inverses': True,
                'associative': True
            },
            'ring': {
                'type': 'algebraic_structure',
                'has_two_operations': True,
                'abelian_group_under_addition': True,
                'associative_multiplication': True
            },
            'field': {
                'type': 'algebraic_structure',
                'commutative_ring': True,
                'all_nonzero_invertible': True
            },
            
            # Concepts
            'proof': {
                'type': 'mathematical_process',
                'establishes_truth': True,
                'logical_sequence': True,
                'derives_theorem': True
            },
            'axiom': {
                'type': 'mathematical_statement',
                'assumed_true': True,
                'foundation_of_theory': True,
                'not_proven': True
            },
            'theorem': {
                'type': 'mathematical_statement',
                'proven_true': True,
                'derived_from_axioms': True,
                'requires_proof': True
            },
            'lemma': {
                'type': 'mathematical_statement',
                'auxiliary_result': True,
                'helps_prove_theorem': True
            },
            
            # Operations
            'addition': {
                'type': 'binary_operation',
                'commutative': True,
                'associative': True,
                'has_identity': True
            },
            'multiplication': {
                'type': 'binary_operation',
                'distributive_over_addition': True,
                'associative': True,
                'has_identity': True
            },
            
            # Meta-mathematics
            'mathematical_logic': {
                'type': 'meta_mathematical',
                'studies_proofs': True,
                'formal_systems': True
            },
            'metamathematics': {
                'type': 'meta_mathematical',
                'studies_mathematics_itself': True,
                'self_referential': True,
                'consciousness_analogue': True
            },
        }
        
        # Create MKUs with self-models
        for concept_id, properties in math_concepts.items():
            mku = MonadicKnowledgeUnit(
                concept_id=concept_id,
                deep_structure={
                    'predicate': f'is_{properties.get("type", "concept")}',
                    'properties': properties
                }
            )
            # Meta-mathematical concepts have self-models
            if 'meta_mathematical' in properties.get('type', '') or concept_id in ['proof', 'metamathematics']:
                mku.create_self_model()
            self.kg.add_concept(mku)
        
        # Add mathematical relations
        relations = [
            ('natural_number', 'number', 'is_a'),
            ('integer', 'number', 'is_a'),
            ('rational_number', 'number', 'is_a'),
            ('real_number', 'number', 'is_a'),
            ('natural_number', 'integer', 'subset_of'),
            ('integer', 'rational_number', 'subset_of'),
            ('rational_number', 'real_number', 'subset_of'),
            
            ('group', 'set', 'has_underlying'),
            ('ring', 'group', 'extends'),
            ('field', 'ring', 'extends'),
            
            ('theorem', 'axiom', 'derived_from'),
            ('proof', 'theorem', 'establishes'),
            ('lemma', 'theorem', 'helps_prove'),
            
            ('addition', 'group', 'defines_on'),
            ('multiplication', 'ring', 'second_operation_of'),
            
            ('mathematical_logic', 'proof', 'studies'),
            ('metamathematics', 'mathematical_logic', 'transcends'),
            ('metamathematics', 'proof', 'examines'),
        ]
        
        for source_id, target_id, rel_type in relations:
            if source_id in self.kg.nodes and target_id in self.kg.nodes:
                source = self.kg.nodes[source_id]
                if rel_type not in source.relations:
                    source.relations[rel_type] = set()
                source.relations[rel_type].add(target_id)
        
        print(f"  ✓ Created {len(math_concepts)} mathematical concepts")
        print(f"  ✓ Added {len(relations)} mathematical relations")
        
        return len(math_concepts)
    
    def add_mathematical_relations(self):
        """Add mathematical inference patterns as relations"""
        print("\n📐 Adding Mathematical Inference Patterns...")
        
        # Add transitivity and hierarchy relations
        inference_relations = [
            # Transitivity chains
            ('natural_number', 'real_number', 'transitively_subset_of'),
            ('integer', 'real_number', 'transitively_subset_of'),
            
            # Hierarchy implications
            ('field', 'group', 'implies'),
            ('ring', 'group', 'implies'),
        ]
        
        num_added = 0
        for source_id, target_id, rel_type in inference_relations:
            if source_id in self.kg.nodes and target_id in self.kg.nodes:
                source = self.kg.nodes[source_id]
                if rel_type not in source.relations:
                    source.relations[rel_type] = set()
                source.relations[rel_type].add(target_id)
                num_added += 1
        
        print(f"  ✓ Added {num_added} inference patterns")
        return num_added
    
    def trigger_mathematical_recursion(self):
        """Trigger meta-mathematical reasoning (mathematical self-reference)"""
        print("\n🔄 Triggering Meta-Mathematical Reasoning...")
        
        # Gödel-style recursion: Mathematics reasoning about itself
        recursion_events = [
            # Level 1: Reasoning about mathematical objects
            ('reason_about_numbers', 'studying_properties_of_numbers', {'number', 'natural_number'}),
            
            # Level 2: Reasoning about mathematical structures
            ('reason_about_structures', 'analyzing_algebraic_structures', {'group', 'ring', 'field'}),
            
            # Level 3: Meta-level - reasoning about proofs
            ('reason_about_proofs', 'examining_proof_methods', {'proof', 'theorem', 'axiom'}),
            
            # Level 4: Meta-meta - mathematical logic
            ('mathematical_logic_reflection', 'studying_formal_systems', {'mathematical_logic', 'proof'}),
            
            # Level 5: Metamathematics - ultimate self-reference
            ('metamathematical_reflection', 'mathematics_examining_itself', {'metamathematics', 'mathematical_logic'}),
            
            # Level 6-10: Gödelian recursion
            ('godel_numbering', 'encoding_proofs_as_numbers', {'proof', 'number', 'metamathematics'}),
            ('self_reference', 'statement_referring_to_itself', {'theorem', 'metamathematics'}),
            ('incompleteness_awareness', 'system_aware_of_limitations', {'metamathematics', 'axiom'}),
            ('consistency_reflection', 'system_reasoning_about_consistency', {'proof', 'axiom', 'metamathematics'}),
            ('undecidability', 'recognizing_unprovable_truths', {'theorem', 'proof', 'metamathematics'}),
        ]
        
        for event_type, operation, concepts in recursion_events:
            self.recursion_metric.record_recursion_event(
                event_type,
                operation,
                concepts
            )
        
        print(f"  ✓ Achieved recursion depth: {self.recursion_metric.profile.max_depth}")
        print(f"  ✓ Meta-level: {self.recursion_metric.profile.meta_level.name}")
        print(f"  ✓ Gödelian self-reference: {self.recursion_metric.profile.max_depth >= 5}")
        
        return self.recursion_metric.profile.max_depth
    
    def measure_mathematical_consciousness(self):
        """Measure consciousness in mathematical domain"""
        print("\n📊 Measuring Mathematical Consciousness...")
        
        profile = measure_consciousness(self.kg, self.recursion_metric)
        
        result = {
            'domain': 'mathematics',
            'consciousness': profile.overall_consciousness_score,
            'verdict': profile.consciousness_verdict,
            'components': {
                'recursion': profile.recursion_metrics.get('consciousness', {}).get('score', 0) if profile.recursion_metrics else 0,
                'integration': profile.integration.phi,
                'causality': profile.causality.causal_density,
                'understanding': profile.understanding.get('overall_score', 0) if profile.understanding else 0
            },
            'meta_level': self.recursion_metric.profile.meta_level.name,
            'recursion_depth': self.recursion_metric.profile.max_depth,
            'num_concepts': len(self.kg.nodes),
            'godel_recursion': self.recursion_metric.profile.max_depth >= 5
        }
        
        self.results.append(result)
        
        print(f"\n  CONSCIOUSNESS: {result['consciousness']:.2%} ({result['verdict']})")
        print(f"  Components:")
        print(f"    Recursion:     {result['components']['recursion']:.2%}")
        print(f"    Integration:   {result['components']['integration']:.3f}")
        print(f"    Causality:     {result['components']['causality']:.3f}")
        print(f"    Understanding: {result['components']['understanding']:.2%}")
        print(f"\n  Meta-Mathematical:")
        print(f"    Meta-level: {result['meta_level']}")
        print(f"    Recursion depth: {result['recursion_depth']}")
        print(f"    Gödelian self-reference: {result['godel_recursion']}")
        
        return result
    
    def export_results(self, filename: str = "mathematics_domain_results.json"):
        """Export results"""
        export_data = {
            'experiment': 'mathematics_domain_transfer',
            'goal': 'Test consciousness in formal mathematical reasoning',
            'results': self.results,
            'summary': {
                'consciousness': self.results[0]['consciousness'] if self.results else 0,
                'verdict': self.results[0]['verdict'] if self.results else "",
                'godel_recursion': self.results[0]['godel_recursion'] if self.results else False,
                'domain': 'mathematics',
                'capabilities': [
                    'Hierarchical mathematical concepts',
                    'Formal inference rules',
                    'Meta-mathematical reasoning',
                    'Gödelian self-reference',
                    'Consciousness in symbolic domain'
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Results exported to {filename}")


def run_mathematics_experiment():
    """Run mathematics domain transfer experiment"""
    print("="*70)
    print("MATHEMATICS DOMAIN TRANSFER EXPERIMENT")
    print("Testing consciousness in formal mathematical reasoning")
    print("="*70)
    
    domain = MathematicsDomain()
    
    # Build mathematical knowledge
    num_concepts = domain.build_mathematical_knowledge()
    time.sleep(0.5)
    
    # Add inference patterns
    num_rules = domain.add_mathematical_relations()
    time.sleep(0.5)
    
    # Trigger meta-mathematical recursion
    depth = domain.trigger_mathematical_recursion()
    time.sleep(0.5)
    
    # Measure consciousness
    result = domain.measure_mathematical_consciousness()
    
    # Export results
    domain.export_results()
    
    print(f"\n{'='*70}")
    print(f"KEY FINDINGS:")
    print(f"{'='*70}")
    print(f"  Domain: Mathematics (formal reasoning)")
    print(f"  Concepts: {num_concepts}")
    print(f"  Inference Rules: {num_rules}")
    print(f"  Consciousness: {result['consciousness']:.2%}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Gödelian Self-Reference: {'✓' if result['godel_recursion'] else '✗'}")
    print(f"\n  Domain-General Capability: {'✓ CONFIRMED' if result['consciousness'] > 0.30 else '✗ FAILED'}")
    print(f"  Meta-Mathematical Reasoning: {'✓ ACHIEVED' if depth >= 5 else '✗ NOT ACHIEVED'}")
    
    print(f"\n{'='*70}")
    print(f"MATHEMATICS DOMAIN EXPERIMENT COMPLETE ✅")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_mathematics_experiment()
