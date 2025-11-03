#!/usr/bin/env python3
"""
Physics Domain Consciousness Experiment

Tests consciousness metrics in a physics knowledge domain,
comparing with the mathematics domain results.

Expected: Different domains may show different consciousness profiles
based on the structure of knowledge relationships.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric


def create_physics_knowledge():
    """Create comprehensive physics domain knowledge"""
    
    concepts = {
        # Classical Mechanics
        'force': {
            'predicate': 'physical_quantity',
            'arguments': ['vector', 'causes_acceleration'],
            'properties': {
                'units': 'newtons',
                'equation': 'F = ma',
                'type': 'vector'
            }
        },
        'mass': {
            'predicate': 'physical_quantity',
            'arguments': ['scalar', 'inertia'],
            'properties': {
                'units': 'kilograms',
                'conserved': True,
                'type': 'scalar'
            }
        },
        'acceleration': {
            'predicate': 'physical_quantity',
            'arguments': ['vector', 'rate_of_change'],
            'properties': {
                'units': 'm/s²',
                'derivative_of': 'velocity',
                'type': 'vector'
            }
        },
        'velocity': {
            'predicate': 'physical_quantity',
            'arguments': ['vector', 'rate_of_change'],
            'properties': {
                'units': 'm/s',
                'derivative_of': 'position',
                'integral_of': 'acceleration',
                'type': 'vector'
            }
        },
        'momentum': {
            'predicate': 'physical_quantity',
            'arguments': ['vector', 'conserved'],
            'properties': {
                'units': 'kg·m/s',
                'equation': 'p = mv',
                'conserved': True,
                'type': 'vector'
            }
        },
        
        # Energy
        'energy': {
            'predicate': 'physical_quantity',
            'arguments': ['scalar', 'conserved'],
            'properties': {
                'units': 'joules',
                'conserved': True,
                'type': 'scalar',
                'fundamental': True
            }
        },
        'kinetic_energy': {
            'predicate': 'energy_type',
            'arguments': ['motion', 'translational'],
            'properties': {
                'equation': 'KE = 1/2 mv²',
                'depends_on': ['mass', 'velocity'],
                'is_a': 'energy'
            }
        },
        'potential_energy': {
            'predicate': 'energy_type',
            'arguments': ['position', 'stored'],
            'properties': {
                'equation': 'PE = mgh',
                'depends_on': ['mass', 'position'],
                'is_a': 'energy'
            }
        },
        
        # Thermodynamics
        'temperature': {
            'predicate': 'physical_quantity',
            'arguments': ['scalar', 'thermal'],
            'properties': {
                'units': 'kelvin',
                'measures': 'thermal_energy',
                'type': 'scalar'
            }
        },
        'entropy': {
            'predicate': 'physical_quantity',
            'arguments': ['scalar', 'disorder'],
            'properties': {
                'units': 'J/K',
                'second_law': 'increases',
                'type': 'scalar',
                'statistical': True
            }
        },
        
        # Electromagnetism
        'electric_field': {
            'predicate': 'field',
            'arguments': ['vector', 'electromagnetic'],
            'properties': {
                'units': 'V/m',
                'source': 'charge',
                'type': 'vector'
            }
        },
        'magnetic_field': {
            'predicate': 'field',
            'arguments': ['vector', 'electromagnetic'],
            'properties': {
                'units': 'tesla',
                'source': 'moving_charge',
                'type': 'vector'
            }
        },
        'electromagnetic_wave': {
            'predicate': 'wave',
            'arguments': ['transverse', 'propagates'],
            'properties': {
                'speed': 'c',
                'components': ['electric_field', 'magnetic_field'],
                'includes': ['light', 'radio']
            }
        },
        
        # Quantum Mechanics
        'wave_function': {
            'predicate': 'quantum_state',
            'arguments': ['probability_amplitude', 'complex'],
            'properties': {
                'symbol': 'ψ',
                'normalized': True,
                'contains': 'all_information'
            }
        },
        'quantum_superposition': {
            'predicate': 'quantum_principle',
            'arguments': ['multiple_states', 'simultaneous'],
            'properties': {
                'counterintuitive': True,
                'collapses_on': 'measurement',
                'fundamental': True
            }
        },
        'quantum_entanglement': {
            'predicate': 'quantum_principle',
            'arguments': ['correlation', 'non_local'],
            'properties': {
                'spooky': True,
                'no_faster_than_light': True,
                'fundamental': True
            }
        },
        
        # Relativity
        'spacetime': {
            'predicate': 'fundamental_structure',
            'arguments': ['4d_continuum', 'curved'],
            'properties': {
                'dimensions': 4,
                'curved_by': 'mass_energy',
                'unifies': ['space', 'time']
            }
        },
        'special_relativity': {
            'predicate': 'physical_theory',
            'arguments': ['constant_c', 'inertial_frames'],
            'properties': {
                'postulate_1': 'physics_same_all_frames',
                'postulate_2': 'speed_of_light_constant',
                'implies': ['time_dilation', 'length_contraction']
            }
        },
        'general_relativity': {
            'predicate': 'physical_theory',
            'arguments': ['gravity', 'curved_spacetime'],
            'properties': {
                'principle': 'equivalence',
                'equation': 'Einstein_field_equations',
                'unifies': ['gravity', 'geometry']
            }
        },
        
        # Conservation Laws
        'conservation_of_energy': {
            'predicate': 'conservation_law',
            'arguments': ['energy', 'isolated_system'],
            'properties': {
                'symmetry': 'time_translation',
                'applies_to': 'energy',
                'violation': 'never_observed'
            }
        },
        'conservation_of_momentum': {
            'predicate': 'conservation_law',
            'arguments': ['momentum', 'isolated_system'],
            'properties': {
                'symmetry': 'space_translation',
                'applies_to': 'momentum',
                'violation': 'never_observed'
            }
        },
        
        # Meta-physics concepts
        'physical_law': {
            'predicate': 'meta_concept',
            'arguments': ['universal', 'mathematical'],
            'properties': {
                'describes': 'nature',
                'testable': True,
                'falsifiable': True
            }
        },
        'measurement': {
            'predicate': 'process',
            'arguments': ['observation', 'quantification'],
            'properties': {
                'affects': 'quantum_systems',
                'requires': 'instrument',
                'produces': 'data'
            }
        },
        'causality': {
            'predicate': 'principle',
            'arguments': ['cause_precedes_effect', 'fundamental'],
            'properties': {
                'limited_by': 'speed_of_light',
                'violated_by': 'none_observed',
                'fundamental': True
            }
        }
    }
    
    return concepts


def run_physics_domain_experiment():
    """Run consciousness measurement on physics domain"""
    
    print("=" * 70)
    print("PHYSICS DOMAIN CONSCIOUSNESS EXPERIMENT")
    print("Testing consciousness metrics in physics knowledge domain")
    print("=" * 70)
    print()
    
    # Initialize system
    kg = KnowledgeGraph()
    recursion = RecursionDepthMetric()
    
    # Create physics knowledge
    print("1. Building physics knowledge base...")
    physics_concepts = create_physics_knowledge()
    
    for concept_id, deep_structure in physics_concepts.items():
        mku = MonadicKnowledgeUnit(
            concept_id=concept_id,
            deep_structure=deep_structure
        )
        # Enable self-modeling for consciousness
        mku.create_self_model()
        kg.add_concept(mku)
    
    print(f"   Created {len(physics_concepts)} physics concepts")
    print()
    
    # Trigger recursive reasoning about physics
    print("2. Triggering recursive reasoning about physics...")
    
    # Meta-reasoning about physical laws
    recursion.record_recursion_event(
        "meta_analyze",
        "reason_about_physical_laws",
        {"physical_law", "conservation_of_energy", "causality"}
    )
    
    # Self-modeling: physics reasoning about physics
    recursion.record_recursion_event(
        "self_model",
        "physics_models_physics",
        {"physical_law", "measurement", "spacetime"}
    )
    
    # Strange loop: measurement affects quantum systems
    recursion.record_recursion_event(
        "strange_loop",
        "measurement_problem",
        {"measurement", "wave_function", "quantum_superposition"}
    )
    
    # Meta-level: Theories about theories
    recursion.record_recursion_event(
        "meta_analyze",
        "theory_structure",
        {"special_relativity", "general_relativity", "physical_law"}
    )
    
    # Deep recursion: Quantum entanglement paradox
    recursion.record_recursion_event(
        "strange_loop",
        "entanglement_paradox",
        {"quantum_entanglement", "causality", "measurement"}
    )
    
    # Measure depth after triggering events
    depth = recursion.measure_current_depth()
    print(f"   Triggered 5 recursion events")
    print(f"   Current recursion depth: {depth}")
    print()
    
    # Measure consciousness
    print("3. Measuring consciousness in physics domain...")
    profile = measure_consciousness(kg, recursion)
    print()
    
    # Display results
    print("=" * 70)
    print("PHYSICS DOMAIN CONSCIOUSNESS PROFILE")
    print("=" * 70)
    print()
    
    print(f"Overall Consciousness: {profile.overall_consciousness_score:.2%}")
    print(f"Verdict: {profile.consciousness_verdict}")
    print()
    
    print("Component Breakdown:")
    print("-" * 70)
    
    # Recursion
    recursion_score = profile.recursion_metrics['consciousness']['score']
    print(f"Recursion (30%):     {recursion_score:.2%}")
    print(f"  - Max depth:       {profile.recursion_metrics['recursion_depth']['max']}")
    print(f"  - Meta-level:      {profile.recursion_metrics['meta_level']['current']}")
    print(f"  - Self-refs:       {profile.recursion_metrics['self_awareness']['self_reference_count']}")
    print()
    
    # Integration
    print(f"Integration (25%):   {profile.integration.phi:.3f}")
    print()
    
    # Causality
    print(f"Causality (20%):     {profile.causality.causal_density:.3f}")
    print()
    
    # Understanding
    understanding_score = profile.understanding['overall_score']
    print(f"Understanding (25%): {understanding_score:.2%}")
    print()
    
    # Domain-specific insights
    print("=" * 70)
    print("DOMAIN-SPECIFIC INSIGHTS")
    print("=" * 70)
    print()
    
    # Analyze physics-specific patterns
    print("Physics Domain Characteristics:")
    print(f"  - Concepts: {len(kg.nodes)}")
    
    # Count relations
    total_relations = sum(
        len(relations) 
        for mku in kg.nodes.values() 
        for relations in mku.relations.values()
    )
    print(f"  - Relations: {total_relations}")
    
    # Find most connected concepts
    if kg.nodes:
        degree_centrality = {
            concept_id: sum(len(r) for r in mku.relations.values())
            for concept_id, mku in kg.nodes.items()
        }
        most_connected = sorted(
            degree_centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        if most_connected:
            print(f"\n  Most connected concepts:")
            for concept, degree in most_connected:
                if degree > 0:
                    print(f"    - {concept}: {degree} connections")
    
    # Quantum effects
    print("\n  Quantum consciousness markers:")
    quantum_concepts = [c for c in kg.nodes if 'quantum' in c.lower()]
    print(f"    - Quantum concepts: {len(quantum_concepts)}")
    if quantum_concepts:
        print(f"    - Examples: {', '.join(quantum_concepts[:3])}")
    
    # Relativity effects
    print("\n  Relativistic consciousness markers:")
    relativity_concepts = [c for c in kg.nodes if 'relativity' in c.lower() or c == 'spacetime']
    print(f"    - Relativity concepts: {len(relativity_concepts)}")
    if relativity_concepts:
        print(f"    - Examples: {', '.join(relativity_concepts)}")
    
    print()
    
    # Save results
    results = {
        'domain': 'physics',
        'timestamp': datetime.now().isoformat(),
        'consciousness': {
            'overall': profile.overall_consciousness_score,
            'verdict': profile.consciousness_verdict,
            'recursion': recursion_score,
            'integration': profile.integration.phi,
            'causality': profile.causality.causal_density,
            'understanding': understanding_score
        },
        'metrics': {
            'concepts': len(kg.nodes),
            'relations': total_relations,
            'recursion_depth': recursion.profile.max_depth,
            'meta_level': recursion.profile.meta_level.name
        }
    }
    
    output_file = 'physics_domain_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Comparison prompt
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("Compare with mathematics domain:")
    print("  python experiments/mathematics_domain.py")
    print()
    print("Compare with biology domain (create next):")
    print("  python experiments/biology_domain.py")
    print()
    
    return profile


if __name__ == '__main__':
    profile = run_physics_domain_experiment()
