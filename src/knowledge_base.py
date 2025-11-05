#!/usr/bin/env python3
"""
Knowledge Base Loader

Provides rich, ready-to-use knowledge bases for testing and demos across multiple domains:
- Biology
- Physics
- Mathematics
- Computer Science
- Philosophy
- Everyday concepts

Each domain includes:
- Well-structured MKUs with deep structure
- Rich properties and relations
- Self-referential capabilities
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    from .mln import KnowledgeGraph, MonadicKnowledgeUnit
except ImportError:
    from mln import KnowledgeGraph, MonadicKnowledgeUnit


@dataclass
class KnowledgeBaseDomain:
    """Metadata for a knowledge domain"""
    name: str
    description: str
    num_concepts: int
    domain_tags: List[str]


class KnowledgeBaseLoader:
    """Load pre-built knowledge bases for testing and demos"""
    
    @staticmethod
    def load_biology_domain() -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """
        Load biology domain with taxonomic hierarchy
        
        Returns:
            (knowledge_graph, domain_metadata)
        """
        kg = KnowledgeGraph(use_gpu=False)
        
        concepts = [
            # Kingdoms
            ('organism', {
                'predicate': 'living_entity',
                'properties': {
                    'alive': True,
                    'reproduces': True,
                    'responds_to_stimuli': True,
                    'requires_energy': True
                }
            }),
            
            # Animals
            ('animal', {
                'predicate': 'organism_type',
                'properties': {
                    'kingdom': 'Animalia',
                    'multicellular': True,
                    'heterotrophic': True,
                    'mobile': True
                }
            }),
            ('mammal', {
                'predicate': 'animal_class',
                'properties': {
                    'warm_blooded': True,
                    'has_hair': True,
                    'nurses_young': True,
                    'vertebrate': True
                }
            }),
            ('primate', {
                'predicate': 'mammal_order',
                'properties': {
                    'opposable_thumbs': True,
                    'forward_facing_eyes': True,
                    'complex_brain': True,
                    'social': True
                }
            }),
            ('human', {
                'predicate': 'primate_species',
                'properties': {
                    'scientific_name': 'Homo sapiens',
                    'bipedal': True,
                    'language_capable': True,
                    'tool_user': True,
                    'self_aware': True
                }
            }),
            ('dog', {
                'predicate': 'mammal_species',
                'properties': {
                    'scientific_name': 'Canis familiaris',
                    'domesticated': True,
                    'social': True,
                    'carnivorous': True,
                    'loyal': True
                }
            }),
            ('cat', {
                'predicate': 'mammal_species',
                'properties': {
                    'scientific_name': 'Felis catus',
                    'domesticated': True,
                    'independent': True,
                    'carnivorous': True,
                    'agile': True
                }
            }),
            ('whale', {
                'predicate': 'mammal_species',
                'properties': {
                    'aquatic': True,
                    'largest_animal': True,
                    'intelligent': True,
                    'social': True,
                    'warm_blooded': True
                }
            }),
            
            # Birds
            ('bird', {
                'predicate': 'animal_class',
                'properties': {
                    'has_feathers': True,
                    'lays_eggs': True,
                    'has_wings': True,
                    'warm_blooded': True,
                    'vertebrate': True
                }
            }),
            ('eagle', {
                'predicate': 'bird_species',
                'properties': {
                    'predator': True,
                    'sharp_vision': True,
                    'powerful_talons': True,
                    'carnivorous': True
                }
            }),
            ('penguin', {
                'predicate': 'bird_species',
                'properties': {
                    'flightless': True,
                    'aquatic': True,
                    'cold_adapted': True,
                    'social': True
                }
            }),
            
            # Plants
            ('plant', {
                'predicate': 'organism_type',
                'properties': {
                    'kingdom': 'Plantae',
                    'photosynthetic': True,
                    'multicellular': True,
                    'autotrophic': True,
                    'stationary': True
                }
            }),
            ('tree', {
                'predicate': 'plant_type',
                'properties': {
                    'woody': True,
                    'tall': True,
                    'perennial': True,
                    'produces_oxygen': True
                }
            }),
            ('flower', {
                'predicate': 'plant_structure',
                'properties': {
                    'reproductive': True,
                    'colorful': True,
                    'produces_pollen': True,
                    'attracts_pollinators': True
                }
            }),
            
            # Cellular level
            ('cell', {
                'predicate': 'biological_unit',
                'properties': {
                    'basic_life_unit': True,
                    'has_membrane': True,
                    'contains_dna': True,
                    'self_replicating': True
                }
            }),
            ('neuron', {
                'predicate': 'cell_type',
                'properties': {
                    'transmits_signals': True,
                    'electrically_excitable': True,
                    'forms_networks': True,
                    'basis_of_thought': True
                }
            }),
        ]
        
        # Add all concepts with self-models
        for concept_id, deep_structure in concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()
            kg.add_concept(mku)
        
        metadata = KnowledgeBaseDomain(
            name="Biology",
            description="Taxonomic hierarchy from organisms to species, including cellular biology",
            num_concepts=len(concepts),
            domain_tags=['biology', 'taxonomy', 'life_sciences', 'evolution']
        )
        
        return kg, metadata
    
    @staticmethod
    def load_physics_domain() -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """Load physics domain with mechanics, quantum, and relativity"""
        kg = KnowledgeGraph(use_gpu=False)
        
        concepts = [
            # Classical mechanics
            ('force', {
                'predicate': 'physical_quantity',
                'properties': {
                    'vector': True,
                    'causes_acceleration': True,
                    'unit': 'Newton',
                    'fundamental': True
                }
            }),
            ('mass', {
                'predicate': 'physical_property',
                'properties': {
                    'scalar': True,
                    'measure_of_inertia': True,
                    'unit': 'kilogram',
                    'fundamental': True
                }
            }),
            ('energy', {
                'predicate': 'physical_quantity',
                'properties': {
                    'conserved': True,
                    'scalar': True,
                    'unit': 'Joule',
                    'can_transform': True
                }
            }),
            ('momentum', {
                'predicate': 'physical_quantity',
                'properties': {
                    'vector': True,
                    'conserved': True,
                    'mass_times_velocity': True,
                    'unit': 'kg⋅m/s'
                }
            }),
            ('velocity', {
                'predicate': 'kinematic_quantity',
                'properties': {
                    'vector': True,
                    'rate_of_position_change': True,
                    'unit': 'm/s',
                    'relative': True
                }
            }),
            ('acceleration', {
                'predicate': 'kinematic_quantity',
                'properties': {
                    'vector': True,
                    'rate_of_velocity_change': True,
                    'unit': 'm/s²',
                    'caused_by_force': True
                }
            }),
            
            # Quantum mechanics
            ('quantum_state', {
                'predicate': 'quantum_concept',
                'properties': {
                    'probabilistic': True,
                    'wave_function': True,
                    'superposition_capable': True,
                    'collapses_on_measurement': True
                }
            }),
            ('quantum_entanglement', {
                'predicate': 'quantum_phenomenon',
                'properties': {
                    'non_local': True,
                    'correlation': True,
                    'spooky_action': True,
                    'basis_of_quantum_computing': True
                }
            }),
            ('wave_particle_duality', {
                'predicate': 'quantum_principle',
                'properties': {
                    'fundamental': True,
                    'complementarity': True,
                    'depends_on_measurement': True,
                    'counter_intuitive': True
                }
            }),
            ('uncertainty_principle', {
                'predicate': 'quantum_law',
                'properties': {
                    'heisenberg': True,
                    'position_momentum_tradeoff': True,
                    'fundamental_limit': True,
                    'information_theoretic': True
                }
            }),
            
            # Relativity
            ('spacetime', {
                'predicate': 'relativistic_concept',
                'properties': {
                    'four_dimensional': True,
                    'curved_by_mass': True,
                    'unified_space_time': True,
                    'minkowski': False
                }
            }),
            ('special_relativity', {
                'predicate': 'physical_theory',
                'properties': {
                    'einstein': True,
                    'constant_light_speed': True,
                    'time_dilation': True,
                    'length_contraction': True
                }
            }),
            ('general_relativity', {
                'predicate': 'physical_theory',
                'properties': {
                    'einstein': True,
                    'gravity_as_curvature': True,
                    'predicts_black_holes': True,
                    'explains_cosmos': True
                }
            }),
            
            # Thermodynamics
            ('entropy', {
                'predicate': 'thermodynamic_quantity',
                'properties': {
                    'measure_of_disorder': True,
                    'always_increases': True,
                    'information_theoretic': True,
                    'unit': 'J/K'
                }
            }),
            ('temperature', {
                'predicate': 'thermodynamic_property',
                'properties': {
                    'measure_of_thermal_energy': True,
                    'intensive_property': True,
                    'unit': 'Kelvin',
                    'related_to_kinetic_energy': True
                }
            }),
        ]
        
        for concept_id, deep_structure in concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()
            kg.add_concept(mku)
        
        metadata = KnowledgeBaseDomain(
            name="Physics",
            description="Classical mechanics, quantum mechanics, relativity, and thermodynamics",
            num_concepts=len(concepts),
            domain_tags=['physics', 'quantum', 'relativity', 'mechanics']
        )
        
        return kg, metadata
    
    @staticmethod
    def load_mathematics_domain() -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """Load mathematics domain with algebra, geometry, and logic"""
        kg = KnowledgeGraph(use_gpu=False)
        
        concepts = [
            # Number systems
            ('number', {
                'predicate': 'mathematical_object',
                'properties': {
                    'abstract': True,
                    'used_for_counting': True,
                    'fundamental': True
                }
            }),
            ('natural_number', {
                'predicate': 'number_type',
                'properties': {
                    'positive': True,
                    'integer': True,
                    'countable': True,
                    'symbol': 'ℕ'
                }
            }),
            ('integer', {
                'predicate': 'number_type',
                'properties': {
                    'includes_negative': True,
                    'whole_number': True,
                    'symbol': 'ℤ',
                    'closed_under_subtraction': True
                }
            }),
            ('rational_number', {
                'predicate': 'number_type',
                'properties': {
                    'ratio_of_integers': True,
                    'can_be_decimal': True,
                    'symbol': 'ℚ',
                    'dense': True
                }
            }),
            ('real_number', {
                'predicate': 'number_type',
                'properties': {
                    'includes_irrational': True,
                    'continuous': True,
                    'symbol': 'ℝ',
                    'complete': True
                }
            }),
            ('complex_number', {
                'predicate': 'number_type',
                'properties': {
                    'has_imaginary_part': True,
                    'algebraically_closed': True,
                    'symbol': 'ℂ',
                    'two_dimensional': True
                }
            }),
            
            # Algebraic structures
            ('group', {
                'predicate': 'algebraic_structure',
                'properties': {
                    'has_operation': True,
                    'has_identity': True,
                    'has_inverses': True,
                    'associative': True
                }
            }),
            ('ring', {
                'predicate': 'algebraic_structure',
                'properties': {
                    'two_operations': True,
                    'addition_commutative': True,
                    'multiplication_associative': True,
                    'distributive': True
                }
            }),
            ('field', {
                'predicate': 'algebraic_structure',
                'properties': {
                    'division_possible': True,
                    'commutative_ring': True,
                    'examples': ['ℝ', 'ℂ', 'ℚ'],
                    'algebraically_rich': True
                }
            }),
            
            # Geometry
            ('point', {
                'predicate': 'geometric_object',
                'properties': {
                    'zero_dimensional': True,
                    'position_only': True,
                    'fundamental': True,
                    'no_extent': True
                }
            }),
            ('line', {
                'predicate': 'geometric_object',
                'properties': {
                    'one_dimensional': True,
                    'infinite': True,
                    'straight': True,
                    'defined_by_two_points': True
                }
            }),
            ('plane', {
                'predicate': 'geometric_object',
                'properties': {
                    'two_dimensional': True,
                    'flat': True,
                    'infinite': True,
                    'defined_by_three_points': True
                }
            }),
            ('circle', {
                'predicate': 'geometric_shape',
                'properties': {
                    'all_points_equidistant': True,
                    'has_radius': True,
                    'perfect_symmetry': True,
                    'pi_related': True
                }
            }),
            
            # Logic
            ('proposition', {
                'predicate': 'logical_statement',
                'properties': {
                    'has_truth_value': True,
                    'boolean': True,
                    'can_be_composed': True,
                    'declarative': True
                }
            }),
            ('proof', {
                'predicate': 'logical_derivation',
                'properties': {
                    'establishes_truth': True,
                    'step_by_step': True,
                    'rigorous': True,
                    'conclusive': True
                }
            }),
            ('theorem', {
                'predicate': 'mathematical_truth',
                'properties': {
                    'proven': True,
                    'important': True,
                    'follows_from_axioms': True,
                    'universal': True
                }
            }),
        ]
        
        for concept_id, deep_structure in concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()
            kg.add_concept(mku)
        
        metadata = KnowledgeBaseDomain(
            name="Mathematics",
            description="Number systems, algebraic structures, geometry, and mathematical logic",
            num_concepts=len(concepts),
            domain_tags=['mathematics', 'algebra', 'geometry', 'logic']
        )
        
        return kg, metadata
    
    @staticmethod
    def load_computer_science_domain() -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """Load computer science domain with algorithms, data structures, and theory"""
        kg = KnowledgeGraph(use_gpu=False)
        
        concepts = [
            # Fundamental concepts
            ('algorithm', {
                'predicate': 'computational_procedure',
                'properties': {
                    'step_by_step': True,
                    'solves_problem': True,
                    'terminates': True,
                    'deterministic': True
                }
            }),
            ('data_structure', {
                'predicate': 'data_organization',
                'properties': {
                    'stores_data': True,
                    'enables_operations': True,
                    'has_complexity': True,
                    'abstract': True
                }
            }),
            ('complexity', {
                'predicate': 'computational_measure',
                'properties': {
                    'time_or_space': True,
                    'big_o_notation': True,
                    'asymptotic': True,
                    'crucial_for_efficiency': True
                }
            }),
            
            # Data structures
            ('array', {
                'predicate': 'data_structure_type',
                'properties': {
                    'contiguous_memory': True,
                    'fixed_size': True,
                    'o1_access': True,
                    'simple': True
                }
            }),
            ('linked_list', {
                'predicate': 'data_structure_type',
                'properties': {
                    'dynamic_size': True,
                    'pointer_based': True,
                    'sequential_access': True,
                    'flexible_insertion': True
                }
            }),
            ('tree', {
                'predicate': 'data_structure_type',
                'properties': {
                    'hierarchical': True,
                    'nodes_and_edges': True,
                    'recursive_structure': True,
                    'logarithmic_operations': True
                }
            }),
            ('graph', {
                'predicate': 'data_structure_type',
                'properties': {
                    'nodes_and_edges': True,
                    'represents_relationships': True,
                    'general_structure': True,
                    'many_algorithms': True
                }
            }),
            ('hash_table', {
                'predicate': 'data_structure_type',
                'properties': {
                    'key_value_pairs': True,
                    'average_o1': True,
                    'uses_hash_function': True,
                    'efficient_lookup': True
                }
            }),
            
            # Algorithms
            ('sorting', {
                'predicate': 'algorithm_class',
                'properties': {
                    'orders_elements': True,
                    'comparison_based': True,
                    'many_variants': True,
                    'fundamental': True
                }
            }),
            ('searching', {
                'predicate': 'algorithm_class',
                'properties': {
                    'finds_elements': True,
                    'linear_or_binary': True,
                    'depends_on_structure': True,
                    'common_operation': True
                }
            }),
            ('recursion', {
                'predicate': 'algorithmic_technique',
                'properties': {
                    'self_referential': True,
                    'base_case_needed': True,
                    'elegant': True,
                    'can_be_inefficient': True
                }
            }),
            ('dynamic_programming', {
                'predicate': 'algorithmic_paradigm',
                'properties': {
                    'solves_subproblems': True,
                    'memoization': True,
                    'optimal_substructure': True,
                    'efficient': True
                }
            }),
            
            # Theory
            ('turing_machine', {
                'predicate': 'computational_model',
                'properties': {
                    'theoretical': True,
                    'universal': True,
                    'tape_based': True,
                    'basis_of_computability': True
                }
            }),
            ('np_complete', {
                'predicate': 'complexity_class',
                'properties': {
                    'hardest_np_problems': True,
                    'polynomial_verifiable': True,
                    'reduction': True,
                    'unsolved_p_vs_np': True
                }
            }),
            ('artificial_intelligence', {
                'predicate': 'cs_field',
                'properties': {
                    'machine_learning': True,
                    'reasoning': True,
                    'perception': True,
                    'goal_driven': True
                }
            }),
        ]
        
        for concept_id, deep_structure in concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()
            kg.add_concept(mku)
        
        metadata = KnowledgeBaseDomain(
            name="Computer Science",
            description="Algorithms, data structures, computational complexity, and CS theory",
            num_concepts=len(concepts),
            domain_tags=['computer_science', 'algorithms', 'data_structures', 'theory']
        )
        
        return kg, metadata
    
    @staticmethod
    def load_philosophy_domain() -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """Load philosophy domain with epistemology, metaphysics, ethics"""
        kg = KnowledgeGraph(use_gpu=False)
        
        concepts = [
            # Epistemology
            ('knowledge', {
                'predicate': 'epistemic_concept',
                'properties': {
                    'justified_true_belief': True,
                    'requires_evidence': True,
                    'fallible': True,
                    'context_dependent': False
                }
            }),
            ('truth', {
                'predicate': 'epistemic_property',
                'properties': {
                    'correspondence_to_reality': True,
                    'objective': True,
                    'fundamental': True,
                    'debated': True
                }
            }),
            ('belief', {
                'predicate': 'mental_state',
                'properties': {
                    'propositional_attitude': True,
                    'can_be_false': True,
                    'subjective': True,
                    'influences_action': True
                }
            }),
            ('skepticism', {
                'predicate': 'philosophical_position',
                'properties': {
                    'questions_knowledge': True,
                    'methodological': True,
                    'healthy_doubt': True,
                    'ancient_tradition': True
                }
            }),
            
            # Metaphysics
            ('existence', {
                'predicate': 'metaphysical_concept',
                'properties': {
                    'being': True,
                    'fundamental': True,
                    'ontological': True,
                    'hard_to_define': True
                }
            }),
            ('consciousness', {
                'predicate': 'metaphysical_phenomenon',
                'properties': {
                    'subjective_experience': True,
                    'hard_problem': True,
                    'qualia': True,
                    'self_aware': True
                }
            }),
            ('free_will', {
                'predicate': 'metaphysical_question',
                'properties': {
                    'agency': True,
                    'moral_responsibility': True,
                    'vs_determinism': True,
                    'deeply_debated': True
                }
            }),
            ('causality', {
                'predicate': 'metaphysical_relation',
                'properties': {
                    'cause_and_effect': True,
                    'temporal': True,
                    'necessary_connection': False,
                    'fundamental_to_science': True
                }
            }),
            
            # Ethics
            ('morality', {
                'predicate': 'ethical_concept',
                'properties': {
                    'right_and_wrong': True,
                    'normative': True,
                    'guides_behavior': True,
                    'culturally_variable': True
                }
            }),
            ('virtue', {
                'predicate': 'ethical_property',
                'properties': {
                    'character_trait': True,
                    'excellence': True,
                    'aristotelian': True,
                    'cultivated': True
                }
            }),
            ('utilitarianism', {
                'predicate': 'ethical_theory',
                'properties': {
                    'consequentialist': True,
                    'maximize_happiness': True,
                    'bentham_mill': True,
                    'calculative': True
                }
            }),
            ('deontology', {
                'predicate': 'ethical_theory',
                'properties': {
                    'duty_based': True,
                    'kantian': True,
                    'categorical_imperative': True,
                    'intention_matters': True
                }
            }),
            
            # Logic
            ('argument', {
                'predicate': 'logical_structure',
                'properties': {
                    'premises_and_conclusion': True,
                    'can_be_valid': True,
                    'can_be_sound': True,
                    'persuasive': True
                }
            }),
            ('fallacy', {
                'predicate': 'logical_error',
                'properties': {
                    'invalid_reasoning': True,
                    'deceptive': True,
                    'common': True,
                    'recognizable': True
                }
            }),
        ]
        
        for concept_id, deep_structure in concepts:
            mku = MonadicKnowledgeUnit(concept_id=concept_id, deep_structure=deep_structure)
            mku.create_self_model()
            kg.add_concept(mku)
        
        metadata = KnowledgeBaseDomain(
            name="Philosophy",
            description="Epistemology, metaphysics, ethics, and philosophical logic",
            num_concepts=len(concepts),
            domain_tags=['philosophy', 'epistemology', 'ethics', 'metaphysics']
        )
        
        return kg, metadata
    
    @staticmethod
    def get_available_domains() -> List[str]:
        """List all available domains"""
        return [
            'biology',
            'physics',
            'mathematics',
            'computer_science',
            'philosophy'
        ]
    
    @staticmethod
    def load_domain(domain_name: str) -> Tuple[KnowledgeGraph, KnowledgeBaseDomain]:
        """
        Load a specific domain by name
        
        Args:
            domain_name: One of 'biology', 'physics', 'mathematics', 
                        'computer_science', 'philosophy'
        
        Returns:
            (knowledge_graph, domain_metadata)
        """
        loaders = {
            'biology': KnowledgeBaseLoader.load_biology_domain,
            'physics': KnowledgeBaseLoader.load_physics_domain,
            'mathematics': KnowledgeBaseLoader.load_mathematics_domain,
            'computer_science': KnowledgeBaseLoader.load_computer_science_domain,
            'philosophy': KnowledgeBaseLoader.load_philosophy_domain,
        }
        
        if domain_name not in loaders:
            raise ValueError(f"Unknown domain: {domain_name}. Available: {list(loaders.keys())}")
        
        return loaders[domain_name]()
    
    @staticmethod
    def load_all_domains() -> Dict[str, Tuple[KnowledgeGraph, KnowledgeBaseDomain]]:
        """Load all available domains"""
        return {
            domain: KnowledgeBaseLoader.load_domain(domain)
            for domain in KnowledgeBaseLoader.get_available_domains()
        }


if __name__ == '__main__':
    # Demo
    print("=" * 70)
    print("KNOWLEDGE BASE LOADER DEMO")
    print("=" * 70)
    print()
    
    print("Available domains:")
    for domain in KnowledgeBaseLoader.get_available_domains():
        print(f"  • {domain}")
    print()
    
    # Load and display each domain
    for domain_name in KnowledgeBaseLoader.get_available_domains():
        kg, metadata = KnowledgeBaseLoader.load_domain(domain_name)
        
        print(f"\n{metadata.name} Domain:")
        print(f"  Description: {metadata.description}")
        print(f"  Concepts: {metadata.num_concepts}")
        print(f"  Tags: {', '.join(metadata.domain_tags)}")
        print(f"  Relations: {sum(len(list(mku.relations.values())[0]) if mku.relations else 0 for mku in kg.nodes.values())}")
        
        # Show sample concepts
        sample_concepts = list(kg.nodes.keys())[:5]
        print(f"  Sample concepts: {', '.join(sample_concepts)}")
    
    print("\n" + "=" * 70)
    print("✓ All domains loaded successfully!")
    print("=" * 70)
