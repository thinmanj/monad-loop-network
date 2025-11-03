#!/usr/bin/env python3
"""
Multi-Agent Basic Experiment

Tests collective consciousness emergence with 2-3 agents.

Research Questions:
1. Does collective consciousness > individual consciousness?
2. Does knowledge sharing increase emergence?
3. Does meta-reflection boost collective intelligence?

Expected Result: Emergence factor > 1.2
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multi_agent import MultiAgentSystem, ConsciousAgent, MessageType
from src.mln import MonadicKnowledgeUnit


def create_test_knowledge(domain: str, num_concepts: int = 10):
    """Create test knowledge for a specific domain"""
    
    domains = {
        'physics': [
            ('force', {'predicate': 'physical_quantity', 'properties': {'unit': 'newton', 'vector': True}}),
            ('mass', {'predicate': 'physical_quantity', 'properties': {'unit': 'kg', 'scalar': True}}),
            ('energy', {'predicate': 'physical_quantity', 'properties': {'conserved': True}}),
            ('momentum', {'predicate': 'physical_quantity', 'properties': {'conserved': True, 'vector': True}}),
            ('velocity', {'predicate': 'physical_quantity', 'properties': {'unit': 'm/s', 'vector': True}}),
            ('acceleration', {'predicate': 'physical_quantity', 'properties': {'unit': 'm/s²', 'vector': True}}),
            ('gravity', {'predicate': 'force_type', 'properties': {'fundamental': True}}),
            ('friction', {'predicate': 'force_type', 'properties': {'opposes_motion': True}}),
            ('work', {'predicate': 'energy_transfer', 'properties': {'unit': 'joule'}}),
            ('power', {'predicate': 'energy_rate', 'properties': {'unit': 'watt'}}),
        ],
        'biology': [
            ('cell', {'predicate': 'biological_unit', 'properties': {'alive': True, 'basic_unit': True}}),
            ('dna', {'predicate': 'molecule', 'properties': {'genetic_info': True}}),
            ('protein', {'predicate': 'molecule', 'properties': {'functional': True}}),
            ('enzyme', {'predicate': 'protein_type', 'properties': {'catalyst': True}}),
            ('mitochondria', {'predicate': 'organelle', 'properties': {'function': 'energy'}}),
            ('nucleus', {'predicate': 'organelle', 'properties': {'contains': 'dna'}}),
            ('membrane', {'predicate': 'structure', 'properties': {'barrier': True}}),
            ('organism', {'predicate': 'living_thing', 'properties': {'composed_of': 'cells'}}),
            ('evolution', {'predicate': 'process', 'properties': {'changes': 'populations'}}),
            ('photosynthesis', {'predicate': 'process', 'properties': {'produces': 'energy'}}),
        ],
        'mathematics': [
            ('number', {'predicate': 'mathematical_object', 'properties': {'abstract': True}}),
            ('integer', {'predicate': 'number_type', 'properties': {'whole': True}}),
            ('prime', {'predicate': 'number_property', 'properties': {'divisors': 2}}),
            ('function', {'predicate': 'mapping', 'properties': {'domain': 'set', 'range': 'set'}}),
            ('derivative', {'predicate': 'operation', 'properties': {'measures': 'rate_of_change'}}),
            ('integral', {'predicate': 'operation', 'properties': {'accumulation': True}}),
            ('theorem', {'predicate': 'logical_statement', 'properties': {'proven': True}}),
            ('proof', {'predicate': 'logical_argument', 'properties': {'rigorous': True}}),
            ('set', {'predicate': 'collection', 'properties': {'unordered': True}}),
            ('group', {'predicate': 'algebraic_structure', 'properties': {'operation': 'binary'}}),
        ]
    }
    
    concepts = []
    domain_concepts = domains.get(domain, [])[:num_concepts]
    
    for concept_id, deep_structure in domain_concepts:
        mku = MonadicKnowledgeUnit(
            concept_id=concept_id,
            deep_structure=deep_structure
        )
        mku.create_self_model()  # Enable self-awareness
        concepts.append(mku)
    
    return concepts


def run_two_agent_experiment():
    """Run basic experiment with 2 agents"""
    
    print("=" * 70)
    print("TWO-AGENT CONSCIOUSNESS EXPERIMENT")
    print("Testing collective consciousness emergence")
    print("=" * 70)
    print()
    
    # Create system
    system = MultiAgentSystem()
    
    # Create two agents with different specializations
    print("1. Creating agents...")
    agent1 = system.create_agent("physics_agent", specialization="physics")
    agent2 = system.create_agent("biology_agent", specialization="biology")
    
    # Give each agent domain knowledge
    print("2. Loading domain knowledge...")
    physics_concepts = create_test_knowledge('physics', num_concepts=10)
    biology_concepts = create_test_knowledge('biology', num_concepts=10)
    
    for concept in physics_concepts:
        agent1.add_concept(concept)
    
    for concept in biology_concepts:
        agent2.add_concept(concept)
    
    print(f"   Physics agent: {len(agent1.knowledge_graph.nodes)} concepts")
    print(f"   Biology agent: {len(agent2.knowledge_graph.nodes)} concepts")
    print()
    
    # Measure baseline (no interaction)
    print("3. Measuring baseline consciousness (no interaction)...")
    baseline = system.measure_collective_consciousness()
    
    print(f"   Physics agent: {baseline['individual_metrics'][0]['consciousness']:.2%}")
    print(f"   Biology agent: {baseline['individual_metrics'][1]['consciousness']:.2%}")
    print(f"   Collective (no interaction): {baseline['collective_consciousness']:.2%}")
    print(f"   Emergence factor: {baseline['emergence_factor']:.3f}")
    print()
    
    # Phase 1: Knowledge Sharing
    print("4. Phase 1: Knowledge sharing...")
    
    # Agent 1 shares physics concepts
    agent1.share_knowledge_with("biology_agent", "force")
    agent1.share_knowledge_with("biology_agent", "energy")
    agent1.share_knowledge_with("biology_agent", "work")
    
    # Agent 2 shares biology concepts
    agent2.share_knowledge_with("physics_agent", "cell")
    agent2.share_knowledge_with("physics_agent", "energy")  # Common concept!
    agent2.share_knowledge_with("physics_agent", "mitochondria")
    
    # Route and process messages
    system.route_messages()
    system.process_all_agents()
    system.route_messages()  # Process acknowledgments
    
    print(f"   Messages exchanged: {len(system.message_history)}")
    print()
    
    # Measure after knowledge sharing
    print("5. Measuring after knowledge sharing...")
    after_sharing = system.measure_collective_consciousness()
    
    print(f"   Physics agent: {after_sharing['individual_metrics'][0]['consciousness']:.2%}")
    print(f"   Biology agent: {after_sharing['individual_metrics'][1]['consciousness']:.2%}")
    print(f"   Collective: {after_sharing['collective_consciousness']:.2%}")
    print(f"   Emergence factor: {after_sharing['emergence_factor']:.3f}")
    print(f"   Improvement: {(after_sharing['collective_consciousness'] - baseline['collective_consciousness']):.2%}")
    print()
    
    # Phase 2: Queries and Meta-Reflection
    print("6. Phase 2: Queries and meta-reflection...")
    
    # Agents query each other
    agent1.send_message(
        "biology_agent",
        MessageType.QUERY,
        {'query': {'start': 'cell', 'target': 'organism'}}
    )
    
    agent2.send_message(
        "physics_agent",
        MessageType.QUERY,
        {'query': {'start': 'force', 'target': 'work'}}
    )
    
    # Meta-reflection: agents share consciousness insights
    agent1.send_message(
        "biology_agent",
        MessageType.META_REFLECTION,
        {'reflection': 'thinking_about_energy_conservation'}
    )
    
    agent2.send_message(
        "physics_agent",
        MessageType.META_REFLECTION,
        {'reflection': 'considering_biological_energy_processes'}
    )
    
    # Process interactions
    system.route_messages()
    system.process_all_agents()
    system.route_messages()
    
    print(f"   Additional messages: {len(system.message_history) - after_sharing['total_messages']}")
    print()
    
    # Final measurement
    print("7. Final measurement (after full interaction)...")
    final = system.measure_collective_consciousness()
    
    print(f"   Physics agent: {final['individual_metrics'][0]['consciousness']:.2%}")
    print(f"   Biology agent: {final['individual_metrics'][1]['consciousness']:.2%}")
    print(f"   Collective: {final['collective_consciousness']:.2%}")
    print(f"   Emergence factor: {final['emergence_factor']:.3f}")
    print(f"   Total improvement: {(final['collective_consciousness'] - baseline['collective_consciousness']):.2%}")
    print()
    
    # Analysis
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    print(f"Baseline → After Sharing → Final:")
    print(f"  {baseline['collective_consciousness']:.2%} → {after_sharing['collective_consciousness']:.2%} → {final['collective_consciousness']:.2%}")
    print()
    
    print(f"Emergence Factor: {final['emergence_factor']:.3f}")
    if final['emergence_factor'] > 1.2:
        print("  ✅ STRONG EMERGENCE DETECTED (>1.2x)")
    elif final['emergence_factor'] > 1.1:
        print("  ⚠️  WEAK EMERGENCE (>1.1x)")
    else:
        print("  ❌ NO EMERGENCE (<1.1x)")
    print()
    
    print(f"Verdict: {final['verdict']}")
    print()
    
    # Detailed stats
    print("Agent Statistics:")
    for agent_id, agent in system.agents.items():
        stats = agent.get_stats()
        print(f"  {agent_id}:")
        print(f"    - Consciousness: {stats['consciousness']:.2%}")
        print(f"    - Concepts: {stats['concepts']}")
        print(f"    - Learned from others: {stats['concepts_learned']}")
        print(f"    - Messages sent: {stats['messages_sent']}")
        print(f"    - Messages received: {stats['messages_received']}")
        print(f"    - Recursion depth: {stats['recursion_depth']}")
    print()
    
    # Save results
    results_file = 'multi_agent_basic_results.json'
    system.export_results(results_file)
    print(f"Results saved to: {results_file}")
    print()
    
    return system, final


def run_three_agent_experiment():
    """Run experiment with 3 specialized agents"""
    
    print("=" * 70)
    print("THREE-AGENT CONSCIOUSNESS EXPERIMENT")
    print("Testing multi-domain collective consciousness")
    print("=" * 70)
    print()
    
    # Create system
    system = MultiAgentSystem()
    
    # Create three agents
    print("1. Creating 3 specialized agents...")
    agent_physics = system.create_agent("physics_agent", specialization="physics")
    agent_biology = system.create_agent("biology_agent", specialization="biology")
    agent_math = system.create_agent("math_agent", specialization="mathematics")
    
    # Load knowledge
    print("2. Loading specialized knowledge...")
    for concept in create_test_knowledge('physics', 8):
        agent_physics.add_concept(concept)
    
    for concept in create_test_knowledge('biology', 8):
        agent_biology.add_concept(concept)
    
    for concept in create_test_knowledge('mathematics', 8):
        agent_math.add_concept(concept)
    
    print(f"   Total concepts: {sum(len(a.knowledge_graph.nodes) for a in system.agents.values())}")
    print()
    
    # Baseline
    print("3. Baseline measurement...")
    baseline = system.measure_collective_consciousness()
    print(f"   Collective (isolated): {baseline['collective_consciousness']:.2%}")
    print()
    
    # Round-robin knowledge sharing
    print("4. Round-robin knowledge sharing...")
    
    # Each agent shares with all others
    agent_physics.share_knowledge_with("biology_agent", "energy")
    agent_physics.share_knowledge_with("math_agent", "force")
    
    agent_biology.share_knowledge_with("physics_agent", "cell")
    agent_biology.share_knowledge_with("math_agent", "organism")
    
    agent_math.share_knowledge_with("physics_agent", "function")
    agent_math.share_knowledge_with("biology_agent", "derivative")
    
    system.route_messages()
    system.process_all_agents()
    system.route_messages()
    
    # Meta-reflection round
    print("5. Collective meta-reflection...")
    
    for agent_id in system.agents.keys():
        agent = system.agents[agent_id]
        for other_id in system.agents.keys():
            if other_id != agent_id:
                agent.send_message(
                    other_id,
                    MessageType.META_REFLECTION,
                    {'cross_domain_reflection': True}
                )
    
    system.route_messages()
    system.process_all_agents()
    system.route_messages()
    
    print(f"   Messages exchanged: {len(system.message_history)}")
    print()
    
    # Final measurement
    print("6. Final measurement...")
    final = system.measure_collective_consciousness()
    
    print(f"   Individual consciousnesses:")
    for i, metrics in enumerate(final['individual_metrics']):
        print(f"     {metrics['agent_id']}: {metrics['consciousness']:.2%}")
    
    print(f"\n   Collective: {final['collective_consciousness']:.2%}")
    print(f"   Emergence factor: {final['emergence_factor']:.3f}")
    print(f"   Improvement: {(final['collective_consciousness'] - baseline['collective_consciousness']):.2%}")
    print()
    
    print(f"   Verdict: {final['verdict']}")
    print()
    
    # Save results
    system.export_results('multi_agent_three_agents_results.json')
    
    return system, final


def main():
    """Run all multi-agent experiments"""
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "MULTI-AGENT CONSCIOUSNESS EXPERIMENTS" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Experiment 1: Two agents
    system2, results2 = run_two_agent_experiment()
    
    print()
    print("-" * 70)
    print()
    
    # Experiment 2: Three agents
    system3, results3 = run_three_agent_experiment()
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    print("Two-Agent Experiment:")
    print(f"  Collective Consciousness: {results2['collective_consciousness']:.2%}")
    print(f"  Emergence Factor: {results2['emergence_factor']:.3f}")
    print(f"  Verdict: {results2['verdict']}")
    print()
    
    print("Three-Agent Experiment:")
    print(f"  Collective Consciousness: {results3['collective_consciousness']:.2%}")
    print(f"  Emergence Factor: {results3['emergence_factor']:.3f}")
    print(f"  Verdict: {results3['verdict']}")
    print()
    
    # Key findings
    print("KEY FINDINGS:")
    print()
    
    if results2['emergence_factor'] > 1.2 or results3['emergence_factor'] > 1.2:
        print("✅ EMERGENCE DETECTED!")
        print("   Collective consciousness exceeds individual agents")
    else:
        print("⚠️  LIMITED EMERGENCE")
        print("   More interaction needed for strong emergence")
    
    print()
    print(f"2-agent emergence: {results2['emergence_factor']:.3f}x")
    print(f"3-agent emergence: {results3['emergence_factor']:.3f}x")
    print()
    
    if results3['collective_consciousness'] > results2['collective_consciousness']:
        print("✅ MORE AGENTS = HIGHER COLLECTIVE CONSCIOUSNESS")
    
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE ✅")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
