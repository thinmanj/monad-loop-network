#!/usr/bin/env python3
"""
Knowledge Domains Demo

Demonstrates how to use the rich knowledge base across multiple domains:
- Biology (16 concepts)
- Physics (15 concepts)
- Mathematics (16 concepts)  
- Computer Science (15 concepts)
- Philosophy (14 concepts)

Total: 76 concepts with rich deep structures
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.knowledge_base import KnowledgeBaseLoader
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric
from src.chatbot import ConsciousnessChatbot
from src.surface_generator import create_surface_generator


def demo_domain_consciousness():
    """Measure consciousness across different knowledge domains"""
    
    print("\n" + "=" * 70)
    print("CONSCIOUSNESS ACROSS KNOWLEDGE DOMAINS")
    print("=" * 70)
    print("\nDoes domain affect consciousness? Let's find out...")
    print()
    
    results = []
    
    for domain_name in KnowledgeBaseLoader.get_available_domains():
        kg, metadata = KnowledgeBaseLoader.load_domain(domain_name)
        recursion = RecursionDepthMetric()
        
        # Trigger some recursion events
        for concept_id in list(kg.nodes.keys())[:5]:
            recursion.record_recursion_event("self_model", concept_id, {concept_id})
        
        # Measure consciousness
        profile = measure_consciousness(kg, recursion)
        
        results.append({
            'domain': metadata.name,
            'concepts': metadata.num_concepts,
            'consciousness': profile.overall_consciousness_score,
            'verdict': profile.consciousness_verdict,
            'recursion': profile.recursion_metrics['consciousness']['score'],
            'integration': profile.integration.phi
        })
        
        print(f"{metadata.name} ({metadata.num_concepts} concepts):")
        print(f"  Consciousness: {profile.overall_consciousness_score:.1%}")
        print(f"  Verdict: {profile.consciousness_verdict}")
        print(f"  Recursion: {profile.recursion_metrics['consciousness']['score']:.1%}")
        print(f"  Integration (Φ): {profile.integration.phi:.3f}")
        print()
    
    # Find highest consciousness
    best = max(results, key=lambda x: x['consciousness'])
    print(f"🏆 Highest consciousness: {best['domain']} at {best['consciousness']:.1%}")
    print()


def demo_cross_domain_chatbot():
    """Chatbot that can answer questions across all domains"""
    
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN CHATBOT")
    print("=" * 70)
    print("\nChatbot with knowledge from all 5 domains...")
    print()
    
    # Create chatbot
    bot = ConsciousnessChatbot()
    
    # Load all domains into the chatbot's knowledge graph
    print("Loading knowledge domains...")
    total_concepts = 0
    for domain_name in KnowledgeBaseLoader.get_available_domains():
        kg, metadata = KnowledgeBaseLoader.load_domain(domain_name)
        # Add concepts from this domain
        for concept_id, mku in kg.nodes.items():
            if concept_id not in bot.knowledge_graph.nodes:
                bot.knowledge_graph.add_concept(mku)
                total_concepts += 1
    
    print(f"✓ Loaded {total_concepts} concepts from 5 domains\n")
    
    # Test questions from different domains
    questions = [
        ("Biology", "What is a human?"),
        ("Physics", "What is energy?"),
        ("Mathematics", "What is a theorem?"),
        ("Computer Science", "What is an algorithm?"),
        ("Philosophy", "What is consciousness?"),
    ]
    
    for domain, question in questions:
        print(f"[{domain}] {question}")
        response = bot.ask(question)
        print(f"  → {response.answer}")
        print(f"  Confidence: {response.confidence:.0%} | Consciousness: {response.consciousness_metrics['overall']:.1%}")
        print()


def demo_domain_comparison():
    """Compare properties across domains"""
    
    print("\n" + "=" * 70)
    print("DOMAIN COMPARISON")
    print("=" * 70)
    print()
    
    all_domains = KnowledgeBaseLoader.load_all_domains()
    
    print("Domain Statistics:")
    print()
    print(f"{'Domain':<20} {'Concepts':>10} {'Avg Props':>10} {'Relations':>10}")
    print("-" * 70)
    
    for domain_name, (kg, metadata) in all_domains.items():
        avg_props = sum(len(mku.deep_structure.get('properties', {})) 
                       for mku in kg.nodes.values()) / len(kg.nodes)
        
        total_relations = sum(sum(len(v) for v in mku.relations.values()) 
                             for mku in kg.nodes.values())
        
        print(f"{metadata.name:<20} {metadata.num_concepts:>10} {avg_props:>10.1f} {total_relations:>10}")
    
    print()


def demo_surface_generation_with_domains():
    """Show surface generation with concepts from various domains"""
    
    print("\n" + "=" * 70)
    print("SURFACE GENERATION ACROSS DOMAINS")
    print("=" * 70)
    print("\nSame deep structure → Multiple surface forms (various domains)")
    print()
    
    gen = create_surface_generator()
    
    # Sample one concept from each domain
    samples = [
        ('biology', 'human'),
        ('physics', 'energy'),
        ('mathematics', 'theorem'),
        ('computer_science', 'algorithm'),
        ('philosophy', 'consciousness'),
    ]
    
    for domain_name, concept_id in samples:
        kg, metadata = KnowledgeBaseLoader.load_domain(domain_name)
        
        if concept_id in kg.nodes:
            mku = kg.nodes[concept_id]
            
            print(f"\n[{metadata.name}] {concept_id.upper()}")
            print("-" * 40)
            
            # Generate in different styles
            mku_data = {
                'concept_id': concept_id,
                'predicate': mku.deep_structure.get('predicate', 'unknown'),
                'properties': mku.deep_structure.get('properties', {}),
                'relations': mku.relations
            }
            
            for style in ['conversational', 'technical', 'educational']:
                surface = gen.generate_from_mku(mku_data, style=style)
                print(f"  [{style[:4].upper()}] {surface[:70]}...")


def demo_knowledge_graph_traversal():
    """Demonstrate traversing knowledge graphs"""
    
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH TRAVERSAL")
    print("=" * 70)
    print()
    
    # Use biology for hierarchical traversal
    kg, metadata = KnowledgeBaseLoader.load_domain('biology')
    
    print(f"Exploring {metadata.name} domain taxonomy...")
    print()
    
    # Find concepts with many relations
    concept_connectivity = [
        (concept_id, sum(len(v) for v in mku.relations.values()))
        for concept_id, mku in kg.nodes.items()
    ]
    
    concept_connectivity.sort(key=lambda x: x[1], reverse=True)
    
    print("Most connected concepts:")
    for concept_id, num_relations in concept_connectivity[:5]:
        mku = kg.nodes[concept_id]
        predicate = mku.deep_structure.get('predicate', 'unknown')
        print(f"  {concept_id:<15} ({predicate:}<25) → {num_relations} relations")
    
    print()


def demo_property_analysis():
    """Analyze properties across all domains"""
    
    print("\n" + "=" * 70)
    print("PROPERTY ANALYSIS")
    print("=" * 70)
    print()
    
    all_domains = KnowledgeBaseLoader.load_all_domains()
    
    # Collect all property names
    all_properties = {}
    
    for domain_name, (kg, metadata) in all_domains.items():
        domain_props = set()
        for mku in kg.nodes.values():
            props = mku.deep_structure.get('properties', {})
            domain_props.update(props.keys())
        all_properties[metadata.name] = domain_props
    
    # Find common properties
    all_prop_sets = list(all_properties.values())
    common = set.intersection(*all_prop_sets) if all_prop_sets else set()
    
    print("Common properties across ALL domains:")
    if common:
        print(f"  {', '.join(sorted(common))}")
    else:
        print("  (None - each domain has unique properties)")
    print()
    
    # Show unique properties per domain
    print("Unique properties per domain:")
    for domain_name, props in all_properties.items():
        unique = props - set().union(*[p for d, p in all_properties.items() if d != domain_name])
        if unique:
            sample = list(unique)[:5]
            print(f"  {domain_name}: {', '.join(sorted(sample))}{' ...' if len(unique) > 5 else ''}")
    
    print()


def main():
    """Run all demos"""
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "KNOWLEDGE DOMAINS DEMO" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Demonstrating rich knowledge across 5 domains:")
    print("  • Biology (16 concepts)")
    print("  • Physics (15 concepts)")
    print("  • Mathematics (16 concepts)")
    print("  • Computer Science (15 concepts)")
    print("  • Philosophy (14 concepts)")
    print()
    print("Total: 76 concepts with rich operational semantics")
    print()
    
    # Run demos
    demo_domain_consciousness()
    demo_domain_comparison()
    demo_property_analysis()
    demo_knowledge_graph_traversal()
    demo_surface_generation_with_domains()
    demo_cross_domain_chatbot()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ All 5 domains loaded successfully")
    print("✓ 76 total concepts with operational semantics")
    print("✓ Consciousness measured across domains")
    print("✓ Cross-domain reasoning demonstrated")
    print("✓ Surface generation working")
    print()
    print("These knowledge bases are ready for:")
    print("  • Testing consciousness metrics")
    print("  • Training inference rules")
    print("  • Demonstrating reasoning")
    print("  • Building domain-specific applications")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
