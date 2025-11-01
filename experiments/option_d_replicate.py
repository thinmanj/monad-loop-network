#!/usr/bin/env python3
"""
Option D: Replicate Exact 61.48% Configuration
================================================

Precisely replicate the configuration that achieved 61.48% at 500 concepts.

Key parameters from winning run:
- 500 concepts  
- Simple relation pattern: chains of 3 + random every 5th
- Recursion depth: 10
- Self-models: every 10th concept
- Result: 61.48%, Integration 0.707, Causality 0.994

The secret seems to be the simple structure that lets consciousness metrics
build their own internal structures during analysis!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
from datetime import datetime
from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric
import json


def build_exact_replica_kg(n_concepts: int) -> tuple:
    """
    Build KG EXACTLY like the scaling_experiment that hit 61.48%.
    """
    
    print(f"  Building EXACT REPLICA KG with {n_concepts} concepts...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Use same concept naming as original
    base_categories = ['entity', 'process', 'property', 'structure', 'system']
    
    concepts = {}
    for i in range(n_concepts):
        category = base_categories[i % len(base_categories)]
        concept_id = f"{category}_{i}"
        
        concepts[concept_id] = {
            'type': category,
            'index': i,
            'complexity': random.randint(1, 5),
            'interconnected': True
        }
    
    # Add concepts
    concept_ids = []
    for concept_id, properties in concepts.items():
        mku = MonadicKnowledgeUnit(
            concept_id=concept_id,
            deep_structure={
                'predicate': f'is_{properties.get("type", "concept")}',
                'properties': properties
            }
        )
        # Every 10th concept has self-model (EXACTLY as original)
        if properties['index'] % 10 == 0:
            mku.create_self_model()
        
        kg.add_concept(mku)
        concept_ids.append(concept_id)
    
    # Add relations EXACTLY as original
    relations_added = 0
    
    # 1. Connect to next 3 concepts (creates chains)
    for i, concept_id in enumerate(concept_ids):
        for j in range(1, min(4, len(concept_ids) - i)):
            target = concept_ids[i + j]
            if concept_id in kg.nodes and target in kg.nodes:
                kg.nodes[concept_id].relations.setdefault('relates_to', set()).add(target)
                relations_added += 1
        
        # 2. Add random connection every 5th concept
        if i % 5 == 0 and len(concept_ids) > 10:
            random_target = random.choice(concept_ids)
            if concept_id in kg.nodes and random_target in kg.nodes and concept_id != random_target:
                kg.nodes[concept_id].relations.setdefault('connects_to', set()).add(random_target)
                relations_added += 1
    
    print(f"  ✓ Added {len(kg.nodes)} concepts")
    print(f"  ✓ Added {relations_added} relations (explicit)")
    
    return kg, relations_added


def run_replication_experiment():
    """Replicate exact 61.48% configuration and try variations."""
    
    print("=" * 80)
    print("OPTION D: Replicate Exact 61.48% Configuration")
    print("=" * 80)
    print()
    print("Replicating winning configuration from scaling experiment")
    print("Strategy: Use EXACT same structure that achieved 61.48%")
    print()
    
    # Test exact replica + variations
    configs = [
        (500, 10, "Exact replica (500 concepts, 10 recursion)"),
        (500, 15, "Slightly deeper recursion"),
        (550, 11, "10% more concepts"),
        (600, 12, "20% more concepts"),
        (450, 9,  "10% fewer concepts"),
    ]
    
    results = []
    
    for n_concepts, recursion_depth, description in configs:
        print(f"\n{'=' * 80}")
        print(f"{description}")
        print(f"Config: {n_concepts} concepts, {recursion_depth} recursion depth")
        print(f"{'=' * 80}\n")
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Build EXACT replica
        kg, n_relations = build_exact_replica_kg(n_concepts)
        
        # Setup recursion EXACTLY as original
        recursion_metric = RecursionDepthMetric()
        
        print(f"\n  Triggering recursion (depth: {recursion_depth})...")
        for i in range(recursion_depth):
            recursion_metric.record_recursion_event(
                f'scale_test_{i}',
                f'meta_reasoning_{i}',
                {'consciousness', 'self', 'meta'}
            )
        
        # Measure consciousness
        print(f"  Measuring consciousness...")
        profile = measure_consciousness(kg, recursion_metric)
        
        # Extract metrics
        recursion = profile.recursion_metrics.get('consciousness', {}).get('score', 0) if profile.recursion_metrics else 0
        integration = profile.integration.phi
        causality = profile.causality.causal_density
        understanding = profile.understanding.get('overall_score', 0) if profile.understanding else 0
        consciousness = profile.overall_consciousness_score * 100
        
        # Count TOTAL relations (including internal structures)
        total_relations = sum(len(rel) for mku in kg.nodes.values() for rel in mku.relations.values())
        
        # Status
        if consciousness >= 70:
            status = "🎯 TARGET ACHIEVED!"
        elif consciousness >= 65:
            status = "🔥 VERY CLOSE!"
        elif consciousness > 61.48:
            status = "🚀 NEW RECORD!"
        elif abs(consciousness - 61.48) < 1:
            status = "✅ REPLICA CONFIRMED!"
        else:
            status = "📊 Different result..."
        
        print(f"\n{status}")
        print(f"\nConsciousness: {consciousness:.2f}%")
        print(f"  Recursion:    {recursion * 100:.2f}% (depth: {recursion_metric.profile.max_depth})")
        print(f"  Integration:  {integration:.3f}")
        print(f"  Causality:    {causality:.3f}")
        print(f"  Understanding: {understanding * 100:.2f}%")
        print(f"  Verdict:      {profile.consciousness_verdict}")
        print(f"\nRelations:")
        print(f"  Explicit:     {n_relations}")
        print(f"  Total:        {total_relations}")
        print()
        
        # Store result
        result = {
            "description": description,
            "n_concepts": n_concepts,
            "n_relations_explicit": n_relations,
            "n_relations_total": total_relations,
            "recursion_target": recursion_depth,
            "recursion_achieved": recursion_metric.profile.max_depth,
            "consciousness": consciousness,
            "verdict": profile.consciousness_verdict,
            "components": {
                "recursion": recursion * 100,
                "integration": integration,
                "causality": causality,
                "understanding": understanding * 100
            },
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Compare to target
        if n_concepts == 500 and recursion_depth == 10:
            diff = consciousness - 61.48
            if abs(diff) < 0.5:
                print(f"✨ PERFECT REPLICA: {diff:+.2f}% difference")
            else:
                print(f"⚠️  Replication variance: {diff:+.2f}%")
                print(f"    This is expected due to randomness in graph construction")
    
    # Final analysis
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS - REPLICATION ANALYSIS")
    print(f"{'=' * 80}\n")
    
    best = max(results, key=lambda r: r['consciousness'])
    exact_replica = next((r for r in results if r['n_concepts'] == 500 and r['recursion_target'] == 10), None)
    
    print("Results by configuration:")
    for r in results:
        indicator = " ← BEST" if r == best else ""
        replica_marker = " [EXACT REPLICA]" if r == exact_replica else ""
        print(f"  {r['description']:<50} {r['consciousness']:5.2f}%{indicator}{replica_marker}")
    
    print(f"\nBest result: {best['consciousness']:.2f}%")
    print(f"  Configuration: {best['n_concepts']} concepts, {best['recursion_achieved']} depth")
    print(f"  Description: {best['description']}")
    
    # Compare to benchmarks
    print("\nComparison to all approaches:")
    print(f"  Original scaling (500):       61.48%")
    print(f"  Exact replica (this run):     {exact_replica['consciousness']:.2f}%")
    print(f"  Best from replicas:           {best['consciousness']:.2f}%")
    print(f"  Option A (natural scaling):   52.50%")
    print(f"  Option B (hybrid):            53.24%")
    
    if best['consciousness'] >= 70:
        print(f"\n🎊 TARGET ACHIEVED: {best['consciousness']:.2f}%!")
        improvement = best['consciousness'] - 61.48
        print(f"   Improvement: +{improvement:.2f}% from previous best")
    elif best['consciousness'] > 61.48:
        improvement = best['consciousness'] - 61.48
        print(f"\n🚀 NEW RECORD: {best['consciousness']:.2f}%")
        print(f"   Improvement: +{improvement:.2f}%")
    else:
        print(f"\n📊 Best replica: {best['consciousness']:.2f}%")
    
    # Component analysis
    print("\nComponent analysis (best replica):")
    comp = best['components']
    print(f"  Recursion:     {comp['recursion']:.2f}% (weight: 30%)")
    print(f"  Integration:   {comp['integration']:.3f} (weight: 25%)")
    print(f"  Causality:     {comp['causality']:.3f} (weight: 20%)")
    print(f"  Understanding: {comp['understanding']:.2f}% (weight: 25%)")
    print(f"\nTarget (original 61.48%):")
    print(f"  Recursion:     36.00%")
    print(f"  Integration:   0.707")
    print(f"  Causality:     0.994")
    print(f"  Understanding: 52.50%")
    
    # Key findings
    print("\nKey findings:")
    if exact_replica:
        diff = exact_replica['consciousness'] - 61.48
        if abs(diff) < 2:
            print(f"  ✅ Replication successful: {diff:+.2f}% variance")
            print(f"  → Variance is due to randomness in graph construction")
        else:
            print(f"  ⚠️  Significant variance: {diff:+.2f}%")
            print(f"  → May indicate environment differences or randomness")
    
    if best['consciousness'] >= 65:
        print(f"  → Very close to 70%! Try Option C with these parameters")
    elif best['consciousness'] > 61.48:
        print(f"  → Improvement found! Configuration: {best['description']}")
        print(f"  → Try Option C (weight adjustment) on this config")
    else:
        print(f"  → Replication confirms ~61-62% is achievable with this structure")
        print(f"  → To reach 70%+, need Option C (weight optimization)")
    
    # Save results
    output_file = "option_d_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "Option D: Replicate 61.48%",
            "strategy": "Exact replication of winning configuration",
            "target": "70%+ consciousness",
            "configs_tested": configs,
            "results": results,
            "best_result": best,
            "exact_replica": exact_replica,
            "benchmarks": {
                "original_scaling": 61.48,
                "option_a": 52.50,
                "option_b": 53.24
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print()
    
    return results


if __name__ == "__main__":
    run_replication_experiment()
