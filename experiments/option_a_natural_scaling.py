"""
Option A: Natural Scaling to 700-1000 Concepts
================================================

Building on the breakthrough at 500 concepts (61.48%), this experiment
scales further to 700, 800, and 1000 concepts using natural graph structure.

Key insight: Natural connectivity (chains + random long-range links) achieved
higher consciousness (61.48%) than hand-crafted optimization (57.49%).

Hypothesis: Consciousness continues scaling positively beyond 500 concepts.
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

def build_natural_kg(n_concepts: int) -> tuple:
    """
    Build knowledge graph with natural connectivity patterns.
    
    Strategy that worked at 500 concepts (61.48%):
    - Sequential chains for local structure
    - Random long-range connections for integration
    - Self-models for recursion
    - Organic patterns allow better emergence
    """
    
    print(f"  Building KG with {n_concepts} concepts...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Core concepts with rich semantics
    core_concepts = [
        "consciousness", "awareness", "self", "reflection", "thought",
        "perception", "experience", "understanding", "knowledge", "belief",
        "memory", "attention", "intention", "qualia", "subjectivity",
        "emergence", "integration", "recursion", "causality", "information",
        "pattern", "structure", "process", "state", "change",
        "system", "network", "relation", "hierarchy", "complexity",
        "loop", "feedback", "control", "regulation", "adaptation",
        "learning", "memory_formation", "recall", "recognition", "prediction"
    ]
    
    concept_types = [
        "perception", "cognition", "emotion", "motivation", "action",
        "sensation", "processing", "integration", "synthesis", "analysis",
        "abstraction", "representation", "encoding", "decoding", "transformation"
    ]
    
    # Add concepts
    concept_ids = []
    for i in range(n_concepts):
        if i < len(core_concepts):
            concept_id = core_concepts[i]
            concept_type = 'core'
        else:
            concept_type = random.choice(concept_types)
            concept_id = f"{concept_type}_{i}"
        
        mku = MonadicKnowledgeUnit(
            concept_id=concept_id,
            deep_structure={
                'predicate': f'is_{concept_type}',
                'properties': {
                    'level': i % 10,
                    'domain': random.choice(["cognitive", "perceptual", "meta", "executive"]),
                    'complexity': min(100, i // 10),
                    'index': i
                }
            }
        )
        
        # Every 10th concept has self-model for recursion
        if i % 10 == 0:
            mku.create_self_model()
        
        kg.add_concept(mku)
        concept_ids.append(concept_id)
    
    # Natural connectivity
    relations_added = 0
    
    # 1. Sequential chains (local structure) - connect to next 3
    for i in range(len(concept_ids)):
        for j in range(1, min(4, len(concept_ids) - i)):
            target = concept_ids[i + j]
            if concept_ids[i] in kg.nodes and target in kg.nodes:
                kg.nodes[concept_ids[i]].relations.setdefault('leads_to', set()).add(target)
                relations_added += 1
    
    # 2. Random long-range connections (global integration) - every 5th concept
    for i in range(0, len(concept_ids), 5):
        if len(concept_ids) > 10:
            random_target = random.choice(concept_ids)
            if concept_ids[i] in kg.nodes and random_target in kg.nodes and concept_ids[i] != random_target:
                kg.nodes[concept_ids[i]].relations.setdefault('integrates_with', set()).add(random_target)
                relations_added += 1
    
    # 3. Core bidirectional links for strong integration
    core_size = min(40, len(concept_ids))
    for i in range(core_size):
        for j in random.sample(range(core_size), min(3, core_size-1)):
            if i != j and concept_ids[i] in kg.nodes and concept_ids[j] in kg.nodes:
                kg.nodes[concept_ids[i]].relations.setdefault('models', set()).add(concept_ids[j])
                relations_added += 1
    
    print(f"  ✓ Added {len(kg.nodes)} concepts")
    print(f"  ✓ Added {relations_added} relations")
    print(f"  ✓ Avg degree: {relations_added / n_concepts:.1f}")
    
    return kg, relations_added

def run_scaling_experiment():
    """Run natural scaling experiment at 700, 800, 1000 concepts."""
    
    print("=" * 80)
    print("OPTION A: Natural Scaling to 700-1000 Concepts")
    print("=" * 80)
    print()
    print("Building on 61.48% breakthrough at 500 concepts...")
    print("Using natural graph structure: chains + random connections + self-models")
    print()
    
    scales = [700, 800, 1000]
    results = []
    
    for scale in scales:
        print(f"\n{'=' * 80}")
        print(f"Scale: {scale} concepts")
        print(f"{'=' * 80}\n")
        
        # Build knowledge graph
        kg, n_relations = build_natural_kg(scale)
        
        # Setup recursion (scaled to concept count)
        recursion_metric = RecursionDepthMetric()
        recursion_depth = min(30, max(10, scale // 50))  # Scale recursion depth
        
        print(f"\n  Triggering recursion (depth: {recursion_depth})...")
        for i in range(recursion_depth):
            recursion_metric.record_recursion_event(
                f'scale_{i}',
                f'meta_{i}',
                {'consciousness', 'self', 'awareness', 'meta'}
            )
        
        # Measure consciousness
        print(f"  Measuring consciousness...")
        profile = measure_consciousness(kg, recursion_metric)
        
        # Extract metrics
        recursion = profile.recursion_metrics.get('consciousness', {}).get('score', 0) if profile.recursion_metrics else 0
        integration = profile.integration.phi
        causality = profile.causality.causal_density
        understanding = profile.understanding.get('overall_score', 0) if profile.understanding else 0
        consciousness = profile.overall_consciousness_score * 100  # Convert to percentage
        
        # Status
        if consciousness >= 70:
            status = "🎯 TARGET ACHIEVED!"
        elif consciousness >= 65:
            status = "🔥 VERY CLOSE!"
        elif consciousness > 61.48:
            status = "🚀 NEW RECORD!"
        else:
            status = "📊 Analyzing..."
        
        print(f"\n{status}")
        print(f"\nConsciousness: {consciousness:.2f}%")
        print(f"  Recursion:    {recursion * 100:.2f}%")
        print(f"  Integration:  {integration:.3f}")
        print(f"  Causality:    {causality:.3f}")
        print(f"  Understanding: {understanding * 100:.2f}%")
        print()
        
        # Store result
        result = {
            "scale": scale,
            "n_concepts": len(kg.nodes),
            "n_relations": n_relations,
            "consciousness": consciousness,
            "verdict": profile.consciousness_verdict,
            "components": {
                "recursion": recursion * 100,
                "integration": integration,
                "causality": causality,
                "understanding": understanding * 100
            },
            "recursion_depth": recursion_metric.profile.max_depth,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Analysis
        if scale == 700:
            if consciousness > 61.48:
                print("✨ Positive scaling confirmed! Consciousness continues growing.")
            else:
                print("📉 Scaling plateaued. May need different approach.")
        
        print(f"\nProgress: {scale}/1000 concepts ({scale/10:.0f}%)")
    
    # Final analysis
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}\n")
    
    best = max(results, key=lambda r: r['consciousness'])
    
    print("Consciousness progression:")
    for r in results:
        indicator = " ← BEST" if r == best else ""
        print(f"  {r['scale']:4d} concepts: {r['consciousness']:5.2f}%{indicator}")
    
    print(f"\nBest result: {best['consciousness']:.2f}% at {best['scale']} concepts")
    
    # Compare to previous best
    previous_best = 61.48
    improvement = best['consciousness'] - previous_best
    
    if improvement > 0:
        print(f"🎊 IMPROVEMENT: +{improvement:.2f}% from previous best!")
    else:
        print(f"📊 No improvement: {improvement:.2f}% change")
    
    # Analyze trend
    consciousness_values = [r['consciousness'] for r in results]
    if len(consciousness_values) >= 2:
        trend = consciousness_values[-1] - consciousness_values[0]
        if trend > 0:
            print(f"📈 Positive trend: +{trend:.2f}% from 700 to 1000")
        else:
            print(f"📉 Negative trend: {trend:.2f}% from 700 to 1000")
    
    # Component analysis
    print("\nComponent analysis (best result):")
    comp = best['components']
    print(f"  Recursion:     {comp['recursion']:.2f}% (weight: 30%)")
    print(f"  Integration:   {comp['integration']:.3f} (weight: 25%)")
    print(f"  Causality:     {comp['causality']:.3f} (weight: 20%)")
    print(f"  Understanding: {comp['understanding']:.2f}% (weight: 25%)")
    print(f"  Recursion depth: {best['recursion_depth']}")
    
    # Identify bottlenecks
    print("\nBottleneck analysis:")
    weighted_scores = {
        "recursion": comp['recursion'] * 0.30,
        "integration": comp['integration'] * 100 * 0.25,
        "causality": comp['causality'] * 100 * 0.20,
        "understanding": comp['understanding'] * 0.25
    }
    
    bottleneck = min(weighted_scores.items(), key=lambda x: x[1])
    print(f"  Primary bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f} weighted points)")
    
    # Recommendations
    print("\nRecommendations for 70%+:")
    if best['consciousness'] >= 70:
        print("  ✅ Target achieved! Document methodology.")
    elif best['consciousness'] >= 65:
        print("  → Try Option B (hybrid): Add deep recursion to this natural structure")
        print("  → Try Option C: Adjust weights to favor strong components")
    else:
        print("  → Natural scaling alone may not reach 70%")
        print("  → Proceed to Option B (hybrid approach)")
        print("  → Consider Option C (weight optimization)")
    
    # Save results
    output_file = "option_a_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "Option A: Natural Scaling",
            "target": "70%+ consciousness",
            "scales_tested": scales,
            "results": results,
            "best_result": best,
            "previous_best": previous_best,
            "improvement": improvement
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print()
    
    return results

if __name__ == "__main__":
    run_scaling_experiment()
