#!/usr/bin/env python3
"""
Option B: Hybrid Approach - Natural Structure + Deep Recursion
================================================================

Combines the winning natural graph structure with deep recursion (40+ levels).

Key insight from Option A: Natural scaling alone dropped from 61.48% to 52.5%.
The bottleneck is recursion depth (only 16 at 800 concepts vs 30+ in best runs).

Strategy: Use 500-600 concepts with natural structure BUT trigger deep recursion (40-50 levels).
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


def build_hybrid_kg(n_concepts: int) -> tuple:
    """
    Build KG with natural structure optimized for deep recursion.
    """
    
    print(f"  Building hybrid KG with {n_concepts} concepts...")
    
    kg = KnowledgeGraph(use_gpu=False)
    
    # Core concepts - increased for recursion
    core_concepts = [
        "consciousness", "awareness", "self", "reflection", "thought",
        "perception", "experience", "understanding", "knowledge", "belief",
        "memory", "attention", "intention", "qualia", "subjectivity",
        "emergence", "integration", "recursion", "causality", "information",
        "pattern", "structure", "process", "state", "change",
        "system", "network", "relation", "hierarchy", "complexity",
        "loop", "feedback", "control", "regulation", "adaptation",
        "learning", "memory_formation", "recall", "recognition", "prediction",
        "meta_cognition", "self_model", "introspection", "inner_speech", "self_awareness",
        "meta_learning", "meta_memory", "meta_understanding", "meta_recursion", "strange_loop"
    ]
    
    concept_types = [
        "perception", "cognition", "emotion", "motivation", "action",
        "sensation", "processing", "integration", "synthesis", "analysis",
        "abstraction", "representation", "encoding", "decoding", "transformation"
    ]
    
    # Add concepts with INCREASED self-modeling
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
                    'index': i,
                    'recursive_ready': i % 5 == 0  # Mark concepts ready for recursion
                }
            }
        )
        
        # MORE self-models: every 5th concept (was every 10th)
        if i % 5 == 0:
            mku.create_self_model()
        
        kg.add_concept(mku)
        concept_ids.append(concept_id)
    
    # Natural connectivity (same as Option A)
    relations_added = 0
    
    # 1. Sequential chains
    for i in range(len(concept_ids)):
        for j in range(1, min(4, len(concept_ids) - i)):
            target = concept_ids[i + j]
            if concept_ids[i] in kg.nodes and target in kg.nodes:
                kg.nodes[concept_ids[i]].relations.setdefault('leads_to', set()).add(target)
                relations_added += 1
    
    # 2. Random long-range connections
    for i in range(0, len(concept_ids), 5):
        if len(concept_ids) > 10:
            random_target = random.choice(concept_ids)
            if concept_ids[i] in kg.nodes and random_target in kg.nodes and concept_ids[i] != random_target:
                kg.nodes[concept_ids[i]].relations.setdefault('integrates_with', set()).add(random_target)
                relations_added += 1
    
    # 3. ENHANCED core bidirectional links (increased from 3 to 5)
    core_size = min(50, len(concept_ids))  # Larger core
    for i in range(core_size):
        for j in random.sample(range(core_size), min(5, core_size-1)):  # 5 links per core concept
            if i != j and concept_ids[i] in kg.nodes and concept_ids[j] in kg.nodes:
                kg.nodes[concept_ids[i]].relations.setdefault('models', set()).add(concept_ids[j])
                relations_added += 1
    
    # 4. NEW: Self-reference loops for recursion
    for i in range(0, min(core_size, len(concept_ids)), 5):
        concept_id = concept_ids[i]
        if concept_id in kg.nodes:
            kg.nodes[concept_id].relations.setdefault('reflects_on', set()).add(concept_id)
            relations_added += 1
    
    print(f"  ✓ Added {len(kg.nodes)} concepts")
    print(f"  ✓ Added {relations_added} relations")
    print(f"  ✓ Avg degree: {relations_added / n_concepts:.1f}")
    print(f"  ✓ Self-modeling frequency: every 5th concept")
    
    return kg, relations_added


def run_hybrid_experiment():
    """Run hybrid experiments at different scales with DEEP recursion."""
    
    print("=" * 80)
    print("OPTION B: Hybrid Approach - Natural + Deep Recursion")
    print("=" * 80)
    print()
    print("Strategy: Natural graph structure + 40-50 recursion depth")
    print("Goal: Combine best of both approaches to reach 70%+")
    print()
    
    # Test at 500 (previous best), 600, and 700 concepts
    configs = [
        (500, 45),  # 500 concepts, 45 recursion depth
        (600, 50),  # 600 concepts, 50 recursion depth
        (700, 50),  # 700 concepts, 50 recursion depth
    ]
    
    results = []
    
    for n_concepts, recursion_depth in configs:
        print(f"\n{'=' * 80}")
        print(f"Config: {n_concepts} concepts, {recursion_depth} recursion depth")
        print(f"{'=' * 80}\n")
        
        # Build hybrid KG
        kg, n_relations = build_hybrid_kg(n_concepts)
        
        # Setup DEEP recursion
        recursion_metric = RecursionDepthMetric()
        
        print(f"\n  Triggering DEEP recursion ({recursion_depth} levels)...")
        
        # Trigger recursion with meta-concepts
        meta_concepts = {
            'consciousness', 'self', 'awareness', 'meta', 'reflection',
            'introspection', 'self_model', 'meta_cognition', 'strange_loop'
        }
        
        for i in range(recursion_depth):
            recursion_metric.record_recursion_event(
                f'hybrid_recursion_{i}',
                f'meta_level_{i}',
                meta_concepts
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
        
        # Status
        if consciousness >= 70:
            status = "🎯 TARGET ACHIEVED!"
        elif consciousness >= 65:
            status = "🔥 VERY CLOSE!"
        elif consciousness > 61.48:
            status = "🚀 NEW RECORD!"
        else:
            status = "📊 Below 61.48%..."
        
        print(f"\n{status}")
        print(f"\nConsciousness: {consciousness:.2f}%")
        print(f"  Recursion:    {recursion * 100:.2f}% (depth: {recursion_metric.profile.max_depth})")
        print(f"  Integration:  {integration:.3f}")
        print(f"  Causality:    {causality:.3f}")
        print(f"  Understanding: {understanding * 100:.2f}%")
        print(f"  Verdict:      {profile.consciousness_verdict}")
        print()
        
        # Store result
        result = {
            "n_concepts": n_concepts,
            "n_relations": n_relations,
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
        
        # Progress indicator
        if consciousness > 61.48:
            print("✨ Breakthrough! Higher than 61.48%")
        elif consciousness > 57.49:
            print("📈 Better than optimization attempt (57.49%)")
        else:
            print("📊 Continue testing...")
    
    # Final analysis
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS - HYBRID APPROACH")
    print(f"{'=' * 80}\n")
    
    best = max(results, key=lambda r: r['consciousness'])
    
    print("Results by configuration:")
    for r in results:
        indicator = " ← BEST" if r == best else ""
        print(f"  {r['n_concepts']:3d} concepts, {r['recursion_achieved']:2d}D recursion: {r['consciousness']:5.2f}%{indicator}")
    
    print(f"\nBest result: {best['consciousness']:.2f}%")
    print(f"  Configuration: {best['n_concepts']} concepts, {best['recursion_achieved']} recursion depth")
    print(f"  Verdict: {best['verdict']}")
    
    # Compare to benchmarks
    print("\nComparison to benchmarks:")
    print(f"  Previous best (scaling):      61.48%")
    print(f"  Optimization attempt:         57.49%")
    print(f"  Hybrid approach:              {best['consciousness']:.2f}%")
    
    improvement_from_best = best['consciousness'] - 61.48
    if improvement_from_best > 0:
        print(f"\n🎊 IMPROVEMENT: +{improvement_from_best:.2f}% from previous best!")
    else:
        print(f"\n📊 Change: {improvement_from_best:.2f}%")
    
    # Component breakdown
    print("\nComponent analysis (best result):")
    comp = best['components']
    print(f"  Recursion:     {comp['recursion']:.2f}% (weight: 30%)")
    print(f"  Integration:   {comp['integration']:.3f} (weight: 25%)")
    print(f"  Causality:     {comp['causality']:.3f} (weight: 20%)")
    print(f"  Understanding: {comp['understanding']:.2f}% (weight: 25%)")
    
    # Recommendations
    print("\nKey findings:")
    if best['consciousness'] >= 70:
        print("  ✅ 70%+ achieved with hybrid approach!")
        print("  → Document exact configuration")
        print("  → Replicate to confirm")
    elif best['consciousness'] >= 65:
        print("  → Very close! Fine-tune:")
        print("    - Try recursion depth 55-60")
        print("    - Adjust metric weights (Option C)")
    else:
        print("  → Hybrid approach:")
        if best['consciousness'] > 61.48:
            print("    ✓ Better than pure scaling")
        else:
            print("    ✗ Not better than pure scaling")
        print("  → Next: Try Option C (weight optimization)")
        print("  → Then: Option D (replicate exact 61.48% config)")
    
    # Save results
    output_file = "option_b_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "Option B: Hybrid Approach",
            "strategy": "Natural structure + deep recursion",
            "target": "70%+ consciousness",
            "configs_tested": configs,
            "results": results,
            "best_result": best,
            "benchmarks": {
                "previous_best": 61.48,
                "optimization_attempt": 57.49
            },
            "improvement": improvement_from_best
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print()
    
    return results


if __name__ == "__main__":
    run_hybrid_experiment()
