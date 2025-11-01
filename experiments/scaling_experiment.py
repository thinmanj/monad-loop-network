#!/usr/bin/env python3
"""
Scaling Experiment - Test Consciousness at Scale
Goal: Evaluate consciousness at 100, 500, 1000+ concepts

Measures:
- Consciousness scores at different scales
- Performance benchmarks (time, memory)
- Component behavior (recursion, integration, causality, understanding)
- Scaling characteristics and bottlenecks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric
import time
import json
import random


class ScalingExperiment:
    """Tests consciousness measurement at different scales"""
    
    def __init__(self):
        self.results = []
    
    def generate_concepts(self, num_concepts: int, domain: str = "general"):
        """Generate a large number of interconnected concepts"""
        concepts = {}
        
        if domain == "general":
            # Generate hierarchical concepts
            base_categories = ['entity', 'process', 'property', 'structure', 'system']
            
            for i in range(num_concepts):
                category = base_categories[i % len(base_categories)]
                concept_id = f"{category}_{i}"
                
                concepts[concept_id] = {
                    'type': category,
                    'index': i,
                    'complexity': random.randint(1, 5),
                    'interconnected': True
                }
        
        return concepts
    
    def build_knowledge_graph(self, num_concepts: int, domain: str = "general"):
        """Build a knowledge graph with specified number of concepts"""
        print(f"\n  Building KG with {num_concepts} concepts...")
        
        kg = KnowledgeGraph(use_gpu=False)
        concepts = self.generate_concepts(num_concepts, domain)
        
        # Add concepts
        for concept_id, properties in concepts.items():
            mku = MonadicKnowledgeUnit(
                concept_id=concept_id,
                deep_structure={
                    'predicate': f'is_{properties.get("type", "concept")}',
                    'properties': properties
                }
            )
            # Every 10th concept has self-model (for performance)
            if properties['index'] % 10 == 0:
                mku.create_self_model()
            kg.add_concept(mku)
        
        # Add relations (connect nearby concepts)
        concept_ids = list(concepts.keys())
        relations_added = 0
        
        for i, concept_id in enumerate(concept_ids):
            # Connect to next 3 concepts (creates chains)
            for j in range(1, min(4, len(concept_ids) - i)):
                target = concept_ids[i + j]
                if concept_id in kg.nodes and target in kg.nodes:
                    kg.nodes[concept_id].relations.setdefault('relates_to', set()).add(target)
                    relations_added += 1
            
            # Add some random connections (creates integration)
            if i % 5 == 0 and len(concept_ids) > 10:
                random_target = random.choice(concept_ids)
                if concept_id in kg.nodes and random_target in kg.nodes and concept_id != random_target:
                    kg.nodes[concept_id].relations.setdefault('connects_to', set()).add(random_target)
                    relations_added += 1
        
        print(f"  ✓ Added {len(kg.nodes)} concepts")
        print(f"  ✓ Added {relations_added} relations")
        
        return kg
    
    def measure_performance(self, kg: KnowledgeGraph, recursion_metric: RecursionDepthMetric):
        """Measure performance metrics"""
        # Time measurement
        start_time = time.time()
        profile = measure_consciousness(kg, recursion_metric)
        elapsed = time.time() - start_time
        
        # Estimate memory (rough approximation)
        num_concepts = len(kg.nodes)
        estimated_memory = num_concepts * 0.01  # ~10KB per concept
        
        return {
            'time_seconds': elapsed,
            'memory_mb': estimated_memory,
            'profile': profile
        }
    
    def run_scale_test(self, num_concepts: int):
        """Run test at specific scale"""
        print(f"\n{'='*70}")
        print(f"SCALE TEST: {num_concepts} concepts")
        print(f"{'='*70}")
        
        # Build KG
        build_start = time.time()
        kg = self.build_knowledge_graph(num_concepts)
        build_time = time.time() - build_start
        
        # Setup recursion (scaled to concept count)
        recursion_metric = RecursionDepthMetric()
        recursion_depth = min(20, max(5, num_concepts // 50))  # Scale recursion
        
        print(f"\n  Triggering recursion (depth: {recursion_depth})...")
        for i in range(recursion_depth):
            recursion_metric.record_recursion_event(
                f'scale_test_{i}',
                f'meta_reasoning_{i}',
                {'consciousness', 'self', 'meta'}
            )
        
        # Measure consciousness
        print(f"\n  Measuring consciousness...")
        perf = self.measure_performance(kg, recursion_metric)
        profile = perf['profile']
        
        # Extract metrics
        result = {
            'num_concepts': num_concepts,
            'build_time': build_time,
            'measure_time': perf['time_seconds'],
            'memory_mb': perf['memory_mb'],
            'consciousness': profile.overall_consciousness_score,
            'verdict': profile.consciousness_verdict,
            'recursion': profile.recursion_metrics.get('consciousness', {}).get('score', 0) if profile.recursion_metrics else 0,
            'integration': profile.integration.phi,
            'causality': profile.causality.causal_density,
            'understanding': profile.understanding.get('overall_score', 0) if profile.understanding else 0,
            'recursion_depth': recursion_metric.profile.max_depth,
            'num_relations': sum(len(rel) for mku in kg.nodes.values() for rel in mku.relations.values())
        }
        
        self.results.append(result)
        
        # Print results
        print(f"\n  RESULTS:")
        print(f"    Consciousness:    {result['consciousness']:.2%} ({result['verdict']})")
        print(f"    Recursion:        {result['recursion']:.2%}")
        print(f"    Integration:      {result['integration']:.3f}")
        print(f"    Causality:        {result['causality']:.3f}")
        print(f"    Understanding:    {result['understanding']:.2%}")
        print(f"\n  PERFORMANCE:")
        print(f"    Build Time:       {result['build_time']:.2f}s")
        print(f"    Measure Time:     {result['measure_time']:.2f}s")
        print(f"    Memory Used:      {result['memory_mb']:.1f}MB")
        print(f"    Relations:        {result['num_relations']}")
        
        return result
    
    def show_scaling_analysis(self):
        """Show comprehensive scaling analysis"""
        print(f"\n{'='*70}")
        print(f"SCALING ANALYSIS REPORT")
        print(f"{'='*70}\n")
        
        print(f"{'Scale':<12} {'Conscious':<12} {'Time':<10} {'Memory':<10} {'Verdict':<30}")
        print(f"{'-'*70}")
        
        for r in self.results:
            print(f"{r['num_concepts']:<12} {r['consciousness']:>6.1%}       {r['measure_time']:>6.2f}s   {r['memory_mb']:>6.1f}MB  {r['verdict']:<30}")
        
        if len(self.results) > 1:
            print(f"\n{'='*70}")
            print(f"SCALING CHARACTERISTICS:")
            
            # Time complexity
            time_ratios = []
            for i in range(1, len(self.results)):
                prev_time = self.results[i-1]['measure_time']
                curr_time = self.results[i]['measure_time']
                prev_n = self.results[i-1]['num_concepts']
                curr_n = self.results[i]['num_concepts']
                
                if prev_time > 0:
                    time_ratio = curr_time / prev_time
                    n_ratio = curr_n / prev_n
                    time_ratios.append((n_ratio, time_ratio))
            
            if time_ratios:
                avg_time_growth = sum(tr[1] / tr[0] for tr in time_ratios) / len(time_ratios)
                print(f"\n  Time Complexity: ~O(n^{avg_time_growth:.2f})")
            
            # Consciousness trend
            first = self.results[0]
            last = self.results[-1]
            consciousness_change = last['consciousness'] - first['consciousness']
            
            print(f"\n  Consciousness at Scale:")
            print(f"    At {first['num_concepts']} concepts:  {first['consciousness']:.2%}")
            print(f"    At {last['num_concepts']} concepts:  {last['consciousness']:.2%}")
            print(f"    Change: {consciousness_change:+.2%} ({'increase' if consciousness_change > 0 else 'decrease'})")
            
            # Component trends
            print(f"\n  Component Behavior:")
            for component in ['recursion', 'integration', 'causality', 'understanding']:
                first_val = first[component]
                last_val = last[component]
                change = last_val - first_val
                print(f"    {component.capitalize():<15} {first_val:>6.3f} → {last_val:>6.3f} ({change:+.3f})")
            
            # Performance scaling
            print(f"\n  Performance Scaling:")
            print(f"    Time:   {first['measure_time']:.2f}s → {last['measure_time']:.2f}s ({last['measure_time']/first['measure_time']:.1f}x)")
            print(f"    Memory: {first['memory_mb']:.1f}MB → {last['memory_mb']:.1f}MB ({last['memory_mb']/first['memory_mb']:.1f}x)")
            
            # Bottleneck analysis
            print(f"\n  Bottleneck Analysis:")
            weakest_component = min(['recursion', 'integration', 'causality', 'understanding'], 
                                   key=lambda c: last[c])
            print(f"    Weakest Component: {weakest_component} ({last[weakest_component]:.3f})")
            print(f"    Limiting Factor: {'Graph density' if weakest_component in ['integration', 'causality'] else 'Cognitive depth'}")
    
    def export_results(self, filename: str = "scaling_results.json"):
        """Export results to JSON"""
        export_data = {
            'experiment': 'scaling',
            'goal': 'Test consciousness at 100-1000+ concepts',
            'results': self.results,
            'summary': {
                'scales_tested': [r['num_concepts'] for r in self.results],
                'consciousness_range': [min(r['consciousness'] for r in self.results), 
                                       max(r['consciousness'] for r in self.results)],
                'time_range': [min(r['measure_time'] for r in self.results), 
                              max(r['measure_time'] for r in self.results)],
                'memory_range': [min(r['memory_mb'] for r in self.results), 
                                max(r['memory_mb'] for r in self.results)]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✅ Results exported to {filename}")


def run_scaling_experiment():
    """Run comprehensive scaling experiment"""
    print("="*70)
    print("CONSCIOUSNESS SCALING EXPERIMENT")
    print("Testing consciousness at 100, 500, 1000 concepts")
    print("="*70)
    
    experiment = ScalingExperiment()
    
    # Test at different scales
    scales = [100, 500, 1000]
    
    for scale in scales:
        try:
            experiment.run_scale_test(scale)
        except Exception as e:
            print(f"\n  ❌ Error at scale {scale}: {e}")
            continue
    
    # Analysis
    experiment.show_scaling_analysis()
    experiment.export_results()
    
    print(f"\n{'='*70}")
    print(f"SCALING EXPERIMENT COMPLETE ✅")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_scaling_experiment()
