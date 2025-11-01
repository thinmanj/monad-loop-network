#!/usr/bin/env python3
"""
Performance Benchmark Suite - Issue #8
Measures system performance across different scales and operations
"""

import sys
import os
import time
import json
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import (
    MonadicKnowledgeUnit,
    KnowledgeGraph,
    HybridIntelligenceSystem,
)


class PerformanceBenchmark:
    """Benchmark suite for MLN performance"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()
    
    def benchmark_add_concepts(self, sizes: List[int], use_gpu: bool = False):
        """Benchmark adding concepts to knowledge graph"""
        print(f"\n{'='*70}")
        print(f"Benchmark: Adding Concepts (GPU={use_gpu})")
        print(f"{'='*70}")
        
        results = {}
        
        for size in sizes:
            kg = KnowledgeGraph(use_gpu=use_gpu)
            
            start = time.time()
            for i in range(size):
                mku = MonadicKnowledgeUnit(
                    f'concept_{i}',
                    {
                        'predicate': 'test_concept',
                        'properties': {'index': i, 'value': i * 2}
                    }
                )
                kg.add_concept(mku)
            elapsed = time.time() - start
            
            concepts_per_sec = size / elapsed if elapsed > 0 else 0
            
            results[size] = {
                'total_time': elapsed,
                'concepts_per_sec': concepts_per_sec,
                'avg_time_per_concept': elapsed / size if size > 0 else 0
            }
            
            print(f"  Size {size:>5}: {elapsed:.3f}s total, "
                  f"{concepts_per_sec:.0f} concepts/sec, "
                  f"{elapsed/size*1000:.2f}ms per concept")
        
        key = f"add_concepts_gpu_{use_gpu}"
        self.results[key] = results
        return results
    
    def benchmark_queries(self, graph_size: int, num_queries: int, use_gpu: bool = False):
        """Benchmark query performance"""
        print(f"\n{'='*70}")
        print(f"Benchmark: Queries (Graph size={graph_size}, GPU={use_gpu})")
        print(f"{'='*70}")
        
        # Build graph
        kg = KnowledgeGraph(use_gpu=use_gpu)
        print(f"  Building graph with {graph_size} concepts...")
        
        for i in range(graph_size):
            mku = MonadicKnowledgeUnit(
                f'node_{i}',
                {'predicate': 'test', 'properties': {'index': i}}
            )
            # Add some relations
            if i > 0:
                mku.relations['connects_to'] = {f'node_{i-1}'}
            if i < graph_size - 1:
                mku.relations['connects_to'] = mku.relations.get('connects_to', set()) | {f'node_{i+1}'}
            kg.add_concept(mku)
        
        print(f"  Graph built. Running {num_queries} queries...")
        
        # Run queries
        start = time.time()
        for i in range(num_queries):
            start_node = f'node_{i % graph_size}'
            target_node = f'node_{(i + 5) % graph_size}'
            chain = kg.query(start_node, target_node)
        elapsed = time.time() - start
        
        queries_per_sec = num_queries / elapsed if elapsed > 0 else 0
        
        result = {
            'graph_size': graph_size,
            'num_queries': num_queries,
            'total_time': elapsed,
            'queries_per_sec': queries_per_sec,
            'avg_time_per_query': elapsed / num_queries if num_queries > 0 else 0
        }
        
        print(f"  Completed: {elapsed:.3f}s total, "
              f"{queries_per_sec:.0f} queries/sec, "
              f"{elapsed/num_queries*1000:.2f}ms per query")
        
        key = f"queries_size_{graph_size}_gpu_{use_gpu}"
        self.results[key] = result
        return result
    
    def benchmark_inference(self, graph_size: int, use_gpu: bool = False):
        """Benchmark inference rule application"""
        print(f"\n{'='*70}")
        print(f"Benchmark: Inference (Graph size={graph_size}, GPU={use_gpu})")
        print(f"{'='*70}")
        
        # Create system with all rules
        system = HybridIntelligenceSystem(use_gpu=use_gpu)
        
        print(f"  Adding {graph_size} concepts with relations...")
        
        # Build chain: A→B→C→...
        for i in range(graph_size):
            concept_id = f'concept_{i}'
            system.add_knowledge(concept_id, {
                'predicate': 'chain_element',
                'properties': {'index': i}
            })
            
            # Add implication to next
            if i < graph_size - 1:
                system.kg.nodes[concept_id].relations['implies'] = {f'concept_{i+1}'}
        
        print(f"  Running inference on all concepts...")
        
        # Apply inference to each concept
        start = time.time()
        total_inferences = 0
        for concept_id in system.kg.nodes:
            if concept_id == 'self':  # Skip meta concept
                continue
            concept = system.kg.nodes[concept_id]
            inferences = system.kg.apply_inference(concept)
            total_inferences += len(inferences)
        elapsed = time.time() - start
        
        inferences_per_sec = total_inferences / elapsed if elapsed > 0 else 0
        
        result = {
            'graph_size': graph_size,
            'total_inferences': total_inferences,
            'total_time': elapsed,
            'inferences_per_sec': inferences_per_sec,
            'avg_time_per_concept': elapsed / graph_size if graph_size > 0 else 0
        }
        
        print(f"  Completed: {total_inferences} inferences in {elapsed:.3f}s, "
              f"{inferences_per_sec:.0f} inferences/sec")
        
        key = f"inference_size_{graph_size}_gpu_{use_gpu}"
        self.results[key] = result
        return result
    
    def benchmark_structural_similarity(self, num_comparisons: int, use_gpu: bool = False):
        """Benchmark structural similarity computation"""
        print(f"\n{'='*70}")
        print(f"Benchmark: Structural Similarity ({num_comparisons} comparisons, GPU={use_gpu})")
        print(f"{'='*70}")
        
        # Create concepts
        concepts = []
        for i in range(int(num_comparisons ** 0.5) + 1):
            mku = MonadicKnowledgeUnit(
                f'similar_{i}',
                {
                    'predicate': 'test',
                    'properties': {
                        'feature_a': i % 10,
                        'feature_b': i % 5,
                        'feature_c': i % 7
                    }
                }
            )
            concepts.append(mku)
        
        print(f"  Computing {num_comparisons} similarity comparisons...")
        
        # Compute pairwise similarities
        start = time.time()
        count = 0
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if count >= num_comparisons:
                    break
                similarity = c1._structural_similarity(c2)
                count += 1
            if count >= num_comparisons:
                break
        elapsed = time.time() - start
        
        comparisons_per_sec = num_comparisons / elapsed if elapsed > 0 else 0
        
        result = {
            'num_comparisons': num_comparisons,
            'total_time': elapsed,
            'comparisons_per_sec': comparisons_per_sec
        }
        
        print(f"  Completed: {elapsed:.3f}s total, "
              f"{comparisons_per_sec:.0f} comparisons/sec")
        
        key = f"similarity_{num_comparisons}_gpu_{use_gpu}"
        self.results[key] = result
        return result
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON"""
        if filename is None:
            filename = f"benchmark_results_{self.timestamp.replace(':', '-')}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        output = {
            'timestamp': self.timestamp,
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {filepath}")
        print(f"{'='*70}")
        
        return filepath
    
    def print_summary(self):
        """Print benchmark summary"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total benchmarks: {len(self.results)}")
        print()
        
        # Group by operation type
        add_results = {k: v for k, v in self.results.items() if 'add_concepts' in k}
        query_results = {k: v for k, v in self.results.items() if 'queries' in k}
        inference_results = {k: v for k, v in self.results.items() if 'inference' in k}
        similarity_results = {k: v for k, v in self.results.items() if 'similarity' in k}
        
        if add_results:
            print("Adding Concepts:")
            for key, data in add_results.items():
                gpu_status = "GPU" if "True" in key else "CPU"
                if isinstance(data, dict) and 100 in data:
                    print(f"  {gpu_status} (100 concepts): {data[100]['concepts_per_sec']:.0f} concepts/sec")
        
        if query_results:
            print("\nQueries:")
            for key, data in query_results.items():
                gpu_status = "GPU" if "True" in key else "CPU"
                print(f"  {gpu_status}: {data['queries_per_sec']:.0f} queries/sec")
        
        if inference_results:
            print("\nInference:")
            for key, data in inference_results.items():
                gpu_status = "GPU" if "True" in key else "CPU"
                print(f"  {gpu_status}: {data['inferences_per_sec']:.0f} inferences/sec")
        
        if similarity_results:
            print("\nStructural Similarity:")
            for key, data in similarity_results.items():
                gpu_status = "GPU" if "True" in key else "CPU"
                print(f"  {gpu_status}: {data['comparisons_per_sec']:.0f} comparisons/sec")
        
        print(f"{'='*70}")


def run_benchmarks():
    """Run complete benchmark suite"""
    print("=" * 70)
    print("MLN PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)
    print()
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark adding concepts (different sizes)
    benchmark.benchmark_add_concepts([10, 50, 100, 500], use_gpu=False)
    
    # Benchmark queries
    benchmark.benchmark_queries(graph_size=100, num_queries=100, use_gpu=False)
    benchmark.benchmark_queries(graph_size=500, num_queries=50, use_gpu=False)
    
    # Benchmark inference
    benchmark.benchmark_inference(graph_size=50, use_gpu=False)
    benchmark.benchmark_inference(graph_size=100, use_gpu=False)
    
    # Benchmark similarity
    benchmark.benchmark_structural_similarity(10000, use_gpu=False)
    
    # Print summary and save
    benchmark.print_summary()
    benchmark.save_results()
    
    print("\n✓ Benchmark suite completed!")
    return benchmark


if __name__ == '__main__':
    run_benchmarks()
