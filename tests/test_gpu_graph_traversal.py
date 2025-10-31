#!/usr/bin/env python3
"""
Tests for GPU-accelerated graph traversal
Issue #40
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gpu_graph_traversal import GPUGraphTraversal, TORCH_AVAILABLE


def test_initialization():
    """Test GPU graph traversal initializes correctly"""
    traversal = GPUGraphTraversal(device='cpu')
    assert traversal.max_depth == 10
    assert traversal.adjacency_matrix is None
    print("✓ Initialization test passed")


def test_build_adjacency_matrix():
    """Test building adjacency matrix from edges"""
    traversal = GPUGraphTraversal(device='cpu')
    
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('A', 'C'),
    ]
    
    traversal.build_adjacency_matrix(edges)
    
    assert traversal.num_nodes == 3
    assert 'A' in traversal.node_to_idx
    assert 'B' in traversal.node_to_idx
    assert 'C' in traversal.node_to_idx
    assert traversal.adjacency_matrix is not None
    print("✓ Build adjacency matrix test passed")


def test_single_path_cpu():
    """Test finding a single path on CPU"""
    traversal = GPUGraphTraversal(device='cpu')
    
    # Simple chain: A -> B -> C
    edges = [('A', 'B'), ('B', 'C')]
    traversal.build_adjacency_matrix(edges)
    
    paths = traversal.parallel_bfs(['A'], ['C'])
    
    assert len(paths) == 1
    assert paths[0] is not None
    assert paths[0] == ['A', 'B', 'C']
    print("✓ Single path (CPU) test passed")


def test_no_path_cpu():
    """Test when no path exists"""
    traversal = GPUGraphTraversal(device='cpu')
    
    # Disconnected: A -> B, C -> D
    edges = [('A', 'B'), ('C', 'D')]
    traversal.build_adjacency_matrix(edges)
    
    paths = traversal.parallel_bfs(['A'], ['D'])
    
    assert len(paths) == 1
    assert paths[0] is None  # No path from A to D
    print("✓ No path test passed")


def test_multiple_queries_cpu():
    """Test multiple queries in batch on CPU"""
    traversal = GPUGraphTraversal(device='cpu')
    
    # Graph: A -> B -> C -> D
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
    traversal.build_adjacency_matrix(edges)
    
    # Multiple queries
    queries = [
        ('A', 'B'),  # Short path
        ('A', 'C'),  # Medium path
        ('A', 'D'),  # Long path
        ('B', 'D'),  # Different start
    ]
    
    paths = traversal.batch_query(queries)
    
    assert len(paths) == 4
    assert paths[0] == ['A', 'B']
    assert paths[1] == ['A', 'B', 'C']
    assert paths[2] == ['A', 'B', 'C', 'D']
    assert paths[3] == ['B', 'C', 'D']
    print("✓ Multiple queries (CPU) test passed")


def test_max_depth_limit():
    """Test that max_depth limits search"""
    traversal = GPUGraphTraversal(device='cpu', max_depth=2)
    
    # Long chain: A -> B -> C -> D -> E
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
    traversal.build_adjacency_matrix(edges)
    
    # Try to reach E from A (requires 4 hops, but max_depth=2)
    paths = traversal.parallel_bfs(['A'], ['E'])
    
    assert paths[0] is None  # Should not find path due to depth limit
    print("✓ Max depth limit test passed")


def test_cycle_handling():
    """Test graph with cycles"""
    traversal = GPUGraphTraversal(device='cpu')
    
    # Graph with cycle: A -> B -> C -> A
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    traversal.build_adjacency_matrix(edges)
    
    paths = traversal.parallel_bfs(['A'], ['C'])
    
    assert len(paths) == 1
    assert paths[0] is not None
    assert 'C' in paths[0]  # Should find path despite cycle
    print("✓ Cycle handling test passed")


def test_gpu_availability():
    """Test GPU detection"""
    if TORCH_AVAILABLE:
        traversal = GPUGraphTraversal(device='auto')
        print(f"  Detected device: {traversal.device}")
        print(f"  Using GPU: {traversal.use_gpu}")
    else:
        print("  PyTorch not available, skipping GPU detection")
    print("✓ GPU availability test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("GPU Graph Traversal Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_initialization,
        test_build_adjacency_matrix,
        test_single_path_cpu,
        test_no_path_cpu,
        test_multiple_queries_cpu,
        test_max_depth_limit,
        test_cycle_handling,
        test_gpu_availability,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
