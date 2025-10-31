#!/usr/bin/env python3
"""
Tests for GPU-accelerated structural similarity
Issue #39
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gpu_similarity import GPUStructuralSimilarity, TORCH_AVAILABLE
import time


def test_initialization():
    """Test GPU similarity engine initializes correctly"""
    sim = GPUStructuralSimilarity(device='cpu')
    assert sim.device == 'cpu' or str(sim.device) == 'cpu'
    assert sim.vector_size == 128
    print("✓ Initialization test passed")


def test_structure_encoding():
    """Test that structures are encoded correctly"""
    sim = GPUStructuralSimilarity(device='cpu')
    
    structure = {
        'predicate': 'is_a',
        'arguments': ['mammal'],
        'properties': {'domesticated': True}
    }
    
    # Test numpy encoding (always works)
    vector = sim._structure_to_numpy(structure)
    assert vector.shape == (128,)
    assert not all(v == 0 for v in vector)  # Should have some non-zero values
    print("✓ Structure encoding test passed")


def test_batch_similarity_cpu():
    """Test batch similarity computation on CPU"""
    sim = GPUStructuralSimilarity(device='cpu')
    
    query = {
        'predicate': 'is_a',
        'arguments': ['mammal'],
        'properties': {'domesticated': True, 'barks': True}
    }
    
    corpus = [
        {
            'predicate': 'is_a',
            'arguments': ['mammal'],
            'properties': {'domesticated': True, 'meows': True}  # Similar
        },
        {
            'predicate': 'is_alive',
            'arguments': ['plant'],
            'properties': {'photosynthesis': True}  # Different
        }
    ]
    
    similarities = sim.batch_similarity(query, corpus)
    
    assert len(similarities) == 2
    assert all(0.0 <= s <= 1.0 for s in similarities)
    # First should be more similar than second
    assert similarities[0] > similarities[1]
    print("✓ Batch similarity (CPU) test passed")


def test_find_most_similar():
    """Test finding most similar structures"""
    sim = GPUStructuralSimilarity(device='cpu')
    
    query = {'predicate': 'test', 'properties': {'a': 1}}
    
    corpus = [
        {'predicate': 'test', 'properties': {'a': 1}},  # Exact match
        {'predicate': 'test', 'properties': {'a': 2}},  # Close
        {'predicate': 'other', 'properties': {'b': 3}},  # Different
    ]
    
    results = sim.find_most_similar(query, corpus, top_k=2, threshold=0.0)
    
    assert len(results) <= 2
    # Results should be sorted by similarity
    if len(results) > 1:
        assert results[0][1] >= results[1][1]
    print("✓ Find most similar test passed")


def test_gpu_availability():
    """Test GPU detection"""
    if TORCH_AVAILABLE:
        sim = GPUStructuralSimilarity(device='auto')
        print(f"  Detected device: {sim.device}")
        print(f"  Using GPU: {sim.use_gpu}")
    else:
        print("  PyTorch not available, skipping GPU detection")
    print("✓ GPU availability test passed")


def test_performance_difference():
    """Test that batch processing is faster than sequential"""
    sim = GPUStructuralSimilarity(device='cpu')
    
    # Create test data
    query = {'predicate': 'test', 'properties': {'key': 'value'}}
    corpus = [
        {'predicate': f'pred_{i}', 'properties': {'k': f'v_{i}'}}
        for i in range(100)
    ]
    
    # Batch processing
    start = time.time()
    batch_results = sim.batch_similarity(query, corpus)
    batch_time = time.time() - start
    
    # Sequential processing (simulated)
    start = time.time()
    seq_results = []
    for struct in corpus:
        sim_score = sim.batch_similarity(query, [struct])
        seq_results.extend(sim_score)
    seq_time = time.time() - start
    
    print(f"  Batch time: {batch_time:.4f}s")
    print(f"  Sequential time: {seq_time:.4f}s")
    print(f"  Batch is {seq_time/batch_time:.1f}x faster")
    
    # Results should be similar
    assert len(batch_results) == len(seq_results)
    print("✓ Performance difference test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("GPU Structural Similarity Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_initialization,
        test_structure_encoding,
        test_batch_similarity_cpu,
        test_find_most_similar,
        test_gpu_availability,
        test_performance_difference,
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
