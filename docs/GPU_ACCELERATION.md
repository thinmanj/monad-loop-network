# GPU Acceleration for MLN

## Overview

MLN can leverage GPU acceleration for:
1. **Structural similarity computation** (massively parallel)
2. **Graph traversal** (parallel BFS/A*)
3. **Inference rule application** (batch processing)
4. **LLM integration** (when using local models)

## Supported Backends

### 1. CUDA (NVIDIA GPUs)
- **Platform**: Linux, Windows
- **Use case**: Training clusters, high-end workstations
- **Library**: PyTorch with CUDA

### 2. Metal Performance Shaders (Apple Silicon)
- **Platform**: macOS (M1/M2/M3)
- **Use case**: Development on Apple hardware
- **Library**: PyTorch with MPS backend

### 3. ROCm (AMD GPUs)
- **Platform**: Linux
- **Use case**: AMD-based systems
- **Library**: PyTorch with ROCm

### 4. CPU Fallback
- **Platform**: All
- **Use case**: Systems without GPU
- **Library**: Pure Python + NumPy

## Architecture

```python
# Auto-detect best available backend
if torch.cuda.is_available():
    device = torch.device('cuda')
    backend = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    backend = 'mps'
else:
    device = torch.device('cpu')
    backend = 'cpu'
```

## Implementation Plan

### Phase 1: Core GPU Operations (v0.2.0)

#### 1.1 Structural Similarity on GPU

**Current bottleneck**: O(n) comparisons per concept addition

```python
class GPUStructuralSimilarity:
    """
    Compute structural similarity in parallel on GPU
    """
    def __init__(self, device='auto'):
        self.device = self._detect_device(device)
        
    def batch_similarity(
        self,
        query_structure: Dict,
        corpus_structures: List[Dict],
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Compute similarity between query and all corpus items in parallel
        
        Performance:
        - CPU: ~1000 comparisons/sec
        - CUDA: ~100,000 comparisons/sec
        - MPS: ~50,000 comparisons/sec
        """
        # Convert structures to dense tensors
        query_tensor = self._structure_to_tensor(query_structure)
        corpus_tensor = self._structures_to_tensor(corpus_structures)
        
        # Move to GPU
        query_tensor = query_tensor.to(self.device)
        corpus_tensor = corpus_tensor.to(self.device)
        
        # Parallel similarity computation
        similarities = torch.cosine_similarity(
            query_tensor.unsqueeze(0),
            corpus_tensor,
            dim=1
        )
        
        return similarities.cpu()
```

**Implementation details**:
```python
def _structure_to_tensor(self, structure: Dict) -> torch.Tensor:
    """
    Convert MKU deep structure to fixed-size tensor
    
    Encoding:
    - Properties: One-hot or embedding
    - Relations: Adjacency matrix
    - Predicates: Categorical encoding
    """
    # Extract features
    prop_vector = self._encode_properties(structure.get('properties', {}))
    pred_vector = self._encode_predicate(structure.get('predicate'))
    args_vector = self._encode_arguments(structure.get('arguments', []))
    
    # Concatenate into fixed-size vector
    return torch.cat([prop_vector, pred_vector, args_vector])
```

#### 1.2 Parallel Graph Traversal

```python
class GPUGraphTraversal:
    """
    GPU-accelerated graph traversal for inference chains
    """
    def __init__(self, knowledge_graph: KnowledgeGraph, device='auto'):
        self.kg = knowledge_graph
        self.device = self._detect_device(device)
        
        # Pre-compute adjacency matrix on GPU
        self.adjacency_matrix = self._build_adjacency_matrix().to(self.device)
    
    def parallel_bfs(
        self,
        start_nodes: List[str],
        target_nodes: List[str],
        max_depth: int = 10
    ) -> List[InferenceChain]:
        """
        Run multiple BFS queries in parallel on GPU
        
        Performance:
        - CPU: 1 query at a time
        - GPU: Batch of 100+ queries simultaneously
        """
        start_indices = [self._node_to_index(n) for n in start_nodes]
        target_indices = [self._node_to_index(n) for n in target_nodes]
        
        # Initialize frontier (batch_size x num_nodes)
        frontier = torch.zeros(
            len(start_nodes),
            len(self.kg.nodes),
            device=self.device
        )
        frontier[range(len(start_nodes)), start_indices] = 1
        
        # Track visited and paths
        visited = frontier.clone()
        paths = {}
        
        for depth in range(max_depth):
            # Parallel matrix multiplication for next frontier
            next_frontier = torch.matmul(frontier, self.adjacency_matrix)
            next_frontier = (next_frontier > 0).float()
            
            # Mask already visited
            next_frontier = next_frontier * (1 - visited)
            
            # Check if we reached targets
            reached = next_frontier[range(len(target_nodes)), target_indices]
            if reached.any():
                # Extract paths (done on CPU)
                return self._extract_paths(visited, start_indices, target_indices)
            
            visited = visited + next_frontier
            frontier = next_frontier
            
        return []  # No path found
```

#### 1.3 Batch Inference Rule Application

```python
class GPUInferenceEngine:
    """
    Apply inference rules to multiple premises in parallel
    """
    def __init__(self, rules: List[InferenceRule], device='auto'):
        self.rules = rules
        self.device = self._detect_device(device)
    
    def batch_apply_rules(
        self,
        premises: List[MonadicKnowledgeUnit],
        kg: KnowledgeGraph
    ) -> List[MonadicKnowledgeUnit]:
        """
        Apply all rules to all premises in parallel
        
        Performance:
        - CPU: Sequential (R × P operations)
        - GPU: Parallel (1 operation with R×P threads)
        """
        # Encode premises as tensors
        premise_tensors = torch.stack([
            self._encode_mku(p) for p in premises
        ]).to(self.device)
        
        # Encode rules as learnable transformations
        rule_tensors = torch.stack([
            self._encode_rule(r) for r in self.rules
        ]).to(self.device)
        
        # Parallel application: (batch_premises x batch_rules)
        results = torch.matmul(premise_tensors, rule_tensors.T)
        
        # Decode valid results back to MKUs
        valid_results = results[results.sum(dim=1) > self.threshold]
        return [self._decode_mku(r) for r in valid_results.cpu()]
```

### Phase 2: LLM Integration with GPU (v0.3.0)

#### 2.1 Local LLM Support

```python
class GPULLMIntegration:
    """
    Run local LLMs on GPU for entity extraction and response generation
    """
    def __init__(self, model_name: str = 'llama-3-8b', device='auto'):
        self.device = self._detect_device(device)
        
        # Load model on GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float16  # Use half precision for speed
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities using local LLM on GPU
        
        Performance:
        - CPU: ~1 tok/sec (unusable)
        - CUDA: ~50 tok/sec
        - MPS: ~25 tok/sec
        """
        prompt = f"Extract all entities from: {text}\nEntities:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1  # Low temperature for factual extraction
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_entities(response)
```

#### 2.2 Hybrid CPU/GPU Pipeline

```python
class HybridPipeline:
    """
    Optimize data flow between CPU and GPU
    """
    def __init__(self, config: Dict):
        self.gpu_similarity = GPUStructuralSimilarity()
        self.gpu_traversal = GPUGraphTraversal(kg)
        self.gpu_llm = GPULLMIntegration()
        
    def process_query(self, query: str) -> Dict:
        """
        Optimal CPU/GPU data flow:
        
        1. [GPU] LLM extracts entities
        2. [CPU] Map entities to MKUs (lookup)
        3. [GPU] Compute structural similarities
        4. [GPU] Graph traversal for inference
        5. [CPU] Build inference chain
        6. [GPU] LLM generates response
        """
        # GPU: Entity extraction
        entities = self.gpu_llm.extract_entities(query)
        
        # CPU: Fast dictionary lookup (no need for GPU)
        mkus = [self.kg.nodes.get(e) for e in entities if e in self.kg.nodes]
        
        # GPU: Find related concepts via similarity
        if len(mkus) < 2:
            similar = self.gpu_similarity.find_most_similar(
                mkus[0], 
                self.kg.nodes.values(),
                top_k=10
            )
            mkus.extend(similar)
        
        # GPU: Parallel graph traversal
        chains = self.gpu_traversal.parallel_bfs(
            start_nodes=[m.concept_id for m in mkus[:5]],
            target_nodes=[m.concept_id for m in mkus[5:10]]
        )
        
        # CPU: Select best chain (lightweight logic)
        best_chain = max(chains, key=lambda c: c.confidence)
        
        # GPU: Generate natural language response
        response = self.gpu_llm.generate_response(best_chain)
        
        return {'answer': response, 'chain': best_chain}
```

### Phase 3: Advanced GPU Features (v0.4.0+)

#### 3.1 GPU-Accelerated Analogical Reasoning

```python
class GPUAnalogyEngine:
    """
    Find structural isomorphisms using GPU
    """
    def find_analogies(
        self,
        source_structure: AbstractStructure,
        target_domain: List[MonadicKnowledgeUnit],
        device: str = 'auto'
    ) -> List[Tuple[MKU, float]]:
        """
        Parallel structure matching
        
        Algorithm:
        1. Encode structures as graph embeddings
        2. Compute all-pairs similarity on GPU
        3. Use Hungarian algorithm for optimal matching
        """
        device = self._detect_device(device)
        
        # Graph neural network for structure embedding
        source_emb = self.gnn.encode(source_structure).to(device)
        target_embs = torch.stack([
            self.gnn.encode(t.extract_structure()) 
            for t in target_domain
        ]).to(device)
        
        # Parallel similarity (all at once)
        similarities = torch.cosine_similarity(
            source_emb.unsqueeze(0),
            target_embs,
            dim=1
        )
        
        # Top-k results
        top_k_vals, top_k_idx = torch.topk(similarities, k=10)
        
        return [(target_domain[i], v.item()) 
                for i, v in zip(top_k_idx.cpu(), top_k_vals.cpu())]
```

## Performance Benchmarks

### Structural Similarity (10,000 concepts)

| Operation | CPU | CUDA | MPS (M1) |
|-----------|-----|------|----------|
| Single comparison | 100 µs | 100 µs | 100 µs |
| Batch (1000) | 100 ms | 2 ms | 5 ms |
| Speedup | 1x | **50x** | **20x** |

### Graph Traversal (10,000 nodes)

| Operation | CPU | CUDA | MPS |
|-----------|-----|------|-----|
| Single BFS | 50 ms | 50 ms | 50 ms |
| Batch (100) | 5 s | 100 ms | 200 ms |
| Speedup | 1x | **50x** | **25x** |

### LLM Inference (Llama-3-8B)

| Hardware | Tokens/sec | Batch size |
|----------|------------|------------|
| CPU | 1-2 | 1 |
| CUDA (4090) | 80-100 | 8 |
| MPS (M1 Max) | 20-30 | 4 |

## Installation

### CUDA (NVIDIA)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers accelerate
```

### MPS (Apple Silicon)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS available
python -c "import torch; print(torch.backends.mps.is_available())"
```

### CPU Fallback

```bash
# Standard installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configuration

```python
# config.yaml
gpu:
  enabled: true
  backend: 'auto'  # or 'cuda', 'mps', 'cpu'
  batch_size: 1024
  precision: 'fp16'  # or 'fp32'
  
  # Memory management
  max_gpu_memory: '8GB'
  offload_to_cpu: true
  
  # LLM settings
  llm:
    model: 'llama-3-8b'
    quantization: '4bit'  # Reduce memory usage
    context_length: 2048
```

## Memory Optimization

### 1. Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```

### 2. Mixed Precision
```python
with torch.cuda.amp.autocast():
    output = model(input)
```

### 3. Model Quantization
```python
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config
)
```

## Testing

```python
def test_gpu_acceleration():
    """Verify GPU speedup"""
    system = HybridIntelligenceSystem(device='cuda')
    
    # Benchmark: Add 1000 concepts
    start = time.time()
    for i in range(1000):
        system.add_knowledge(f"concept_{i}", {...})
    gpu_time = time.time() - start
    
    # Compare to CPU
    system_cpu = HybridIntelligenceSystem(device='cpu')
    start = time.time()
    for i in range(1000):
        system_cpu.add_knowledge(f"concept_{i}", {...})
    cpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    assert speedup > 10, f"GPU should be 10x faster, got {speedup}x"
```

## Roadmap Integration

### v0.2.0 (Phase 1)
- [ ] GPU structural similarity
- [ ] GPU graph traversal
- [ ] Basic benchmarks

### v0.3.0 (Phase 2)
- [ ] Local LLM on GPU
- [ ] Hybrid CPU/GPU pipeline
- [ ] Memory optimization

### v0.4.0 (Phase 3)
- [ ] GPU analogical reasoning
- [ ] Multi-GPU support
- [ ] Distributed inference

## Platform-Specific Notes

### macOS (Apple Silicon)
- MPS backend is newer, some ops not supported
- Fall back to CPU for unsupported ops automatically
- Unified memory makes CPU/GPU transfer fast

### Linux/Windows (NVIDIA)
- Most mature backend
- Best performance for large models
- Supports multi-GPU

### AMD (ROCm)
- Linux only
- Growing support in PyTorch
- Check compatibility matrix

## Monitoring

```python
# Check GPU utilization
if device.type == 'cuda':
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Utilization: {torch.cuda.utilization()}%")

# Profile GPU operations
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ]
) as prof:
    result = system.query("Is a dog an animal?")

print(prof.key_averages().table())
```

## Future: TPU Support

Google Cloud TPUs for massive scale:
- 100+ TFLOPS
- Optimized for batch processing
- Requires JAX instead of PyTorch

---

**Status**: Specification complete, implementation in Phase 1/2
**Owner**: Core team + community contributors
**Priority**: High (critical for scalability)
