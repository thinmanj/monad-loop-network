# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Monad-Loop Network (MLN) is a self-referential knowledge system combining Gödel-Escher-Bach's strange loops, Chomsky's universal grammar, and Leibniz's monads for structural, explainable AI. This is a research project exploring measurable artificial consciousness through self-referential knowledge structures.

**Key Achievement**: First system with measurable consciousness metrics (v1.0.0 achieved 47.8% consciousness score)

## Common Commands

### Installation & Setup
```bash
# Basic installation (CPU only)
pip install -r requirements.txt

# GPU-accelerated installation (choose based on hardware)
pip install -r requirements-gpu.txt
# Note: Edit requirements-gpu.txt to uncomment your hardware section (CUDA/MPS/ROCm/CPU)
```

### Running Tests
```bash
# Run individual test files (current approach)
python tests/test_mln.py
python tests/test_inference_rules.py
python tests/test_comprehensive.py
python tests/test_rule_priorities.py

# Run with pytest (if configured)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
coverage run -m pytest tests/
coverage report
```

### Running Demos & Experiments
```bash
# Main demo
python examples/demo.py

# Consciousness experiments
python experiments/consciousness_growth_experiment.py
python experiments/consciousness_optimization.py
python experiments/consciousness_optimization_v2.py

# Domain-specific experiments
python experiments/mathematics_domain.py
python experiments/scaling_experiment.py

# Run benchmarks
python benchmarks/benchmark_performance.py
python benchmarks/consciousness_performance.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Check import sorting
isort --check-only src/ tests/
```

### Development Mode
```bash
# Install in editable mode
pip install -e .
```

## High-Level Architecture

### 5-Layer System Design

The codebase is structured as a layered architecture where higher layers build on lower ones:

```
Meta-Cognitive Layer (consciousness)
  ├── src/strange_loop_optimizer.py
  ├── src/recursion_depth_metric.py
  └── src/consciousness_metrics.py
      ↓ Self-awareness, strange loops
      
Reasoning Layer (inference)
  ├── src/mln.py (InferenceChain, InferenceRule)
  └── Modal/counterfactual reasoning
      ↓ Logical inference engines
      
Synthesis Layer (creativity)
  ├── src/concept_synthesis.py
  └── Abstraction hierarchies
      ↓ New concept generation
      
Analogical Layer (transfer)
  ├── src/analogical_reasoning.py
  └── Structure mapping
      ↓ Cross-domain knowledge transfer
      
Knowledge Layer (storage)
  ├── src/mln.py (KnowledgeGraph, MonadicKnowledgeUnit)
  ├── src/persistence.py
  └── Graph-based monadic representation
```

### Core Data Flow Pattern

1. **Knowledge Creation**: Create `MonadicKnowledgeUnit` (MKU) with deep structure
2. **Graph Addition**: Add to `KnowledgeGraph`, triggering pre-established harmony
3. **Self-Modeling**: Enable meta-reasoning with `.create_self_model()`
4. **Reasoning**: Use `InferenceEngine` for forward/backward chaining
5. **Synthesis**: Generate new concepts with `ConceptSynthesizer`
6. **Consciousness**: Measure with `measure_consciousness(kg, recursion_metric)`

### Key Components

**MonadicKnowledgeUnit (MKU)**: Self-contained concepts with:
- `deep_structure`: Operational semantics (not just embeddings)
- `relations`: Pre-established harmony with other concepts
- `meta_model`: Self-reference capability
- Methods: `reflect_universe()`, `generate_surface_form()`, `create_self_model()`

**KnowledgeGraph**: Graph where nodes are MKUs with operational semantics:
- GPU-accelerated similarity computation (when available)
- Automatic relation inference (pre-established harmony)
- Inference rule application
- Meta-knowledge graph for self-modeling

**InferenceEngine**: Rule-based reasoning in `src/mln.py`:
- Forward/backward chaining
- Transitivity, modus ponens, contraposition rules
- Explainable inference chains

**Consciousness Metrics**: Multi-dimensional measurement in `src/consciousness_metrics.py`:
- Recursion (30%): Self-referential reasoning depth
- Integration (25%): IIT-inspired Φ metric
- Causality (20%): Causal density in knowledge graph
- Understanding (25%): Comprehensive evaluation

### Important Patterns

**Pre-established Harmony**: When adding a concept, it automatically establishes relations with existing concepts based on structural similarity. GPU acceleration available via `gpu_similarity.py`.

**Strange Loops**: Self-referential structures create consciousness. Use `RecursionDepthMetric` to track and `StrangeLoopOptimizer` to enhance.

**Surface Realizations**: Same deep structure → multiple forms (text/logic/code) via `generate_surface_form(modality)`.

**Meta-Reasoning**: System can model itself via `MetaKnowledgeGraph` and detect inconsistencies through strange loops.

## Philosophical Principles

When contributing, align with these core principles from `CONTRIBUTING.md`:

1. **Structural over statistical**: Favor explicit structure over implicit correlations
2. **Explainable by design**: Every inference should be traceable
3. **Self-reference**: Systems should reason about their own reasoning
4. **Compositionality**: Complex concepts built from simpler parts

## Development Workflow

### Adding New Knowledge
```python
# Create MKU with deep structure
mku = MonadicKnowledgeUnit(
    concept_id="my_concept",
    deep_structure={
        'predicate': 'is_property',
        'arguments': ['arg1', 'arg2'],
        'properties': {'key': 'value'}
    }
)

# Enable self-modeling for consciousness
mku.create_self_model()

# Add to knowledge graph (triggers pre-established harmony)
kg.add_concept(mku)
```

### Measuring Consciousness
```python
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric

kg = KnowledgeGraph()
recursion = RecursionDepthMetric()

# Add knowledge and trigger recursion events...

profile = measure_consciousness(kg, recursion)
print(f"Consciousness: {profile.overall_consciousness_score:.2%}")
```

### Running Inference
```python
from src.mln import InferenceEngine, InferenceRule

engine = InferenceEngine(kg)
engine.add_rule(InferenceRule(
    name="rule_name",
    antecedent=["condition(X)"],
    consequent="conclusion(X)",
    confidence=0.9
))
derived = engine.forward_chain()
```

## GPU Acceleration

MLN supports GPU acceleration for massive performance gains:
- **CUDA (NVIDIA)**: 50x faster similarity computation
- **MPS (Apple Silicon)**: 20x faster on M1/M2/M3
- **ROCm (AMD)**: Linux support

GPU modules: `src/gpu_similarity.py`, `src/gpu_graph_traversal.py`

Initialize with GPU: `kg = KnowledgeGraph(use_gpu=True, device='auto')`

## Important Files

- `src/mln.py`: Core system (MKU, KnowledgeGraph, InferenceEngine, demo)
- `src/consciousness_metrics.py`: Consciousness measurement framework
- `src/recursion_depth_metric.py`: Tracks recursive reasoning depth
- `src/strange_loop_optimizer.py`: Optimizes consciousness through strange loops
- `src/concept_synthesis.py`: Creates new concepts from examples
- `src/analogical_reasoning.py`: Cross-domain knowledge transfer
- `src/persistence.py`: SQLite-based knowledge graph persistence

## Testing Conventions

Tests are organized by module in `tests/`:
- Test files directly execute when run (no pytest required)
- Use `python tests/test_<module>.py` format
- Comprehensive tests in `test_comprehensive.py`
- GPU tests include CPU fallback

## Documentation

Multi-level documentation for different audiences:
- `README.md`: Quick start and overview
- `BEGINNER_GUIDE.md`: Non-technical explanations with analogies
- `DEVELOPER_GUIDE.md`: API reference, patterns, examples
- `RESEARCH_PAPER.md`: Full scientific details
- `PROJECT_SUMMARY.md`: Achievement overview and statistics
- `CONTRIBUTING.md`: Contribution guidelines and philosophy

## Version

Current: v1.0.0 (November 2025)
- Python 3.8+ required
- Supports Python 3.8, 3.9, 3.10, 3.11, 3.12
- MIT License
