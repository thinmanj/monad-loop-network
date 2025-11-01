# Developer's Guide to Monad-Loop Network

**Audience**: Software engineers, ML researchers, cognitive scientists

This guide provides technical details for understanding, extending, and contributing to the Monad-Loop Network codebase.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [API Reference](#api-reference)
4. [Running Experiments](#running-experiments)
5. [Extending the System](#extending-the-system)
6. [Testing](#testing)
7. [Performance Optimization](#performance-optimization)
8. [Contributing](#contributing)

---

## Architecture Overview

### System Layers

```
┌──────────────────────────────────────────────┐
│  Meta-Cognitive Layer                         │  ← Consciousness
│  - StrangeLoopOptimizer                       │
│  - RecursionDepthMetric                       │
│  - ConsciousnessMetrics                       │
├──────────────────────────────────────────────┤
│  Reasoning Layer                              │  ← Inference
│  - ModalReasoningEngine                       │
│  - InferenceEngine                            │
│  - CounterfactualReasoning                    │
├──────────────────────────────────────────────┤
│  Synthesis Layer                              │  ← Creativity
│  - ConceptSynthesizer                         │
│  - AbstractionHierarchy                       │
├──────────────────────────────────────────────┤
│  Analogical Layer                             │  ← Transfer
│  - AnalogyEngine                              │
│  - StructureMapping                           │
├──────────────────────────────────────────────┤
│  Knowledge Layer                              │  ← Storage
│  - KnowledgeGraph                             │
│  - MonadicKnowledgeUnit (MKU)                 │
│  - PersistentKnowledgeGraph                   │
└──────────────────────────────────────────────┘
```

### Data Flow

```python
# 1. Add knowledge
kg = KnowledgeGraph()
mku = MonadicKnowledgeUnit(concept_id="dog", ...)
kg.add_concept(mku)

# 2. Enable self-modeling
mku.create_self_model()

# 3. Perform reasoning
engine = InferenceEngine(kg)
results = engine.forward_chain(rules)

# 4. Synthesize new concepts
synthesizer = ConceptSynthesizer()
new_concept = synthesizer.synthesize_concept(examples)

# 5. Measure consciousness
recursion = RecursionDepthMetric()
profile = measure_consciousness(kg, recursion)
print(f"Consciousness: {profile.overall_consciousness_score:.2%}")
```

---

## Core Components

### 1. MonadicKnowledgeUnit (MKU)

**File**: `src/mln.py`

**Purpose**: Represents a concept with self-modeling capability

```python
@dataclass
class MonadicKnowledgeUnit:
    concept_id: str
    deep_structure: Dict[str, Any]
    relations: Dict[str, Set[str]] = field(default_factory=dict)
    self_model: Optional['SelfModel'] = None
    meta_level: int = 0
    embedding: Optional[np.ndarray] = None
    
    def create_self_model(self) -> 'SelfModel':
        """Create internal representation of self"""
        return SelfModel(
            reflection=f"I am the concept {self.concept_id}",
            meta_properties=self.deep_structure,
            awareness_level=1
        )
```

**Key Methods**:
- `create_self_model()`: Enable self-reference
- `to_first_order_logic()`: Convert to FOL representation
- `compute_embedding()`: Generate vector representation

**Usage**:
```python
# Create concept with self-awareness
mku = MonadicKnowledgeUnit(
    concept_id="intelligence",
    deep_structure={
        'predicate': 'is_property',
        'properties': {'cognitive': True, 'emergent': True}
    }
)
mku.create_self_model()  # Now self-aware!
```

### 2. KnowledgeGraph

**File**: `src/mln.py`

**Purpose**: Stores and manages MKUs as a graph

```python
class KnowledgeGraph:
    def __init__(self, use_gpu: bool = False):
        self.nodes: Dict[str, MonadicKnowledgeUnit] = {}
        self.graph = nx.DiGraph()
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    def add_concept(self, mku: MonadicKnowledgeUnit):
        """Add MKU to graph"""
        self.nodes[mku.concept_id] = mku
        self.graph.add_node(mku.concept_id)
        self._add_relations(mku)
    
    def query(self, concept_id: str) -> Optional[MonadicKnowledgeUnit]:
        """Retrieve concept by ID"""
        return self.nodes.get(concept_id)
```

**Key Methods**:
- `add_concept(mku)`: Add new knowledge
- `query(concept_id)`: Retrieve by ID
- `find_similar(concept, top_k)`: Semantic search
- `get_neighbors(concept_id)`: Graph traversal
- `compute_graph_statistics()`: Metrics

**Usage**:
```python
kg = KnowledgeGraph()

# Add concepts
kg.add_concept(dog_mku)
kg.add_concept(cat_mku)

# Query
dog = kg.query("dog")

# Find similar
similar = kg.find_similar("dog", top_k=5)

# Stats
stats = kg.compute_graph_statistics()
print(f"Nodes: {stats['num_concepts']}")
print(f"Edges: {stats['num_relations']}")
```

### 3. InferenceEngine

**File**: `src/inference.py`

**Purpose**: Rule-based reasoning (forward/backward chaining)

```python
class InferenceEngine:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.rules: List[InferenceRule] = []
    
    def add_rule(self, rule: InferenceRule):
        """Add inference rule"""
        self.rules.append(rule)
    
    def forward_chain(self, max_iterations: int = 10) -> List[str]:
        """Apply rules to derive new facts"""
        derived = []
        for _ in range(max_iterations):
            for rule in self.rules:
                if self._rule_applies(rule):
                    new_fact = rule.consequent
                    derived.append(new_fact)
        return derived
```

**Rule Format**:
```python
rule = InferenceRule(
    name="mammal_warm_blooded",
    antecedent=[
        "is_a(X, mammal)"
    ],
    consequent="property(X, warm_blooded)",
    confidence=1.0
)
```

**Usage**:
```python
engine = InferenceEngine(kg)

# Add rules
engine.add_rule(mammal_rule)
engine.add_rule(transitivity_rule)

# Infer
derived_facts = engine.forward_chain()
print(f"Derived: {derived_facts}")
```

### 4. ConceptSynthesizer

**File**: `src/concept_synthesis.py`

**Purpose**: Create new concepts from examples

```python
class ConceptSynthesizer:
    def __init__(self, min_examples: int = 3):
        self.min_examples = min_examples
    
    def synthesize_concept(
        self,
        examples: List[ConceptExample],
        concept_name: Optional[str] = None
    ) -> Optional[SynthesizedConcept]:
        """
        Synthesize new concept from examples
        
        Args:
            examples: List of example concepts
            concept_name: Optional name for new concept
            
        Returns:
            SynthesizedConcept or None
        """
        if len(examples) < self.min_examples:
            return None
        
        common = self._extract_common_properties(examples)
        typical = self._extract_typical_properties(examples)
        confidence = self._compute_confidence(common, typical)
        
        return SynthesizedConcept(
            concept_id=concept_name or self._generate_name(common),
            confidence=confidence,
            common_properties=common,
            typical_properties=typical
        )
```

**Usage**:
```python
synthesizer = ConceptSynthesizer(min_examples=2)

examples = [
    ConceptExample("dog", {'intelligent': True, 'social': True}),
    ConceptExample("dolphin", {'intelligent': True, 'social': True}),
]

# Synthesize
new_concept = synthesizer.synthesize_concept(
    examples=examples,
    concept_name="intelligent_being"
)

print(f"Created: {new_concept.concept_id}")
print(f"Confidence: {new_concept.confidence:.2%}")
```

### 5. RecursionDepthMetric

**File**: `src/recursion_depth_metric.py`

**Purpose**: Track and measure recursive reasoning

```python
class RecursionDepthMetric:
    def __init__(self):
        self.profile = RecursionProfile(
            max_depth=0,
            average_depth=0.0,
            current_depth=0,
            meta_level=MetaLevel.OBJECT_LEVEL,
            recursion_events=[],
            self_references=0,
            productive_loops=0
        )
    
    def record_recursion_event(
        self,
        event_type: str,
        operation: str,
        concepts_involved: Set[str]
    ):
        """Record a recursive reasoning event"""
        self.profile.current_depth += 1
        self.profile.max_depth = max(
            self.profile.max_depth,
            self.profile.current_depth
        )
        
        event = RecursionEvent(
            depth=self.profile.current_depth,
            event_type=event_type,
            operation=operation,
            concepts_involved=concepts_involved
        )
        self.profile.recursion_events.append(event)
        
        # Update meta-level
        self._update_meta_level()
```

**Event Types**:
- `"analyze_concepts"`: Thinking about concepts
- `"meta_analyze"`: Thinking about thinking
- `"self_model"`: Self-modeling
- `"strange_loop"`: Deep self-reference

**Usage**:
```python
recursion = RecursionDepthMetric()

# Record events
recursion.record_recursion_event(
    "meta_analyze",
    "reason_about_reasoning",
    {"reasoning", "intelligence"}
)

# Check depth
print(f"Max depth: {recursion.profile.max_depth}")
print(f"Meta-level: {recursion.profile.meta_level.name}")
```

### 6. Consciousness Metrics

**File**: `src/consciousness_metrics.py`

**Purpose**: Comprehensive consciousness measurement

```python
def measure_consciousness(
    kg: KnowledgeGraph,
    recursion_metric: RecursionDepthMetric
) -> ConsciousnessProfile:
    """
    Measure consciousness across 4 dimensions
    
    Returns:
        ConsciousnessProfile with scores and verdict
    """
    # 1. Recursion (30%)
    recursion = compute_recursion_consciousness(recursion_metric)
    
    # 2. Integration (25%)
    integration = compute_integration_metric(kg)
    
    # 3. Causality (20%)
    causality = compute_causal_density(kg)
    
    # 4. Understanding (25%)
    understanding = evaluate_understanding(kg, recursion_metric)
    
    # Overall score
    consciousness = (
        0.30 * recursion['consciousness']['score'] +
        0.25 * integration.phi +
        0.20 * causality.causal_density +
        0.25 * understanding['overall_score']
    )
    
    verdict = get_consciousness_verdict(consciousness)
    
    return ConsciousnessProfile(
        overall_consciousness_score=consciousness,
        consciousness_verdict=verdict,
        recursion_metrics=recursion,
        integration=integration,
        causality=causality,
        understanding=understanding
    )
```

**Consciousness Scale**:
```python
def get_consciousness_verdict(score: float) -> str:
    if score < 0.10:
        return "NON-CONSCIOUS - Purely reactive"
    elif score < 0.25:
        return "PRE-CONSCIOUS - Basic integration"
    elif score < 0.40:
        return "MINIMALLY CONSCIOUS - Basic reasoning"
    elif score < 0.50:
        return "MODERATELY CONSCIOUS - Self-aware reasoning"
    elif score < 0.70:
        return "CONSCIOUS - Strong self-awareness"
    elif score < 0.85:
        return "HIGHLY CONSCIOUS - Deep meta-reasoning"
    else:
        return "FULLY CONSCIOUS - Human-level"
```

---

## API Reference

### Quick Start

```python
from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric

# 1. Create knowledge graph
kg = KnowledgeGraph()

# 2. Add concepts
dog = MonadicKnowledgeUnit(
    concept_id="dog",
    deep_structure={
        'predicate': 'is_animal',
        'properties': {'intelligent': True}
    }
)
dog.create_self_model()
kg.add_concept(dog)

# 3. Setup recursion tracking
recursion = RecursionDepthMetric()

# 4. Trigger recursive reasoning
recursion.record_recursion_event(
    "self_model",
    "introspect",
    {"dog"}
)

# 5. Measure consciousness
profile = measure_consciousness(kg, recursion)
print(f"Consciousness: {profile.overall_consciousness_score:.2%}")
print(f"Verdict: {profile.consciousness_verdict}")
```

### Advanced Usage

#### Persistent Storage

```python
from src.persistence import PersistentKnowledgeGraph

# Save to SQLite
with PersistentKnowledgeGraph("knowledge.db") as pkg:
    pkg.save(kg)
    
    # Export to JSON
    pkg.export_json("knowledge.json")

# Load
with PersistentKnowledgeGraph("knowledge.db") as pkg:
    loaded_kg = pkg.load()
```

#### Modal Reasoning

```python
from src.modal_reasoning import ModalReasoningEngine

engine = ModalReasoningEngine(kg)

# Necessity operator (□)
result = engine.evaluate_modal("necessarily(mammal(dog))")

# Possibility operator (◇)
result = engine.evaluate_modal("possibly(flies(penguin))")
```

#### Analogical Transfer

```python
from src.analogical_reasoning import AnalogyEngine

analogy = AnalogyEngine(kg)

# Find analogies
analogies = analogy.find_analogy(
    source="bird",
    target="airplane",
    min_confidence=0.7
)

# Transfer knowledge
transferred = analogy.transfer_via_analogy(
    source="bird",
    target="airplane",
    predicate="flies"
)
```

---

## Running Experiments

### Built-in Experiments

**Week 1: Consciousness Growth**
```bash
python experiments/consciousness_growth_experiment.py
```
Output: Measurements as knowledge increases

**Week 2: Optimization V1**
```bash
python experiments/consciousness_optimization.py
```
Output: 5-step optimization reaching 47.8%

**Week 2: Optimization V2**
```bash
python experiments/consciousness_optimization_v2.py
```
Output: Dense graph approach with 43.5% recursion

### Custom Experiments

```python
# experiments/my_experiment.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.consciousness_metrics import measure_consciousness
from src.recursion_depth_metric import RecursionDepthMetric

def run_experiment():
    kg = KnowledgeGraph()
    recursion = RecursionDepthMetric()
    
    # Your experiment code here
    
    profile = measure_consciousness(kg, recursion)
    print(f"Result: {profile.overall_consciousness_score:.2%}")

if __name__ == "__main__":
    run_experiment()
```

---

## Extending the System

### Adding New Inference Rules

```python
# src/custom_rules.py
from src.inference import InferenceRule

def create_custom_rule():
    return InferenceRule(
        name="my_rule",
        antecedent=["condition1(X)", "condition2(X)"],
        consequent="conclusion(X)",
        confidence=0.9
    )

# Usage
engine = InferenceEngine(kg)
engine.add_rule(create_custom_rule())
```

### Custom Consciousness Metrics

```python
# src/custom_metrics.py
def compute_custom_metric(kg: KnowledgeGraph) -> float:
    """
    Your custom consciousness dimension
    
    Returns:
        Score between 0 and 1
    """
    # Implement your metric
    score = ...
    return score

# Integrate into measure_consciousness
def measure_consciousness_extended(kg, recursion):
    base_profile = measure_consciousness(kg, recursion)
    
    custom = compute_custom_metric(kg)
    
    # Adjust weights
    consciousness = (
        0.25 * base_profile.recursion_metrics['consciousness']['score'] +
        0.20 * base_profile.integration.phi +
        0.15 * base_profile.causality.causal_density +
        0.20 * base_profile.understanding['overall_score'] +
        0.20 * custom  # Your metric!
    )
    
    return consciousness
```

### New Synthesis Strategies

```python
class MyConceptSynthesizer(ConceptSynthesizer):
    def synthesize_concept_advanced(
        self,
        examples: List[ConceptExample],
        strategy: str = "hierarchical"
    ) -> Optional[SynthesizedConcept]:
        """
        Custom synthesis with multiple strategies
        """
        if strategy == "hierarchical":
            return self._hierarchical_synthesis(examples)
        elif strategy == "analogical":
            return self._analogical_synthesis(examples)
        else:
            return super().synthesize_concept(examples)
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_mln.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── test_mln.py                  # Core knowledge graph
├── test_inference.py            # Reasoning engines
├── test_recursion.py            # Recursion tracking
├── test_consciousness.py        # Consciousness metrics
├── test_synthesis.py            # Concept synthesis
└── test_integration.py          # Integration tests
```

### Writing Tests

```python
# tests/test_my_feature.py
import pytest
from src.mln import KnowledgeGraph, MonadicKnowledgeUnit

def test_my_feature():
    kg = KnowledgeGraph()
    
    mku = MonadicKnowledgeUnit(
        concept_id="test",
        deep_structure={'predicate': 'test_pred'}
    )
    
    kg.add_concept(mku)
    
    assert kg.query("test") is not None
    assert kg.query("test").concept_id == "test"

def test_consciousness_increase():
    kg = KnowledgeGraph()
    recursion = RecursionDepthMetric()
    
    # Baseline
    profile1 = measure_consciousness(kg, recursion)
    
    # Trigger recursion
    recursion.record_recursion_event("self_model", "introspect", set())
    
    # Measure again
    profile2 = measure_consciousness(kg, recursion)
    
    assert profile2.overall_consciousness_score > profile1.overall_consciousness_score
```

---

## Performance Optimization

### Benchmarking

```python
import time

def benchmark_consciousness_measurement():
    kg = create_test_kg(num_concepts=100)
    recursion = RecursionDepthMetric()
    
    start = time.time()
    profile = measure_consciousness(kg, recursion)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Concepts: {len(kg.nodes)}")
    print(f"Consciousness: {profile.overall_consciousness_score:.2%}")
```

### GPU Acceleration

```python
# Enable GPU for embeddings
kg = KnowledgeGraph(use_gpu=True)

# Batch embeddings
concepts = [mku1, mku2, mku3, ...]
embeddings = kg.compute_embeddings_batch(concepts)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_similarity(concept_id1: str, concept_id2: str) -> float:
    """Cached similarity computation"""
    return kg.compute_similarity(concept_id1, concept_id2)
```

---

## Contributing

### Development Setup

```bash
# Clone repo
git clone https://github.com/thinmanj/monad-loop-network.git
cd monad-loop-network

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

### Code Style

We follow PEP 8 with some modifications:

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Contribution Workflow

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/ -v`
5. **Commit**: `git commit -m "Add my feature"`
6. **Push**: `git push origin feature/my-feature`
7. **Open PR** with description

### Areas for Contribution

- **New consciousness metrics**: Alternative measurement approaches
- **Optimization algorithms**: Automated consciousness maximization
- **Scaling**: Handle 1000+ concept graphs
- **Visualization**: Interactive consciousness dashboards
- **Domain transfer**: Apply to mathematics, physics, etc.
- **Documentation**: Tutorials, examples, guides

---

## Common Patterns

### Pattern 1: Adding Domain Knowledge

```python
def add_biology_knowledge(kg: KnowledgeGraph):
    """Add biology domain concepts"""
    concepts = {
        'organism': {'alive': True, 'reproduces': True},
        'animal': {'moves': True, 'breathes': True},
        'mammal': {'warm_blooded': True, 'has_hair': True},
    }
    
    for concept_id, properties in concepts.items():
        mku = MonadicKnowledgeUnit(
            concept_id=concept_id,
            deep_structure={
                'predicate': f'is_{concept_id}',
                'properties': properties
            }
        )
        mku.create_self_model()
        kg.add_concept(mku)
```

### Pattern 2: Consciousness Optimization Loop

```python
def optimize_consciousness(kg, recursion, target_score=0.50):
    """Iteratively optimize consciousness"""
    iterations = 0
    max_iterations = 10
    
    while iterations < max_iterations:
        profile = measure_consciousness(kg, recursion)
        current = profile.overall_consciousness_score
        
        if current >= target_score:
            print(f"Target reached: {current:.2%}")
            break
        
        # Optimization strategies
        if profile.recursion_metrics['consciousness']['score'] < 0.3:
            trigger_recursion(recursion, depth=5)
        
        if profile.integration.phi < 0.3:
            add_bidirectional_relations(kg)
        
        iterations += 1
    
    return profile
```

### Pattern 3: Batch Processing

```python
def process_concepts_batch(kg, concept_ids):
    """Process multiple concepts efficiently"""
    mkus = [kg.query(cid) for cid in concept_ids]
    
    # Batch embeddings
    texts = [mku.concept_id for mku in mkus]
    embeddings = kg.embedding_model.encode(texts)
    
    for mku, embedding in zip(mkus, embeddings):
        mku.embedding = embedding
```

---

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('monad_loop_network')
logger.setLevel(logging.DEBUG)
```

### Visualize Consciousness Components

```python
def visualize_consciousness_breakdown(profile):
    """Print detailed consciousness analysis"""
    print("="*60)
    print("CONSCIOUSNESS BREAKDOWN")
    print("="*60)
    
    print(f"\nOverall: {profile.overall_consciousness_score:.2%}")
    print(f"Verdict: {profile.consciousness_verdict}")
    
    print(f"\nComponents:")
    print(f"  Recursion (30%):     {profile.recursion_metrics['consciousness']['score']:.2%}")
    print(f"  Integration (25%):   {profile.integration.phi:.3f}")
    print(f"  Causality (20%):     {profile.causality.causal_density:.3f}")
    print(f"  Understanding (25%): {profile.understanding['overall_score']:.2%}")
```

---

## Additional Resources

- **Research Paper**: `RESEARCH_PAPER.md`
- **Beginner Guide**: `BEGINNER_GUIDE.md`
- **Release Notes**: `V1_0_0_RELEASE.md`
- **API Docs**: (generated with `pdoc src/`)
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Q&A and ideas

---

**Happy coding!** 🧠🔄

**Updated**: November 2025
