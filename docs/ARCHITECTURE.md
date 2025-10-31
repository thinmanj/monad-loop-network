# Architecture Guide

## System Overview

The Monad-Loop Network (MLN) is designed as a multi-layered architecture where each layer implements a specific philosophical principle:

```
Layer 1: Monadic Knowledge Units (Leibniz)
    ↓
Layer 2: Knowledge Graph (Relational structure)
    ↓
Layer 3: Inference Engine (Chomsky transformations)
    ↓
Layer 4: Strange Loop Processor (GEB meta-reasoning)
    ↓
Layer 5: Hybrid Intelligence System (Integration)
```

## Core Components

### 1. Monadic Knowledge Unit (MKU)

**Purpose**: Represent concepts as self-contained units with operational semantics.

**Key Properties**:
- `concept_id`: Unique identifier
- `deep_structure`: Internal representation of meaning
  - `predicate`: The core relation/property
  - `arguments`: Entities related by the predicate
  - `properties`: Attributes
  - `constraints`: Logical constraints
- `transformations`: Functions for surface realization
- `relations`: Connections to other MKUs
- `meta_model`: Self-representation for introspection

**Methods**:
- `reflect_universe(kg)`: Establish relations (Leibniz's pre-established harmony)
- `generate_surface_form(modality)`: Transform deep → surface (Chomsky)
- `create_self_model()`: Enable self-reference (GEB)

### 2. Knowledge Graph

**Purpose**: Store and query MKUs with operational semantics.

**Key Features**:
- Nodes are MKUs (not just vectors)
- Edges are typed relations (subtype, composition, association)
- Inference rules can be applied dynamically

**Methods**:
- `add_concept(mku)`: Add MKU and update relations
- `query(start, target)`: Find inference chain
- `add_inference_rule(rule)`: Add transformation
- `apply_inference(premise)`: Derive new knowledge

### 3. Inference Rules

**Purpose**: Implement Chomsky-style transformations for reasoning.

**Built-in Rules**:
- **TransitivityRule**: If A→B and B→C, then A→C
- **SubstitutionRule**: Replace equivalent concepts
- **Extensible**: Define custom rules by inheriting `InferenceRule`

### 4. Strange Loop Processor

**Purpose**: Implement GEB's self-reference and meta-reasoning.

**Capabilities**:
- **Self-modeling**: Creates `MetaKnowledgeGraph` (system models itself)
- **Introspection**: Examine reasoning traces
- **Inconsistency detection**: Find logical contradictions
- **Gödel sentences**: Construct self-referential statements

**Methods**:
- `create_strange_loop()`: Bootstrap self-reference
- `introspect(query)`: Analyze reasoning process
- `detect_inconsistency()`: Find contradictions
- `godel_sentence()`: Generate self-referential statement

### 5. Hybrid Intelligence System

**Purpose**: Integrate statistical (LLM) and symbolic reasoning.

**Current Implementation**:
- Pure symbolic reasoning (no LLM dependency)
- Designed for future LLM integration

**Future Architecture**:
```python
LLM (perception) → Entity extraction
    ↓
MKU mapping → Symbolic concepts
    ↓
Inference Engine → Logical reasoning
    ↓
Meta-validation → Verify inference chain
    ↓
Surface generation → Natural language response
```

## Data Flow

### Query Processing Pipeline

1. **Input**: Natural language question + concepts
2. **Mapping**: Extract/map to MKUs
3. **Inference**: Find reasoning chain using graph traversal + inference rules
4. **Validation**: Meta-reasoning checks validity
5. **Output**: Answer + complete inference trace

### Example Flow

```
Query: "Is a dog an animal?"

1. Map concepts:
   - start: MKU('dog')
   - target: MKU('animal')

2. Graph traversal:
   dog → [subtype] → mammal → [association] → animal

3. Inference chain:
   [dog, mammal, animal]

4. Meta-validation:
   ✓ Each step has valid relation
   ✓ Chain is consistent

5. Output:
   {
     'answer': 'yes',
     'inference_chain': 'dog → mammal → animal',
     'is_valid': true,
     'reasoning_trace': [...]
   }
```

## Key Algorithms

### Pre-established Harmony (Leibniz)

When a new MKU is added:
```python
def reflect_universe(self, kg):
    for other_mku in kg.nodes:
        similarity = compute_structural_similarity(self, other_mku)
        if similarity > threshold:
            relation_type = infer_relation_type(self, other_mku)
            self.relations[relation_type].add(other_mku.id)
```

**Complexity**: O(n) where n = number of existing concepts

### Inference Chain Discovery

BFS with inference rules:
```python
def query(start, target):
    queue = [(start, [])]
    visited = set()
    
    while queue:
        current, path = queue.pop(0)
        if current == target:
            return InferenceChain(path + [current])
        
        visited.add(current)
        
        # Follow relations
        for related in current.relations:
            if related not in visited:
                queue.append((related, path + [current]))
    
    return InferenceChain([])  # No path
```

**Complexity**: O(V + E) where V = vertices, E = edges

### Meta-reasoning Validation

```python
def is_valid(chain):
    for i in range(len(chain) - 1):
        current, next = chain[i], chain[i+1]
        if next.id not in current.get_all_relations():
            return False
    return True
```

**Complexity**: O(k) where k = chain length

## Extension Points

### 1. Custom MKU Types

```python
class DomainSpecificMKU(MonadicKnowledgeUnit):
    def __init__(self, concept_id, domain_data):
        super().__init__(concept_id)
        self.domain_data = domain_data
    
    def domain_specific_operation(self):
        # Custom behavior
        pass
```

### 2. Custom Inference Rules

```python
class MyCustomRule(InferenceRule):
    def can_apply(self, premise):
        # Check applicability
        return condition
    
    def apply(self, premise, kg):
        # Apply transformation
        return new_mku
```

### 3. Custom Surface Generators

```python
class CustomMKU(MonadicKnowledgeUnit):
    def generate_surface_form(self, modality):
        if modality == 'custom_format':
            return self._generate_custom()
        return super().generate_surface_form(modality)
```

## Performance Considerations

### Memory

- **MKU size**: O(p) where p = number of properties
- **Graph size**: O(n + e) where n = nodes, e = edges
- **Meta-graph overhead**: O(n) additional for self-model

### Computational

- **Add concept**: O(n) for relation establishment
- **Query**: O(V + E) graph traversal
- **Inference**: O(r × p) where r = rules, p = premises
- **Meta-reasoning**: O(k) chain validation

### Scalability

**Current limits**:
- Works well up to ~10,000 concepts
- Beyond that, consider:
  - Indexed relations (hash maps)
  - Lazy relation establishment
  - Distributed knowledge graphs

**Future optimizations**:
- Relation caching
- Incremental updates
- Parallel inference

## Testing Strategy

### Unit Tests

- Test each component in isolation
- Mock dependencies
- Focus on edge cases

### Integration Tests

- Test component interactions
- End-to-end query processing
- Meta-reasoning scenarios

### Property-Based Tests

- Invariants (e.g., valid chains always connect)
- Consistency checks
- Gödel sentence behavior

## Deployment

### Standalone

```bash
python -m src.mln
```

### As Library

```python
from src.mln import HybridIntelligenceSystem

system = HybridIntelligenceSystem()
# Use system...
```

### With LLM Integration (Future)

```python
from src.mln import HybridIntelligenceSystem
from transformers import AutoModel

llm = AutoModel.from_pretrained('model-name')
system = HybridIntelligenceSystem(llm=llm)
```

## Debugging

### Trace Reasoning

```python
result = system.query(...)
print(result['meta_analysis']['reasoning_trace'])
```

### Visualize Graph

```python
# Export to GraphML or similar
for node in system.kg.nodes.values():
    print(f"{node.concept_id}: {node.relations}")
```

### Introspection

```python
meta_analysis = system.slp.introspect("query")
print(json.dumps(meta_analysis, indent=2))
```

## Future Architecture

### Phase 1: Neurosymbolic (Current + LLM)
```
LLM → Extract entities → MKUs → Symbolic reasoning → Answer
```

### Phase 2: Self-Improvement
```
Feedback loop → Detect failures → Synthesize new concepts → Update graph
```

### Phase 3: Analogical Reasoning
```
Source domain → Extract structure → Map to target → New insights
```

### Phase 4: Consciousness Metrics
```
Strange loop depth → Integration measure → Φ (phi) metric
```
