# API Reference

## Core Classes

### MonadicKnowledgeUnit

A self-contained concept with operational semantics.

```python
class MonadicKnowledgeUnit:
    def __init__(
        self,
        concept_id: str,
        deep_structure: Dict[str, Any] = None,
        transformations: List[Callable] = None,
        relations: Dict[str, Set[str]] = None,
        meta_model: Optional[MetaRepresentation] = None
    )
```

**Parameters**:
- `concept_id` (str): Unique identifier for the concept
- `deep_structure` (dict): Internal meaning representation
  - `predicate`: Core relation/property
  - `arguments`: Related entities
  - `properties`: Attributes
  - `constraints`: Logical constraints
- `transformations` (list): Functions for surface generation
- `relations` (dict): Connections to other concepts
- `meta_model` (MetaRepresentation): Self-representation

**Methods**:

#### `reflect_universe(knowledge_graph: KnowledgeGraph) -> None`
Establish relations with other concepts based on structural similarity (Leibniz's pre-established harmony).

#### `generate_surface_form(modality: str = 'text') -> str`
Transform deep structure into surface representation.

**Parameters**:
- `modality` (str): Output format ('text', 'logic', 'code')

**Returns**: String representation in specified modality

#### `create_self_model() -> MetaRepresentation`
Create a meta-representation for self-reference and introspection.

**Returns**: MetaRepresentation object

---

### KnowledgeGraph

Graph structure storing MKUs with operational semantics.

```python
class KnowledgeGraph:
    def __init__(self)
```

**Attributes**:
- `nodes` (Dict[str, MonadicKnowledgeUnit]): Concept storage
- `inference_rules` (List[InferenceRule]): Transformation rules

**Methods**:

#### `add_concept(mku: MonadicKnowledgeUnit) -> None`
Add a concept and establish relations with existing concepts.

**Parameters**:
- `mku` (MonadicKnowledgeUnit): Concept to add

#### `query(start_id: str, target_id: str) -> InferenceChain`
Find reasoning chain from start to target concept.

**Parameters**:
- `start_id` (str): Starting concept ID
- `target_id` (str): Target concept ID

**Returns**: InferenceChain with reasoning steps

#### `add_inference_rule(rule: InferenceRule) -> None`
Add a transformation rule for reasoning.

**Parameters**:
- `rule` (InferenceRule): Inference rule to add

#### `apply_inference(premise: MonadicKnowledgeUnit) -> List[MonadicKnowledgeUnit]`
Apply inference rules to derive new knowledge.

**Parameters**:
- `premise` (MonadicKnowledgeUnit): Starting concept

**Returns**: List of derived concepts

---

### InferenceChain

Represents a reasoning chain with validation.

```python
@dataclass
class InferenceChain:
    steps: List[MonadicKnowledgeUnit]
```

**Methods**:

#### `explain() -> str`
Generate human-readable explanation of reasoning.

**Returns**: String explanation of inference steps

#### `is_valid() -> bool`
Validate logical consistency of the chain.

**Returns**: True if chain is valid, False otherwise

---

### InferenceRule (Abstract)

Base class for inference rules.

```python
class InferenceRule(ABC):
    @abstractmethod
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        pass
    
    @abstractmethod
    def apply(
        self,
        premise: MonadicKnowledgeUnit,
        kg: KnowledgeGraph
    ) -> Optional[MonadicKnowledgeUnit]:
        pass
```

**Built-in Rules**:

#### TransitivityRule
Implements transitive closure: If A→B and B→C, then A→C

#### SubstitutionRule
Substitutes equivalent concepts

---

### StrangeLoopProcessor

Implements self-reference and meta-reasoning.

```python
class StrangeLoopProcessor:
    def __init__(self, knowledge_graph: KnowledgeGraph)
```

**Attributes**:
- `kg` (KnowledgeGraph): Target knowledge graph
- `meta_kg` (MetaKnowledgeGraph): Meta-representation
- `reasoning_trace` (List[str]): Trace of reasoning steps

**Methods**:

#### `create_strange_loop() -> None`
Create self-referential structure (system models itself).

#### `introspect(query: str) -> Dict`
Examine own reasoning process.

**Parameters**:
- `query` (str): Query to introspect

**Returns**: Dictionary with reasoning trace and graph state

#### `detect_inconsistency() -> List[str]`
Find logical inconsistencies in the knowledge graph.

**Returns**: List of inconsistency descriptions

#### `godel_sentence() -> str`
Generate self-referential statement (Gödel-style).

**Returns**: Self-referential statement string

---

### HybridIntelligenceSystem

Main interface combining all components.

```python
class HybridIntelligenceSystem:
    def __init__(self)
```

**Attributes**:
- `kg` (KnowledgeGraph): Knowledge graph
- `slp` (StrangeLoopProcessor): Meta-reasoning processor

**Methods**:

#### `add_knowledge(concept_id: str, deep_structure: Dict) -> None`
Add new knowledge to the system.

**Parameters**:
- `concept_id` (str): Concept identifier
- `deep_structure` (dict): Deep structure representation

**Example**:
```python
system.add_knowledge('dog', {
    'predicate': 'is_a',
    'arguments': ['mammal'],
    'properties': {'domesticated': True}
})
```

#### `query(question: str, start_concept: str, target_concept: str) -> Dict`
Process query with explainable reasoning.

**Parameters**:
- `question` (str): Natural language question
- `start_concept` (str): Starting concept ID
- `target_concept` (str): Target concept ID

**Returns**: Dictionary with:
- `question` (str): Original question
- `inference_chain` (str): Reasoning explanation
- `is_valid` (bool): Validity of inference
- `additional_inferences` (list): Derived concepts
- `meta_analysis` (dict): Meta-reasoning information

**Example**:
```python
result = system.query(
    "Is a dog a mammal?",
    start_concept='dog',
    target_concept='mammal'
)
print(result['inference_chain'])
```

#### `explain_reasoning() -> str`
Generate human-readable explanation of reasoning process.

**Returns**: JSON string with reasoning trace

#### `detect_inconsistencies() -> List[str]`
Find logical inconsistencies in knowledge.

**Returns**: List of inconsistency descriptions

---

## Usage Examples

### Basic Usage

```python
from src.mln import HybridIntelligenceSystem

# Initialize system
system = HybridIntelligenceSystem()

# Add knowledge
system.add_knowledge('mammal', {
    'predicate': 'is_alive',
    'arguments': ['warm_blooded'],
    'properties': {'breathes': 'air'}
})

system.add_knowledge('dog', {
    'predicate': 'is_a',
    'arguments': ['mammal'],
    'properties': {'domesticated': True}
})

# Query
result = system.query(
    "Is a dog alive?",
    start_concept='dog',
    target_concept='mammal'
)

print(result['inference_chain'])
print(f"Valid: {result['is_valid']}")
```

### Advanced: Custom Inference Rule

```python
from src.mln import InferenceRule, MonadicKnowledgeUnit, KnowledgeGraph

class MyRule(InferenceRule):
    def can_apply(self, premise: MonadicKnowledgeUnit) -> bool:
        return 'custom_property' in premise.deep_structure
    
    def apply(self, premise: MonadicKnowledgeUnit, kg: KnowledgeGraph):
        return MonadicKnowledgeUnit(
            concept_id=f"derived_{premise.concept_id}",
            deep_structure={'derived': True}
        )

# Add to system
system = HybridIntelligenceSystem()
system.kg.add_inference_rule(MyRule())
```

### Advanced: Meta-reasoning

```python
# Introspect reasoning
meta_info = system.slp.introspect("my query")
print(f"Concepts: {meta_info['graph_state']['num_concepts']}")
print(f"Trace: {meta_info['reasoning_trace']}")

# Detect inconsistencies
inconsistencies = system.detect_inconsistencies()
for inc in inconsistencies:
    print(f"Found: {inc}")

# Generate Gödel sentence
godel = system.slp.godel_sentence()
print(godel)
```

### Advanced: Surface Realizations

```python
# Get concept
dog = system.kg.nodes['dog']

# Generate different surface forms
print(dog.generate_surface_form('text'))   # Natural language
print(dog.generate_surface_form('logic'))  # Logical form
print(dog.generate_surface_form('code'))   # Code representation
```

---

## Type Definitions

### Deep Structure

```python
{
    'predicate': str,           # Core relation
    'arguments': List[str],     # Related entities
    'properties': Dict[str, Any],  # Attributes
    'constraints': List[str]    # Logical constraints
}
```

### Query Result

```python
{
    'question': str,
    'inference_chain': str,
    'is_valid': bool,
    'additional_inferences': List[str],
    'meta_analysis': {
        'query': str,
        'reasoning_trace': List[str],
        'graph_state': {
            'num_concepts': int,
            'num_rules': int
        },
        'meta_analysis': Dict  # If meta_kg exists
    }
}
```

---

## Error Handling

The system currently uses Python's built-in exceptions:

- `KeyError`: Concept not found in graph
- `ValueError`: Invalid deep structure or parameters
- `TypeError`: Incorrect type for parameters

**Example**:
```python
try:
    result = system.query("Q", "unknown", "concept")
except KeyError:
    print("Concept not found")
```

---

## Performance Tips

1. **Batch concept addition**: Add multiple concepts before querying
2. **Limit relation depth**: Use shallow hierarchies for faster queries
3. **Cache results**: Store frequently used inference chains
4. **Selective meta-reasoning**: Only introspect when needed

---

## Future API Extensions

- `system.learn_from_example(example)`: Abductive learning
- `system.find_analogy(source, target)`: Analogical reasoning
- `system.measure_consciousness()`: Consciousness metrics
- `system.integrate_llm(llm_model)`: LLM integration
