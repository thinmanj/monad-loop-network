# Monad-Loop Network (MLN)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A self-referential knowledge system combining Gödel-Escher-Bach's strange loops, Chomsky's universal grammar, and Leibniz's monads for structural, explainable AI.

## 🧠 Philosophy

Current LLMs are statistical pattern matchers—they correlate tokens without genuine understanding. MLN represents a different paradigm:

- **Structural Knowledge**: Concepts have operational semantics, not just vector embeddings
- **Explainable Reasoning**: Complete inference chains, not black-box predictions
- **Self-Reference**: Systems that can reason about their own reasoning (meta-cognition)
- **Compositionality**: Deep structures transform into multiple surface realizations

## 🎯 Key Concepts

### 1. Monadic Knowledge Units (Leibniz)
Self-contained concepts that "reflect the universe" from their perspective. Each monad:
- Contains deep structure (meaning)
- Establishes relations automatically (pre-established harmony)
- Has operational semantics (can execute transformations)

### 2. Deep Structure ↔ Surface Structure (Chomsky)
Meaning exists at the deep level. Multiple surface forms (text, code, logic) are isomorphic projections:
```
Deep Structure: IS_A(dog, mammal)
  ↓
Surface Forms:
  - Text:  "A dog is a mammal"
  - Logic: ∀x: dog(x) → mammal(x)
  - Code:  class Dog(Mammal): pass
```

### 3. Strange Loops (Gödel-Escher-Bach)
Self-referential systems create consciousness and meaning. MLN implements:
- Meta-knowledge graphs (system models itself)
- Introspection (examine own reasoning)
- Gödel sentences (expose system limits)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/monad-loop-network.git
cd monad-loop-network
pip install -r requirements.txt
```

### Basic Usage

```python
from src.mln import HybridIntelligenceSystem

# Create system
system = HybridIntelligenceSystem()

# Add knowledge
system.add_knowledge('dog', {
    'predicate': 'is_a',
    'arguments': ['mammal'],
    'properties': {'domesticated': True}
})

# Query with explainable reasoning
result = system.query(
    question="Is a dog a mammal?",
    start_concept='dog',
    target_concept='mammal'
)

print(result['inference_chain'])  # Shows reasoning steps
print(result['is_valid'])          # Meta-validation
```

### Run Demo

```bash
python examples/demo.py
```

## 📊 Comparison: MLN vs. Statistical LLMs

| Aspect | Statistical LLMs | MLN System |
|--------|------------------|------------|
| **Reasoning** | Pattern matching | Logical inference with trace |
| **Explainability** | Opaque | Full derivation available |
| **Learning** | Weight adjustment | Structural concept formation |
| **Self-awareness** | None | Meta-reasoning capability |
| **Knowledge** | Implicit (weights) | Explicit (structured) |
| **Compositionality** | Weak | Strong (Chomsky-style) |
| **Consistency** | Statistical | Logically enforced |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│   Hybrid Intelligence System            │
├─────────────────────────────────────────┤
│  ┌────────────┐      ┌───────────────┐  │
│  │ LLM Layer  │      │ Symbolic      │  │
│  │ (Perception)│ ───▶ │ Reasoning     │  │
│  └────────────┘      └───────────────┘  │
│         │                    │           │
│         ▼                    ▼           │
│  ┌─────────────────────────────────┐    │
│  │   Knowledge Graph (MKUs)        │    │
│  │   - Operational semantics       │    │
│  │   - Pre-established harmony     │    │
│  └─────────────────────────────────┘    │
│         │                                │
│         ▼                                │
│  ┌─────────────────────────────────┐    │
│  │   Strange Loop Processor        │    │
│  │   - Meta-reasoning              │    │
│  │   - Self-introspection          │    │
│  │   - Inconsistency detection     │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## 📚 Use Cases

### 1. Medical Diagnosis
- **Deep structure**: Causal disease mechanisms
- **Surface structure**: Observable symptoms
- **Meta-reasoning**: "Why did I diagnose X?" → traceable inference

### 2. Code Understanding
- **Deep structure**: Computational semantics
- **Surface structure**: Syntax in various languages
- **Self-reference**: System reasons about its own code generation

### 3. Scientific Discovery
- **Abductive reasoning**: Form new hypotheses (new MKUs)
- **Strange loops**: "What experiments would validate my reasoning?"

## 🔬 Research Directions

1. **Neurosymbolic Integration**: LLM perception + symbolic inference
2. **Analogical Reasoning**: Structural isomorphism between domains
3. **Self-Improvement**: System learns by structural concept formation
4. **Consciousness Metrics**: Measure "loop complexity" (IIT-inspired)

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Deep dive into system design
- [Philosophical Foundations](docs/PHILOSOPHY.md) - GEB, Chomsky, Leibniz
- [API Reference](docs/API.md) - Complete API documentation
- [Examples](examples/) - Practical use cases

## 🤝 Contributing

Contributions welcome! This is an experimental research project exploring alternatives to pure statistical AI.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Douglas Hofstadter** - *Gödel, Escher, Bach* (strange loops, consciousness)
- **Noam Chomsky** - Universal grammar, deep structure
- **Gottfried Leibniz** - Monadology, pre-established harmony
- **Richard Feynman** - Inspiration for questioning fundamental constants

## 📞 Contact

For questions, discussions, or collaborations, open an issue or reach out!

## 🗺️ Roadmap

- [x] Core MKU system
- [x] Knowledge graph with operational semantics
- [x] Strange loop processor (meta-reasoning)
- [ ] Integration with existing LLMs (hybrid system)
- [ ] Analogical reasoning engine
- [ ] Self-improvement mechanisms
- [ ] Large-scale knowledge acquisition
- [ ] Consciousness metrics

---

*"The answer to life, the universe, and everything is not 42—it's understanding the structure of the question itself."*
