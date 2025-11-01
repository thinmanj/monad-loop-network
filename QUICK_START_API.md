# Quick Start: API Server & Web UI

**Issues #32-33: Quick Wins Complete!**

---

## 🚀 Start the API Server

```bash
# Install FastAPI and uvicorn (if not already installed)
pip install fastapi uvicorn pydantic

# Start the server
python api_server.py
```

Server will start on `http://localhost:8000`

**API Documentation**: http://localhost:8000/docs (interactive Swagger UI)

---

## 🌐 Launch the Web UI

1. Start the API server first (see above)
2. Open `web_ui/index.html` in your browser

```bash
# Option 1: Open directly
open web_ui/index.html

# Option 2: Serve with Python (recommended for CORS)
python -m http.server 8080 --directory web_ui
# Then open http://localhost:8080
```

---

## 📡 API Endpoints

### Core Endpoints

**GET /**
- Health check and API info

**GET /health**
- System health status

**GET /concepts**
- List all concept IDs

**GET /concepts/{concept_id}**
- Get concept details

**POST /concepts**
- Create new concept

**POST /query**
- Query the knowledge graph
```json
{
  "query": "dog to animal",
  "use_nlp": false,
  "max_depth": 10
}
```

**POST /inference**
- Get inference chain between concepts
```json
{
  "start_concept": "dog",
  "target_concept": "animal"
}
```

**POST /analogy**
- Find analogies using structural similarity
```json
{
  "source_concept": "dog",
  "target_domain_concepts": ["cat", "bird", "fish"]
}
```

**POST /synthesize**
- Synthesize new concept from examples (abductive learning)
```json
{
  "concept_name": "mammal",
  "examples": [
    {"properties": {"warm_blooded": true, "gives_birth": true}},
    {"properties": {"warm_blooded": true, "gives_birth": true}}
  ]
}
```

**GET /stats**
- System statistics (concepts, relations, GPU status, learner stats)

**GET /strategy/recommend/{query_type}**
- Recommend best inference strategy
- Query types: `classification`, `analogy`, `explanation`, `prediction`, etc.

---

## 🎨 Web UI Features

### 1. Query Interface
- Natural language or symbolic queries
- Visual inference chain display
- Confidence scores
- Explanations

### 2. Knowledge Base Browser
- **Concepts Tab**: Browse all concepts
- **Statistics Tab**: View system metrics
  - Total concepts
  - Total relations
  - Inference rules
  - GPU status

### 3. Advanced Tools

**Inference Chain**
- Find reasoning path between two concepts
- Visual step-by-step display

**Analogical Reasoning**
- Find structural analogies
- Similarity scores

**Concept Synthesis**
- Create new concepts from examples
- Abductive learning

---

## 💡 Example Workflows

### Workflow 1: Basic Query

1. **Start API server**: `python api_server.py`
2. **Open Web UI**: `web_ui/index.html`
3. **Query**: Enter "dog to animal"
4. **Result**: See inference chain: dog → mammal → animal

### Workflow 2: Find Analogies

1. **Navigate to**: Advanced Tools → Analogy tab
2. **Source**: "dog"
3. **Targets**: "cat, bird, fish"
4. **Result**: Shows similarity scores for each analog

### Workflow 3: Synthesize Concept

1. **Navigate to**: Advanced Tools → Synthesize tab
2. **Name**: "new_mammal"
3. **Click**: Synthesize from Examples
4. **Result**: System creates concept with inferred properties

### Workflow 4: API Integration

```python
import requests

# Query endpoint
response = requests.post('http://localhost:8000/query', json={
    "query": "dog to animal",
    "use_nlp": False
})

result = response.json()
print(result['inference_chain'])
# Output: ['dog', 'mammal', 'animal']
```

---

## 🔧 Configuration

### API Server

Edit `api_server.py` to configure:
- `host` and `port` (default: 0.0.0.0:8000)
- CORS origins (default: all - `["*"]`)
- GPU usage (default: enabled)
- Demo knowledge (animal hierarchy pre-loaded)

### Web UI

Edit `web_ui/index.html`:
- `API_BASE` constant (line 333)
- Default: `http://localhost:8000`
- Change if API runs on different host/port

---

## 🎯 Demo Knowledge Pre-loaded

The API server starts with demo concepts:

- **animal** (living thing)
- **mammal** (is animal, warm-blooded, gives birth)
- **dog** (is mammal, domesticated, barks)
- **cat** (is mammal, domesticated, meows)

Try these queries:
- `dog to animal`
- `cat to animal`
- `dog` (source) → `cat` (analogy target)

---

## 🐛 Troubleshooting

### "Connection refused"
- Make sure API server is running: `python api_server.py`
- Check server output for errors
- Verify port 8000 is not in use

### "CORS error"
- Serve Web UI via HTTP server (not file://)
- Use: `python -m http.server 8080 --directory web_ui`

### "Module not found"
- Install dependencies: `pip install fastapi uvicorn pydantic`
- Make sure you're in the project root directory

### "No concepts found"
- API server should auto-initialize demo concepts
- Try creating a concept via API docs: http://localhost:8000/docs

---

## 📦 Next Steps

1. **Add more concepts** via API or Web UI
2. **Integrate with external ontologies** (Phase 2 features)
3. **Build custom queries** using the API
4. **Extend Web UI** with graph visualization

---

## 🎉 Quick Wins Complete!

✅ **Issue #32**: REST API with FastAPI
- 10+ endpoints
- Full Phase 4 feature coverage
- Interactive API docs
- CORS enabled for web UI

✅ **Issue #33**: Web UI
- Modern, responsive design
- Query interface with visual inference chains
- Concept browser with statistics
- Advanced tools (inference, analogy, synthesis)
- Real-time status updates

**Next**: Phase 5 - Consciousness Metrics (tomorrow!)
