#!/usr/bin/env python3
"""
MLN API Server - Issue #32
REST API for Monad-Loop Network

Quick Win: Production-ready API for web UI and external integrations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

from src.mln import MonadicKnowledgeUnit, KnowledgeGraph, InferenceChain
from src.neurosymbolic import NeurosymbolicInterface
from src.analogical_reasoning import AnalogicalReasoning
from src.concept_synthesis import ConceptSynthesizer, ConceptExample
from src.inference_strategy_learner import InferenceStrategyLearner, QueryType
from src.strange_loop_optimizer import StrangeLoopOptimizer


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class ConceptCreate(BaseModel):
    """Request to create a new concept"""
    concept_id: str
    predicate: Optional[str] = None
    arguments: List[str] = []
    properties: Dict[str, Any] = {}
    constraints: List[str] = []


class QueryRequest(BaseModel):
    """Request to query the knowledge graph"""
    query: str
    use_nlp: bool = True
    max_depth: int = 10


class InferenceRequest(BaseModel):
    """Request for inference chain"""
    start_concept: str
    target_concept: str


class AnalogyRequest(BaseModel):
    """Request for analogical reasoning"""
    source_concept: str
    target_domain_concepts: List[str]


class ConceptSynthesisRequest(BaseModel):
    """Request to synthesize a new concept from examples"""
    concept_name: str
    examples: List[Dict[str, Any]]
    negative_examples: List[Dict[str, Any]] = []


class ConceptResponse(BaseModel):
    """Response containing concept details"""
    concept_id: str
    deep_structure: Dict[str, Any]
    relations: Dict[str, List[str]]
    relation_count: int


class QueryResponse(BaseModel):
    """Response to a query"""
    success: bool
    result: Optional[str] = None
    inference_chain: List[str] = []
    confidence: float = 0.0
    explanation: Optional[str] = None
    error: Optional[str] = None


class StatisticsResponse(BaseModel):
    """System statistics"""
    total_concepts: int
    total_relations: int
    inference_rules: int
    gpu_enabled: bool
    strategy_learner_stats: Dict[str, Any] = {}
    strange_loop_stats: Dict[str, Any] = {}


# ============================================================================
# API Server
# ============================================================================

app = FastAPI(
    title="Monad-Loop Network API",
    description="REST API for neurosymbolic AGI system with self-improvement",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global State (In production, use proper state management)
# ============================================================================

# Initialize core systems
kg = KnowledgeGraph(use_gpu=True)
neurosymbolic = NeurosymbolicInterface(kg)
analogical = AnalogicalReasoning(kg)
synthesizer = ConceptSynthesizer(kg)
strategy_learner = InferenceStrategyLearner()
loop_optimizer = StrangeLoopOptimizer()

# Add some default concepts for demo
def initialize_demo_knowledge():
    """Initialize with some demo concepts"""
    # Animals hierarchy
    animal = MonadicKnowledgeUnit(
        concept_id="animal",
        deep_structure={
            "predicate": "is_living_thing",
            "properties": {"breathes": True, "moves": True, "type": "organism"}
        }
    )
    kg.add_concept(animal)
    
    mammal = MonadicKnowledgeUnit(
        concept_id="mammal",
        deep_structure={
            "predicate": "is_animal",
            "properties": {"warm_blooded": True, "gives_birth": True, "has_hair": True}
        }
    )
    kg.add_concept(mammal)
    
    dog = MonadicKnowledgeUnit(
        concept_id="dog",
        deep_structure={
            "predicate": "is_mammal",
            "properties": {"domesticated": True, "barks": True, "loyal": True}
        }
    )
    kg.add_concept(dog)
    
    cat = MonadicKnowledgeUnit(
        concept_id="cat",
        deep_structure={
            "predicate": "is_mammal",
            "properties": {"domesticated": True, "meows": True, "independent": True}
        }
    )
    kg.add_concept(cat)

initialize_demo_knowledge()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - health check"""
    return {
        "status": "running",
        "version": "0.5.0",
        "system": "Monad-Loop Network",
        "phase": "4 Complete (Self-Improvement)",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "concepts": "/concepts",
            "query": "/query",
            "inference": "/inference",
            "analogy": "/analogy",
            "synthesize": "/synthesize",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "concepts": len(kg.nodes),
        "gpu_available": kg.use_gpu
    }


@app.get("/concepts", response_model=List[str])
async def list_concepts():
    """List all concept IDs"""
    return list(kg.nodes.keys())


@app.get("/concepts/{concept_id}", response_model=ConceptResponse)
async def get_concept(concept_id: str):
    """Get details for a specific concept"""
    if concept_id not in kg.nodes:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_id}' not found")
    
    mku = kg.nodes[concept_id]
    return ConceptResponse(
        concept_id=concept_id,
        deep_structure=mku.deep_structure,
        relations=mku.relations,
        relation_count=sum(len(v) for v in mku.relations.values())
    )


@app.post("/concepts", response_model=ConceptResponse)
async def create_concept(concept: ConceptCreate):
    """Create a new concept"""
    if concept.concept_id in kg.nodes:
        raise HTTPException(status_code=400, detail=f"Concept '{concept.concept_id}' already exists")
    
    mku = MonadicKnowledgeUnit(
        concept_id=concept.concept_id,
        deep_structure={
            "predicate": concept.predicate,
            "arguments": concept.arguments,
            "properties": concept.properties,
            "constraints": concept.constraints
        }
    )
    
    kg.add_concept(mku)
    
    return ConceptResponse(
        concept_id=concept.concept_id,
        deep_structure=mku.deep_structure,
        relations=mku.relations,
        relation_count=sum(len(v) for v in mku.relations.values())
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge graph (with optional NLP)"""
    try:
        if request.use_nlp:
            # Use neurosymbolic interface
            result = neurosymbolic.process_query(request.query)
            
            return QueryResponse(
                success=True,
                result=result.get("answer", "No answer found"),
                inference_chain=result.get("reasoning", []),
                confidence=result.get("confidence", 0.0),
                explanation=result.get("explanation")
            )
        else:
            # Direct symbolic query (expects "concept1 to concept2" format)
            parts = request.query.split(" to ")
            if len(parts) != 2:
                raise HTTPException(
                    status_code=400,
                    detail="Query format: 'concept1 to concept2'"
                )
            
            start, target = parts[0].strip(), parts[1].strip()
            chain = kg.query(start, target)
            
            if not chain.steps:
                return QueryResponse(
                    success=False,
                    error=f"No path found from '{start}' to '{target}'"
                )
            
            return QueryResponse(
                success=True,
                result=f"Path found: {len(chain.steps)} steps",
                inference_chain=[step.concept_id for step in chain.steps],
                confidence=1.0 if chain.is_valid() else 0.5,
                explanation=chain.explain()
            )
    
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e)
        )


@app.post("/inference")
async def inference_chain(request: InferenceRequest):
    """Get inference chain between two concepts"""
    if request.start_concept not in kg.nodes:
        raise HTTPException(status_code=404, detail=f"Start concept '{request.start_concept}' not found")
    
    if request.target_concept not in kg.nodes:
        raise HTTPException(status_code=404, detail=f"Target concept '{request.target_concept}' not found")
    
    chain = kg.query(request.start_concept, request.target_concept)
    
    return {
        "start": request.start_concept,
        "target": request.target_concept,
        "found": len(chain.steps) > 0,
        "steps": [step.concept_id for step in chain.steps],
        "valid": chain.is_valid(),
        "explanation": chain.explain()
    }


@app.post("/analogy")
async def find_analogy(request: AnalogyRequest):
    """Find analogies using structural similarity"""
    if request.source_concept not in kg.nodes:
        raise HTTPException(status_code=404, detail=f"Source concept '{request.source_concept}' not found")
    
    source_mku = kg.nodes[request.source_concept]
    source_structure = analogical.extract_structure(source_mku)
    
    # Find analogs in target domain
    target_mkus = [kg.nodes[cid] for cid in request.target_domain_concepts if cid in kg.nodes]
    
    if not target_mkus:
        raise HTTPException(status_code=400, detail="No valid target concepts found")
    
    analogs = []
    for target_mku in target_mkus:
        similarity = analogical.compute_similarity(source_structure, target_mku)
        if similarity > 0.3:  # Threshold
            analogs.append({
                "concept": target_mku.concept_id,
                "similarity": similarity
            })
    
    # Sort by similarity
    analogs.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "source": request.source_concept,
        "analogs": analogs[:5]  # Top 5
    }


@app.post("/synthesize")
async def synthesize_concept(request: ConceptSynthesisRequest):
    """Synthesize a new concept from examples (abductive learning)"""
    # Convert request examples to ConceptExample objects
    examples = [
        ConceptExample(
            concept_id=f"example_{i}",
            properties=ex.get("properties", {}),
            relations=ex.get("relations", {})
        )
        for i, ex in enumerate(request.examples)
    ]
    
    negative_examples = [
        ConceptExample(
            concept_id=f"negative_{i}",
            properties=ex.get("properties", {}),
            relations=ex.get("relations", {})
        )
        for i, ex in enumerate(request.negative_examples)
    ]
    
    # Synthesize concept
    synthesized = synthesizer.synthesize_concept(
        concept_name=request.concept_name,
        examples=examples,
        negative_examples=negative_examples
    )
    
    if not synthesized:
        raise HTTPException(status_code=400, detail="Failed to synthesize concept (not enough commonality)")
    
    return {
        "concept_name": synthesized.concept_name,
        "properties": synthesized.properties,
        "typical_properties": synthesized.typical_properties,
        "relations": synthesized.relations,
        "confidence": synthesized.confidence,
        "mku": {
            "concept_id": synthesized.mku.concept_id,
            "deep_structure": synthesized.mku.deep_structure
        }
    }


@app.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """Get system statistics"""
    return StatisticsResponse(
        total_concepts=len(kg.nodes),
        total_relations=sum(sum(len(v) for v in mku.relations.values()) for mku in kg.nodes.values()),
        inference_rules=len(kg.inference_rules),
        gpu_enabled=kg.use_gpu,
        strategy_learner_stats=strategy_learner.get_statistics(),
        strange_loop_stats=loop_optimizer.get_statistics()
    )


@app.get("/strategy/recommend/{query_type}")
async def recommend_strategy(query_type: str):
    """Recommend best inference strategy for a query type"""
    try:
        qtype = QueryType[query_type.upper()]
        strategy = strategy_learner.recommend_strategy(qtype)
        recommendations = strategy_learner.recommend_strategies(qtype, n=5)
        
        return {
            "query_type": query_type,
            "recommended": strategy.value,
            "top_strategies": [
                {"strategy": s.value, "score": score}
                for s, score in recommendations
            ]
        }
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query type. Valid types: {[qt.value for qt in QueryType]}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MONAD-LOOP NETWORK API SERVER")
    print("=" * 70)
    print(f"Starting server on http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Concepts loaded: {len(kg.nodes)}")
    print(f"GPU enabled: {kg.use_gpu}")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
