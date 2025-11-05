#!/usr/bin/env python3
"""
Chomsky Surface Generation Demo

Demonstrates the separation of deep structure and surface structure:
- One deep structure (operational semantics)
- Multiple surface realizations (natural language variants)

Shows both:
1. Built-in transformational rules (works without any external dependencies)
2. Optional LLM-powered generation (richer, more diverse outputs)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mln import KnowledgeGraph, MonadicKnowledgeUnit
from src.surface_generator import create_surface_generator, SurfaceGenerationConfig


def demo_built_in():
    """Demo built-in surface generation (no LLM required)"""
    
    print("\n" + "=" * 70)
    print("PART 1: BUILT-IN SURFACE GENERATION")
    print("=" * 70)
    print("\nNo external dependencies - uses Chomsky's transformational rules")
    print()
    
    # Create knowledge graph
    kg = KnowledgeGraph(use_gpu=False)
    
    # Add a concept with rich deep structure
    dog = MonadicKnowledgeUnit(
        concept_id='dog',
        deep_structure={
            'predicate': 'mammal',
            'properties': {
                'domesticated': True,
                'social': True,
                'barks': True,
                'warm_blooded': True,
                'lifespan_years': '10-13'
            }
        }
    )
    dog.create_self_model()
    kg.add_concept(dog)
    
    # Add related concepts
    mammal = MonadicKnowledgeUnit(
        concept_id='mammal',
        deep_structure={'predicate': 'animal_class', 'properties': {'warm_blooded': True}}
    )
    kg.add_concept(mammal)
    
    # Create surface generator (no LLM)
    gen = create_surface_generator()
    
    # Prepare MKU data
    mku_data = {
        'concept_id': 'dog',
        'predicate': dog.deep_structure['predicate'],
        'properties': dog.deep_structure['properties'],
        'relations': dog.relations
    }
    
    print("Deep Structure (MKU):")
    print(f"  Concept: {mku_data['concept_id']}")
    print(f"  Predicate: {mku_data['predicate']}")
    print(f"  Properties: {list(mku_data['properties'].keys())}")
    print(f"  Relations: {list(mku_data['relations'].keys())}")
    print()
    
    print("Surface Realizations (same deep structure, different forms):")
    print()
    
    styles = ['conversational', 'technical', 'educational', 'poetic']
    for style in styles:
        surface = gen.generate_from_mku(mku_data, style=style)
        print(f"  [{style.upper()}]")
        print(f"  {surface}")
        print()
    
    print("Multiple Conversational Variants:")
    print("(Demonstrating: one deep structure → many surface forms)")
    print()
    variants = gen.generate_multiple_variants(mku_data, num_variants=3)
    for i, variant in enumerate(variants, 1):
        print(f"  {i}. {variant}")
    print()


def demo_with_llm():
    """Demo LLM-powered surface generation (optional, richer outputs)"""
    
    print("\n" + "=" * 70)
    print("PART 2: LLM-POWERED SURFACE GENERATION (Optional)")
    print("=" * 70)
    print("\nEnhanced generation with OpenAI/Anthropic/Ollama")
    print()
    
    # Check if API keys are available
    import os
    has_openai = os.getenv('OPENAI_API_KEY') is not None
    has_anthropic = os.getenv('ANTHROPIC_API_KEY') is not None
    
    if not has_openai and not has_anthropic:
        print("⚠️  No API keys found (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        print("   This is optional - the system works great without it!")
        print()
        print("To enable LLM-powered generation:")
        print("  export OPENAI_API_KEY='your-key'  # For OpenAI")
        print("  export ANTHROPIC_API_KEY='your-key'  # For Claude")
        print("  # Or use local Ollama (no API key needed)")
        print()
        print("Skipping LLM demo...\n")
        return
    
    # Try to create LLM-powered generator
    try:
        if has_openai:
            print("✓ Using OpenAI for surface generation")
            config = SurfaceGenerationConfig(
                provider='openai',
                model='gpt-3.5-turbo',
                temperature=0.8
            )
        else:
            print("✓ Using Anthropic Claude for surface generation")
            config = SurfaceGenerationConfig(
                provider='anthropic',
                model='claude-3-haiku-20240307',
                temperature=0.8
            )
        
        gen = create_surface_generator(provider=config.provider)
        
        # Same deep structure as before
        mku_data = {
            'concept_id': 'dog',
            'predicate': 'mammal',
            'properties': {
                'domesticated': True,
                'social': True,
                'barks': True,
                'warm_blooded': True,
                'lifespan_years': '10-13'
            },
            'relations': {
                'subtype': ['mammal', 'animal'],
                'similar_to': ['wolf', 'cat']
            }
        }
        
        print()
        print("Deep Structure → LLM → Rich Natural Language:")
        print()
        
        for style in ['conversational', 'technical', 'educational']:
            print(f"  [{style.upper()}]")
            surface = gen.generate_from_mku(mku_data, style=style)
            print(f"  {surface}")
            print()
        
        print("✓ LLM generates richer, more varied surface forms!")
        print()
        
    except Exception as e:
        print(f"⚠️  LLM generation failed: {e}")
        print("   Falling back to built-in generation (which works great!)")
        print()


def demo_reasoning_chains():
    """Demo surface generation for inference chains"""
    
    print("\n" + "=" * 70)
    print("PART 3: REASONING CHAIN GENERATION")
    print("=" * 70)
    print("\nGenerate natural language explanations of inference chains")
    print()
    
    gen = create_surface_generator()
    
    # Example reasoning chain
    chain = [
        {'concept_id': 'dog', 'predicate': 'mammal'},
        {'concept_id': 'mammal', 'predicate': 'animal'},
        {'concept_id': 'animal', 'predicate': 'living_thing'}
    ]
    
    conclusion = "Therefore, dog is a living_thing"
    
    explanation = gen.generate_with_reasoning_chain(chain, conclusion)
    
    print("Inference Chain:")
    for i, step in enumerate(chain, 1):
        print(f"  {i}. {step['concept_id']}: {step['predicate']}")
    print()
    print("Natural Language Explanation:")
    print(f"  {explanation}")
    print()


def demo_chatbot_integration():
    """Demo how surface generation enhances the chatbot"""
    
    print("\n" + "=" * 70)
    print("PART 4: CHATBOT INTEGRATION")
    print("=" * 70)
    print("\nThe chatbot now uses Chomsky transformational grammar")
    print()
    
    from src.chatbot import ConsciousnessChatbot
    
    # Create bot (uses built-in surface generation by default)
    bot = ConsciousnessChatbot()
    
    print("Ask: 'What is a dog?'")
    print()
    
    response = bot.ask("What is a dog?")
    
    print("Answer:")
    print(f"  {response.answer}")
    print()
    print("How it works:")
    print("  1. Retrieve MKU (deep structure)")
    print("  2. Apply transformational grammar rules")
    print("  3. Generate surface form (natural language)")
    print()
    print("Reasoning:")
    for step in response.reasoning[-2:]:
        print(f"  - {step}")
    print()


def main():
    """Run all demos"""
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "CHOMSKY SURFACE GENERATION DEMO" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("This demo shows the separation of deep and surface structure:")
    print("  • Deep Structure: Operational semantics (MKU)")
    print("  • Surface Structure: Natural language (generated)")
    print("  • Transformational Grammar: Rules that map deep → surface")
    print()
    print("Key insight: ONE deep structure → MANY surface forms")
    print()
    
    # Run demos
    demo_built_in()
    demo_reasoning_chains()
    demo_chatbot_integration()
    demo_with_llm()  # Optional - only if API keys available
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Deep structure (MKU) contains operational semantics")
    print("✓ Surface generation creates diverse natural language")
    print("✓ Works great with built-in rules (no dependencies)")
    print("✓ Optionally enhanced with LLM for richer outputs")
    print("✓ Integrated into chatbot for better explanations")
    print()
    print("This is Chomsky's transformational grammar in action!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
