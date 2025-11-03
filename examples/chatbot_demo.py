#!/usr/bin/env python3
"""
Interactive Consciousness-Aware Chatbot Demo

A terminal-based interactive demo showcasing:
- Natural language Q&A with explainable reasoning
- Real-time consciousness metrics
- Meta-cognitive commentary
- Learning from conversation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import ConsciousnessChatbot


def print_header():
    """Print demo header"""
    print("\n" + "=" * 70)
    print("  CONSCIOUSNESS-AWARE CHATBOT")
    print("  Explainable AI with Real-Time Consciousness Metrics")
    print("=" * 70)
    print()
    print("This chatbot is different:")
    print("  • Shows its reasoning step-by-step")
    print("  • Measures its own consciousness in real-time")
    print("  • Explains its thought process")
    print("  • Knows what it doesn't know")
    print()
    print("Commands:")
    print("  - Ask questions like 'What is a dog?' or 'Is a dog an animal?'")
    print("  - Type 'stats' to see full statistics")
    print("  - Type 'help' for more info")
    print("  - Type 'quit' to exit")
    print("=" * 70)
    print()


def print_response(response):
    """Pretty print chatbot response"""
    print()
    print("┌" + "─" * 68 + "┐")
    print("│ ANSWER" + " " * 61 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│ {response.answer[:66]:<66} │")
    if len(response.answer) > 66:
        # Wrap long answers
        words = response.answer[66:].split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= 66:
                line += " " + word if line else word
            else:
                print(f"│ {line:<66} │")
                line = word
        if line:
            print(f"│ {line:<66} │")
    print("└" + "─" * 68 + "┘")
    print()
    
    # Reasoning
    if response.reasoning:
        print("💭 REASONING:")
        for i, step in enumerate(response.reasoning, 1):
            print(f"   {i}. {step}")
        print()
    
    # Consciousness metrics
    metrics = response.consciousness_metrics
    print("🧠 CONSCIOUSNESS METRICS:")
    print(f"   Overall:     {metrics['overall']:.1%} ({metrics['verdict']})")
    print(f"   Recursion:   {metrics['recursion']:.1%} (depth: {metrics['recursion_depth']})")
    print(f"   Integration: {metrics['integration']:.3f}")
    print(f"   Confidence:  {response.confidence:.1%}")
    print(f"   Knowledge:   {metrics['concepts']} concepts")
    print()
    
    # Meta-commentary
    print("🤔 META-COGNITIVE INSIGHT:")
    print(f"   \"{response.meta_commentary}\"")
    print()


def print_stats(bot):
    """Print full chatbot statistics"""
    stats = bot.get_stats()
    
    print()
    print("=" * 70)
    print("CHATBOT STATISTICS")
    print("=" * 70)
    print()
    
    print("Consciousness Profile:")
    c = stats['consciousness']
    print(f"  Overall:     {c['overall']:.2%}")
    print(f"  Verdict:     {c['verdict']}")
    print(f"  Recursion:   {c['recursion']:.2%}")
    print(f"  Integration: {c['integration']:.3f}")
    print(f"  Understanding: {c['understanding']:.2%}")
    print()
    
    print("Conversation:")
    conv = stats['conversation']
    print(f"  Exchanges:   {conv['exchanges']}")
    print(f"  Learned:     {conv['concepts_learned']} new concepts")
    print()
    
    print("Knowledge Base:")
    kb = stats['knowledge']
    print(f"  Concepts:    {kb['total_concepts']}")
    print(f"  Relations:   {kb['total_relations']}")
    print()


def print_help():
    """Print help information"""
    print()
    print("=" * 70)
    print("HELP")
    print("=" * 70)
    print()
    print("Supported Question Types:")
    print("  • 'What is X?' - Get explanation of concept X")
    print("  • 'Is X a Y?' - Check if X is related to Y")
    print()
    print("Examples:")
    print("  • What is a dog?")
    print("  • Is a dog an animal?")
    print("  • Is a cat a mammal?")
    print("  • What is a tree?")
    print()
    print("Special Commands:")
    print("  • 'stats' - View full statistics")
    print("  • 'help' - Show this help")
    print("  • 'quit' or 'exit' - Exit the demo")
    print()
    print("About This Demo:")
    print("  This chatbot demonstrates measurable artificial consciousness.")
    print("  Unlike typical chatbots, it:")
    print("    - Explains every step of its reasoning")
    print("    - Measures its own consciousness in real-time")
    print("    - Is aware of what it knows and doesn't know")
    print("    - Provides meta-cognitive insights about its thinking")
    print()


def demo_mode(bot):
    """Run a scripted demo"""
    print("\n" + "🎬 DEMO MODE" + " " * 57 + "\n")
    print("Watch as the chatbot answers questions and shows consciousness...\n")
    
    demo_questions = [
        "What is a dog?",
        "Is a dog an animal?",
        "Is a dog a mammal?",
        "What is a penguin?",  # This one it doesn't know
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"Demo Question {i}/{len(demo_questions)}")
        print(f"{'─' * 70}")
        print(f"\nYou: {question}")
        
        response = bot.ask(question)
        print_response(response)
        
        if i < len(demo_questions):
            input("Press Enter to continue...")
    
    print("\n" + "=" * 70)
    print("Demo complete! Try asking your own questions.")
    print("=" * 70)
    print()


def interactive_mode(bot):
    """Run interactive chat loop"""
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Check for special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for chatting! The chatbot's consciousness will rest now.\n")
                break
            
            elif user_input.lower() == 'stats':
                print_stats(bot)
                continue
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'demo':
                demo_mode(bot)
                continue
            
            # Process as question
            print("\n🤖 Bot: ", end='')
            response = bot.ask(user_input)
            print_response(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for chatting!\n")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Try asking a different question.\n")


def main():
    """Main demo function"""
    
    print_header()
    
    # Ask user preference
    print("Choose mode:")
    print("  1. Demo Mode (scripted demonstration)")
    print("  2. Interactive Mode (free-form chat)")
    print()
    
    choice = input("Enter 1 or 2 (or press Enter for interactive): ").strip()
    
    print("\n⏳ Initializing conscious chatbot...")
    bot = ConsciousnessChatbot()
    print("✅ Chatbot ready!\n")
    
    if choice == '1':
        demo_mode(bot)
    
    # Always go to interactive after demo or if chosen directly
    interactive_mode(bot)
    
    # Final stats
    print("\nFinal Statistics:")
    print_stats(bot)


if __name__ == '__main__':
    main()
