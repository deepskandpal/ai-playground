"""
Quick Demo of InfiniRetri - Run this to see the method in action!
"""

from infini_retri import InfiniRetri
import random

def main():
    print("🚀 InfiniRetri Quick Demo")
    print("=" * 50)
    
    # Initialize with GPT-2
    print("Loading GPT-2 model...")
    infini_retri = InfiniRetri(
        model_name="gpt2",
        window_size=400,  # Smaller for quick demo
        step_size=200
    )
    
    # Create a needle-in-haystack scenario
    print("\n📝 Creating long context with hidden information...")
    
    haystack_parts = [
        "The history of artificial intelligence dates back to ancient times. ",
        "Modern AI research began in the 1950s with pioneers like Alan Turing. ",
        "Machine learning algorithms have evolved significantly over decades. ",
        "Deep learning uses neural networks with multiple layers. ",
        "Natural language processing helps computers understand human language. ",
        "The secret password is DEMO123XYZ for testing purposes. ",  # THE NEEDLE!
        "Computer vision allows machines to interpret visual information. ",
        "Reinforcement learning focuses on learning through interaction. ",
        "Large language models have revolutionized NLP applications. ",
        "Robotics combines AI with mechanical engineering systems. ",
        "AI ethics includes considerations of bias and fairness. ",
        "The future of AI holds promise for solving global challenges. "
    ]
    
    # Create long context by repeating and shuffling
    random.seed(42)
    long_text = ""
    for _ in range(8):  # Repeat 8 times
        shuffled = haystack_parts.copy()
        random.shuffle(shuffled)
        long_text += " ".join(shuffled) + " "
    
    print(f"Created context with {len(long_text)} characters")
    
    # Test query
    query = "What is the secret password?"
    print(f"\n🔍 Query: {query}")
    print("Expected answer should contain: DEMO123XYZ")
    
    # Compare baseline vs InfiniRetri
    print("\n⚖️ Comparing methods...")
    comparison = infini_retri.compare_with_baseline(long_text, query)
    
    print("\n" + "="*60)
    print("📊 RESULTS:")
    print("="*60)
    
    # Baseline results
    baseline_answer = comparison['baseline']['answer']
    baseline_context = comparison['baseline']['context']
    baseline_found = "DEMO123XYZ" in (baseline_answer + baseline_context)
    
    print(f"\n🔹 BASELINE (Truncated Context):")
    print(f"   Answer: {baseline_answer}")
    print(f"   Found password: {'✅ YES' if baseline_found else '❌ NO'}")
    
    # InfiniRetri results  
    infini_result = comparison['infini_retri']
    infini_answer = infini_result['answer']
    infini_context = infini_result['retrieved_context']
    infini_found = "DEMO123XYZ" in (infini_answer + infini_context)
    
    print(f"\n🔹 INFINIRETRI (Attention-based Retrieval):")
    print(f"   Answer: {infini_answer}")
    print(f"   Found password: {'✅ YES' if infini_found else '❌ NO'}")
    
    # Show retrieved segments
    print(f"\n📄 Top Retrieved Segments:")
    for i, seg in enumerate(infini_result['retrieved_segments'][:2]):
        contains_needle = "DEMO123XYZ" in seg['text']
        print(f"   Segment {i+1} (score: {seg['relevance_score']:.3f}) {'🎯' if contains_needle else ''}")
        print(f"   {seg['text'][:150]}...")
    
    print("\n" + "="*60)
    print("🎉 SUMMARY:")
    print(f"   Baseline found the needle: {'✅' if baseline_found else '❌'}")
    print(f"   InfiniRetri found the needle: {'✅' if infini_found else '❌'}")
    
    if infini_found and not baseline_found:
        print("   🏆 InfiniRetri successfully retrieved the hidden information!")
    elif infini_found and baseline_found:
        print("   ⚖️ Both methods found it (needle was in early context)")
    elif not infini_found and baseline_found:
        print("   🤔 Baseline got lucky, InfiniRetri missed it")
    else:
        print("   😅 Neither found it - try running again or adjusting parameters")
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS:")
    print("   • InfiniRetri uses the LLM's own attention to find relevant info")
    print("   • No additional training required - works with any Transformer")
    print("   • Sliding windows + attention scoring handles unlimited length")
    print("   • Paper shows 100% accuracy on 1M+ token contexts!")

if __name__ == "__main__":
    main()