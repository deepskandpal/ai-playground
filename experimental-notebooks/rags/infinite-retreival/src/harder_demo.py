"""
Harder Demo of InfiniRetri - Needle positioned later in the context
This better demonstrates the advantage over baseline truncation
"""

from infini_retri import InfiniRetri
import random

def main():
    print("🎯 InfiniRetri Challenging Demo - Needle at the End!")
    print("=" * 60)
    
    # Initialize with GPT-2
    print("Loading GPT-2 model...")
    infini_retri = InfiniRetri(
        model_name="gpt2",
        window_size=300,  # Smaller window to force multiple segments
        step_size=150
    )
    
    # Create a more challenging needle-in-haystack scenario
    print("\n📝 Creating long context with needle positioned LATE...")
    
    # Create many filler segments without the needle
    filler_parts = [
        "Artificial intelligence has a long history dating back centuries. ",
        "Early AI research focused on symbolic reasoning and expert systems. ",
        "Machine learning emerged as a dominant paradigm in the 1980s. ",
        "Neural networks were inspired by biological brain structures. ",
        "Deep learning revolutionized AI in the 2010s with breakthrough results. ",
        "Computer vision systems can now recognize objects with high accuracy. ",
        "Natural language processing enables human-computer communication. ",
        "Robotics combines AI with physical systems for autonomous operation. ",
        "Reinforcement learning teaches agents through trial and error. ",
        "Transfer learning allows models to adapt to new domains efficiently. ",
        "Ensemble methods combine multiple models for better performance. ",
        "Feature engineering was crucial before deep learning automation. ",
        "Cross-validation helps prevent overfitting in machine learning. ",
        "Hyperparameter tuning optimizes model performance systematically. ",
        "Data preprocessing is essential for successful AI applications. ",
        "Model interpretability remains a challenge in complex systems. ",
        "Ethical AI considerations include fairness and transparency. ",
        "Scalability issues arise when deploying AI systems in production. ",
        "Distributed computing enables training of very large models. ",
        "Cloud platforms provide accessible AI infrastructure for everyone. "
    ]
    
    # The needle - we'll place this at the END
    needle_part = "The special access code for the secure system is HIDDEN999SECRET and should be kept confidential. "
    
    # Build context: lots of filler, then the needle at the end
    long_text = ""
    
    # Add many repetitions of filler content (this will be > 300 tokens, forcing multiple windows)
    for _ in range(15):  # This creates a very long context
        random.seed(42 + _)  # Different seed each time for variety
        shuffled_filler = filler_parts.copy()
        random.shuffle(shuffled_filler)
        long_text += " ".join(shuffled_filler) + " "
    
    # Add the needle at the very END (this will be in the last segment)
    long_text += needle_part
    
    print(f"Created context with {len(long_text)} characters")
    print(f"Estimated tokens: ~{len(long_text.split())}")
    print(f"Window size: {infini_retri.window_size} tokens")
    print("📍 Needle is positioned at the END of the context")
    
    # Test query
    query = "What is the special access code for the secure system?"
    print(f"\n🔍 Query: {query}")
    print("Expected answer should contain: HIDDEN999SECRET")
    
    # Run the comparison
    print("\n⚖️ Comparing methods...")
    comparison = infini_retri.compare_with_baseline(long_text, query)
    
    print("\n" + "="*60)
    print("📊 RESULTS:")
    print("="*60)
    
    # Baseline results (should miss the needle since it's at the end)
    baseline_answer = comparison['baseline']['answer']
    baseline_context = comparison['baseline']['context']
    baseline_found = "HIDDEN999SECRET" in (baseline_answer + baseline_context)
    
    print(f"\n🔹 BASELINE (First {infini_retri.window_size} tokens only):")
    print(f"   Answer: {baseline_answer}")
    print(f"   Context ends with: ...{baseline_context[-100:]}")
    print(f"   Found access code: {'✅ YES' if baseline_found else '❌ NO'}")
    print(f"   Why: {'Found in early context' if baseline_found else 'Code is at the end, truncated away'}")
    
    # InfiniRetri results (should find the needle via attention-based retrieval)
    infini_result = comparison['infini_retri']
    infini_answer = infini_result['answer']
    infini_context = infini_result['retrieved_context']
    infini_found = "HIDDEN999SECRET" in (infini_answer + infini_context)
    
    print(f"\n🔹 INFINIRETRI (Attention-based Retrieval):")
    print(f"   Answer: {infini_answer}")
    print(f"   Found access code: {'✅ YES' if infini_found else '❌ NO'}")
    print(f"   Why: {'Found via attention patterns' if infini_found else 'Attention mechanism missed it'}")
    
    # Show retrieved segments and their scores
    print(f"\n📄 InfiniRetri Segment Analysis:")
    print(f"   Total segments processed: {len(infini_result['all_segments'])}")
    
    # Sort all segments by relevance score
    all_segments = sorted(infini_result['all_segments'], key=lambda x: x['relevance_score'], reverse=True)
    
    print(f"   Top 3 segments by relevance score:")
    for i, seg in enumerate(all_segments[:3]):
        contains_needle = "HIDDEN999SECRET" in seg['text']
        print(f"     #{i+1}: Score {seg['relevance_score']:.4f} {'🎯 (HAS NEEDLE!)' if contains_needle else ''}")
        print(f"          {seg['text'][:120]}...")
    
    # Find where the needle segment ranked
    needle_segments = [i for i, seg in enumerate(all_segments) if "HIDDEN999SECRET" in seg['text']]
    if needle_segments:
        needle_rank = needle_segments[0] + 1
        needle_score = all_segments[needle_segments[0]]['relevance_score']
        print(f"\n🎯 Needle Analysis:")
        print(f"   Needle segment rank: #{needle_rank} out of {len(all_segments)}")
        print(f"   Needle segment score: {needle_score:.4f}")
        print(f"   In top 3 retrieved: {'✅ YES' if needle_rank <= 3 else '❌ NO'}")
    
    print("\n" + "="*60)
    print("🏆 FINAL VERDICT:")
    print("="*60)
    
    if infini_found and not baseline_found:
        print("   🥇 InfiniRetri WINS! Found the needle that baseline missed!")
        print("   💡 This demonstrates the power of attention-based retrieval")
        print("      over simple context truncation.")
    elif infini_found and baseline_found:
        print("   🤝 TIE: Both methods found the needle")
        print("   🤔 Try increasing context length to make it more challenging")
    elif not infini_found and baseline_found:
        print("   😲 Baseline wins (unlikely scenario)")
        print("   🔧 This suggests the attention mechanism needs tuning")
    else:
        print("   😅 Neither method found it")
        print("   🔍 The attention-based scoring may need refinement")
    
    print("\n" + "="*60)
    print("🧠 WHAT THIS DEMONSTRATES:")
    print("   • Traditional approaches fail when relevant info is far from the start")
    print("   • InfiniRetri's attention mechanism can find needles anywhere in the context")
    print("   • Sliding windows + retrieval breaks the context length barrier")
    print("   • The LLM's own attention patterns guide the search effectively")
    print("   • This is why the paper achieved 100% accuracy on 1M+ token tasks!")

if __name__ == "__main__":
    main()