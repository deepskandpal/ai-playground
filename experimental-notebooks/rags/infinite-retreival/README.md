# InfiniRetri: Attention Enhanced LLMs for Long-Context Processing

This repository contains a simplified implementation of the InfiniRetri method described in the paper "Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing" by Xiaoju Ye, Zhichun Wang, and Jingyuan Wang.

## 📄 Paper Summary

**Key Problem**: Large Language Models (LLMs) struggle with input sequences that exceed their context window size, whether for simple retrieval tasks or complex multi-hop reasoning.

**Solution**: InfiniRetri leverages the LLM's own attention patterns to identify and retrieve relevant segments from infinitely long contexts, without requiring additional training.

**Main Achievement**: 100% accuracy on Needle-In-a-Haystack (NIH) tests over 1M tokens using just a 0.5B parameter model, with up to 288% improvement on real-world benchmarks.

## 🧠 Key Insights from the Paper

1. **Attention-Retrieval Alignment**: The attention allocation patterns in LLMs naturally align with retrieval-augmented capabilities
2. **Training-Free**: Works with any Transformer-based LLM without additional training
3. **Breaking Context Barriers**: Uses sliding windows + attention-based retrieval to process unlimited length contexts
4. **Internal Capabilities**: Leverages the LLM's own attention rather than external embedding models

## 🏗️ How InfiniRetri Works

### 1. Sliding Window Segmentation
- Divides long text into overlapping windows
- Each window fits within the model's context limit
- Overlap ensures no information is lost between segments

### 2. Attention-Based Scoring
- Analyzes attention patterns between query and context
- Uses deeper layers (clearer attention patterns)
- Scores segments based on query-context attention alignment

### 3. Retrieval and Generation
- Ranks segments by relevance scores
- Retrieves top-K most relevant segments
- Generates final answer using retrieved context

## 📁 Project Structure

```
infinite-retrieval/
├── infinite-retrieval.pdf      # Original paper
├── src/
│   ├── infini_retri.py        # Main InfiniRetri implementation
│   ├── quick_demo.py          # Simple demo script  
│   └── harder_demo.py         # Challenging needle-in-haystack demo
├── notebooks/
│   └── infini_retri_demo.ipynb # Interactive Jupyter notebook demo
└── README.md                   # This file
```

## 🚀 Running the Demo

### Quick Setup
```bash
# Navigate to project directory
cd infinite-retrieval

# Run simple demo
uv run python src/quick_demo.py

# Run challenging demo
uv run python src/harder_demo.py

# Or use the interactive notebook
uv run jupyter notebook notebooks/infini_retri_demo.ipynb
```

## 🔬 What Our Implementation Demonstrates

### ✅ Successfully Implemented:
- Sliding window mechanism for processing long contexts
- Attention pattern visualization
- Basic attention-based relevance scoring
- Comparison with baseline (truncated context)
- Training-free approach working with GPT-2

### 🎯 Core Concepts Shown:
1. **Attention Patterns**: How LLMs naturally focus on relevant information
2. **Context Window Limitations**: Why simple truncation fails
3. **Sliding Window Processing**: Breaking long texts into manageable chunks
4. **Training-Free Enhancement**: No additional model training required

### 📊 Expected vs Actual Results:

**Paper Results (with optimized implementation)**:
- 100% accuracy on 1M token NIH tasks
- 288% improvement on real benchmarks
- Works with models from 0.5B to 7B+ parameters

**Our Demo Results (simplified implementation)**:
- Shows the core mechanism working
- Demonstrates attention pattern analysis  
- Highlights the sliding window approach
- Basic relevance scoring (needs refinement for challenging cases)

## 🧪 Demo Scenarios

### 1. Simple Demo (`quick_demo.py`)
- Creates a moderate-length context with a hidden "password"
- Shows both baseline and InfiniRetri approaches
- Demonstrates successful retrieval in easier cases

### 2. Challenging Demo (`harder_demo.py`)  
- Places the "needle" at the very end of a long context
- Forces multiple sliding windows
- Shows where baseline truncation fails completely
- Highlights the need for sophisticated attention scoring

### 3. Interactive Notebook (`infini_retri_demo.ipynb`)
- Step-by-step walkthrough of all concepts
- Attention pattern visualizations
- Detailed analysis of segment scoring
- Comprehensive comparison between approaches

## 🔍 Key Implementation Details

### Attention Analysis
```python
def get_attention_scores(self, text: str, query: str = None) -> Dict:
    # Get attention weights from all layers
    outputs = self.model(**inputs)
    attentions = outputs.attentions
    
    # Use deeper layers for clearer patterns
    last_layer_attention = attentions[-1]
```

### Sliding Window Processing
```python
def segment_and_score(self, long_text: str, query: str) -> List[Dict]:
    # Create overlapping windows
    for i in range(0, len(tokens), self.step_size):
        segment_tokens = tokens[i:i + self.window_size]
        relevance_score = self.calculate_attention_relevance_score(...)
```

### Relevance Scoring
```python
def calculate_attention_relevance_score(self, attention_data: Dict, query: str) -> float:
    # Find query tokens and analyze what they attend to
    # This implements the paper's core insight about attention-retrieval alignment
```

## 💡 Why This Matters

### Traditional Approach Problems:
- **Context Window Limits**: Can only process fixed-length inputs
- **Information Loss**: Important info beyond the window is ignored
- **Expensive Solutions**: Extending context windows requires costly retraining

### InfiniRetri Advantages:
- **Unlimited Length**: Processes any length input through intelligent retrieval
- **Training-Free**: Works immediately with existing models
- **Cost Effective**: No expensive retraining or large context windows needed
- **Attention-Aware**: Uses the model's own understanding of relevance

## 🔬 Research Implications

This work suggests that:
1. **LLMs already know how to retrieve** - we just need to use their attention patterns
2. **Context extension isn't the only solution** - better utilization of existing capabilities can be more effective
3. **Training-free methods** can achieve remarkable results
4. **Future RAG systems** might benefit from using LLM attention rather than external embeddings

## 🛠️ Potential Improvements

Our simplified implementation could be enhanced with:
1. **Better Attention Scoring**: More sophisticated analysis of attention patterns
2. **Multi-Layer Analysis**: Combining insights from multiple attention layers  
3. **Query-Aware Segmentation**: Adjusting windows based on query characteristics
4. **Caching Optimizations**: Avoiding redundant computations across segments
5. **Larger Models**: Testing with models closer to the paper's scale

## 📚 References

- Original Paper: "Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing"
- Authors: Xiaoju Ye, Zhichun Wang, Jingyuan Wang
- arXiv: 2502.12962v1 [cs.CL] 18 Feb 2025

## 🎯 Conclusion

InfiniRetri represents a paradigm shift from simply making context windows bigger to making better use of what LLMs already know through their attention mechanisms. Our implementation demonstrates the core concepts and shows how this training-free approach can enhance any Transformer-based model's ability to handle long contexts.

The key insight - **"attention allocation patterns align with retrieval-augmented capabilities"** - opens new possibilities for both long-context processing and retrieval-augmented generation systems.