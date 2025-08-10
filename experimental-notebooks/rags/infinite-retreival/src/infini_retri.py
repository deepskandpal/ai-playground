"""
InfiniRetri: A simplified implementation of Infinite Retrieval method
Based on the paper "Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing"
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class InfiniRetri:
    """
    InfiniRetri: Training-free method for handling infinite-length contexts using attention patterns
    
    Key insights from the paper:
    1. Attention allocation patterns align with retrieval-augmented capabilities
    2. LLMs can use their own attention to identify relevant information segments  
    3. Sliding window + attention-based retrieval breaks information barriers
    """
    
    def __init__(self, model_name: str = "gpt2", window_size: int = 512, step_size: int = 256):
        """
        Initialize InfiniRetri with a small LLM
        
        Args:
            model_name: HuggingFace model name (default: gpt2 for demo)
            window_size: Size of each processing window
            step_size: Step size for sliding window (overlap = window_size - step_size)
        """
        self.model_name = model_name
        self.window_size = window_size
        self.step_size = step_size
        
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,  # Critical: we need attention weights
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        
    def get_attention_scores(self, text: str, query: str = None) -> Dict:
        """
        Get attention scores for a given text (and optional query)
        This is the core of InfiniRetri - analyzing attention patterns
        """
        # Prepare input
        if query:
            # For QA tasks, we concatenate query and context
            input_text = f"Question: {query}\nContext: {text}\nAnswer:"
        else:
            input_text = text
            
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.window_size,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract attention weights from all layers
        attentions = outputs.attentions  # Tuple of (batch_size, num_heads, seq_len, seq_len)
        
        return {
            'attentions': attentions,
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'input_ids': inputs['input_ids'],
            'logits': outputs.logits
        }
    
    def visualize_attention_pattern(self, text: str, query: str = None, layer_idx: int = -1):
        """
        Visualize attention patterns as described in the paper
        Shows how attention focuses on relevant parts for retrieval
        """
        attention_data = self.get_attention_scores(text, query)
        attentions = attention_data['attentions']
        tokens = attention_data['tokens']
        
        # Use last layer by default (as paper shows deeper layers have clearer patterns)
        if layer_idx == -1:
            layer_idx = len(attentions) - 1
            
        # Average across attention heads
        attention_matrix = attentions[layer_idx].squeeze(0).mean(dim=0).cpu().numpy()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            cbar=True
        )
        plt.title(f'Attention Pattern - Layer {layer_idx} (InfiniRetri Style)')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return attention_matrix
    
    def segment_and_score(self, long_text: str, query: str) -> List[Dict]:
        """
        Core InfiniRetri algorithm:
        1. Segment long text into overlapping windows
        2. Score each segment using attention-based retrieval
        3. Identify most relevant segments
        """
        # Tokenize the long text
        tokens = self.tokenizer.tokenize(long_text)
        
        segments = []
        segment_scores = []
        
        # Create sliding windows
        for i in range(0, len(tokens), self.step_size):
            segment_tokens = tokens[i:i + self.window_size]
            if len(segment_tokens) < 50:  # Skip very short segments
                break
                
            segment_text = self.tokenizer.convert_tokens_to_string(segment_tokens)
            
            # Get attention scores for this segment with the query
            attention_data = self.get_attention_scores(segment_text, query)
            
            # Calculate relevance score based on attention patterns
            # This implements the paper's insight: "attention allocation aligns with retrieval-augmented"
            relevance_score = self.calculate_attention_relevance_score(
                attention_data, query
            )
            
            segments.append({
                'text': segment_text,
                'start_idx': i,
                'end_idx': i + len(segment_tokens),
                'relevance_score': relevance_score,
                'attention_data': attention_data
            })
            
        return segments
    
    def calculate_attention_relevance_score(self, attention_data: Dict, query: str) -> float:
        """
        Calculate how relevant a segment is based on attention patterns
        This is inspired by the paper's observation that attention patterns align with retrieval
        """
        attentions = attention_data['attentions']
        tokens = attention_data['tokens']
        
        # Use the last layer (as paper shows it has clearest patterns)
        last_layer_attention = attentions[-1].squeeze(0)  # (num_heads, seq_len, seq_len)
        
        # Average across heads
        avg_attention = last_layer_attention.mean(dim=0)  # (seq_len, seq_len)
        
        # Find query tokens in the input
        query_tokens = self.tokenizer.tokenize(query.lower())
        
        relevance_scores = []
        for i, token in enumerate(tokens):
            token_clean = token.replace('Ġ', '').lower()  # GPT-2 uses Ġ for spaces
            
            # If this token appears in query, check what it attends to
            if any(q_token in token_clean for q_token in query_tokens):
                # Sum attention weights from this query token to all context tokens
                attention_from_query = avg_attention[i].sum().item()
                relevance_scores.append(attention_from_query)
        
        # Return average relevance score
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def infini_retrieval(self, long_text: str, query: str, top_k: int = 3) -> Dict:
        """
        Main InfiniRetri method:
        1. Segment long text with sliding windows
        2. Score segments using attention-based relevance
        3. Retrieve top-k most relevant segments
        4. Generate answer using retrieved segments
        """
        print(f"Processing text of {len(long_text)} characters with InfiniRetri...")
        
        # Step 1: Segment and score
        segments = self.segment_and_score(long_text, query)
        
        # Step 2: Rank by relevance score
        segments.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Step 3: Get top-k segments
        top_segments = segments[:top_k]
        
        print(f"Found {len(segments)} segments, retrieving top {top_k}")
        for i, seg in enumerate(top_segments):
            print(f"  Segment {i+1} score: {seg['relevance_score']:.4f}")
        
        # Step 4: Combine retrieved segments for answer generation
        retrieved_context = "\n".join([seg['text'] for seg in top_segments])
        
        # Generate answer using retrieved context
        answer = self.generate_answer(query, retrieved_context)
        
        return {
            'query': query,
            'retrieved_segments': top_segments,
            'retrieved_context': retrieved_context,
            'answer': answer,
            'all_segments': segments
        }
    
    def generate_answer(self, query: str, context: str, max_length: int = 100) -> str:
        """
        Generate answer using retrieved context
        """
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.window_size - max_length  # Leave space for generation
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            generated[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def compare_with_baseline(self, long_text: str, query: str) -> Dict:
        """
        Compare InfiniRetri with baseline (truncated context)
        """
        print("=== Comparison: InfiniRetri vs Baseline ===")
        
        # Baseline: Just use first window_size tokens
        truncated_text = self.tokenizer.decode(
            self.tokenizer.encode(long_text)[:self.window_size-100],
            skip_special_tokens=True
        )
        
        baseline_answer = self.generate_answer(query, truncated_text)
        
        # InfiniRetri: Use our method
        infini_result = self.infini_retrieval(long_text, query)
        
        return {
            'baseline': {
                'context': truncated_text,
                'answer': baseline_answer
            },
            'infini_retri': infini_result
        }