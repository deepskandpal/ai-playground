# CLAUDE.md - AI Assistant Guide for ai-playground Repository

This document provides comprehensive guidance for AI assistants working with the ai-playground codebase. It covers repository structure, development workflows, coding conventions, and best practices.

**Last Updated**: 2026-01-18
**Repository**: ai-playground
**Purpose**: ML/AI experimentation and research notebooks

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Development Environment](#development-environment)
4. [Coding Conventions](#coding-conventions)
5. [Documentation Standards](#documentation-standards)
6. [Working with Notebooks](#working-with-notebooks)
7. [Project Organization Patterns](#project-organization-patterns)
8. [Common Workflows](#common-workflows)
9. [Key Files Reference](#key-files-reference)
10. [Best Practices for AI Assistants](#best-practices-for-ai-assistants)

---

## Repository Overview

**ai-playground** is a collection of ML/AI experiments, research implementations, and educational notebooks covering:

- **Mathematics for Deep Learning**: Mathematical foundations and concepts
- **Retrieval-Augmented Generation (RAG)**: Advanced RAG implementations including InfiniRetri
- **Reinforcement Learning**: RL algorithms and policy search methods
- **Fine-tuning**: LLM fine-tuning experiments and comprehensive guides

### Repository Philosophy

- **Research-first**: Focus on understanding and implementing cutting-edge papers
- **Educational**: Comprehensive documentation and step-by-step explanations
- **Experimentation**: Sandbox for trying new ideas and techniques
- **Clean separation**: Source code implementations separate from interactive notebooks

---

## Directory Structure

```
ai-playground/
├── README.md                           # Repository overview
├── CLAUDE.md                          # This file - AI assistant guide
├── complete_finetuning_guide.md       # Comprehensive fine-tuning tutorial
├── .gitignore                         # Standard Python/Jupyter patterns
└── experimental-notebooks/            # All experiments organized by topic
    ├── README.md                      # Experiments overview
    │
    ├── maths-for-dl/                  # Mathematical foundations
    │   ├── pyproject.toml             # Project dependencies
    │   ├── uv.lock                    # Dependency lock file
    │   ├── maths-for-dl.ipynb         # Main notebook
    │   └── linear_algebra_intro.ipynb # Specific topic notebooks
    │
    ├── fine-tuning/                   # LLM fine-tuning experiments
    │   ├── pyproject.toml
    │   └── 02. llm_finetuning_hf.ipynb
    │
    ├── rags/                          # RAG implementations
    │   └── infinite-retreival/        # InfiniRetri paper implementation
    │       ├── README.md              # Project-specific documentation
    │       ├── pyproject.toml         # Project dependencies
    │       ├── uv.lock
    │       ├── src/                   # Python implementations
    │       │   ├── infini_retri.py    # Main class implementation
    │       │   ├── quick_demo.py      # Simple demonstration
    │       │   └── harder_demo.py     # Advanced demonstration
    │       └── notebooks/             # Interactive demos
    │           ├── infini_retri_demo.ipynb
    │           └── infini_retri_paper_analysis.ipynb
    │
    ├── reinforcement-learning/        # RL experiments
    │   └── initial-research/
    │       ├── GRPO_Example.ipynb
    │       └── reinforcement_learning_policy_search.ipynb
    │
    └── Triton-Puzzles.ipynb           # GPU kernel optimization puzzles
```

### Key Organizational Principles

1. **Hierarchical by topic**: Each major ML area gets its own directory
2. **Self-contained projects**: Major implementations have their own `pyproject.toml`
3. **Dual structure**: Both `src/` (reusable code) and `notebooks/` (interactive demos)
4. **Multiple README levels**: Root, experimental-notebooks, and per-project documentation

---

## Development Environment

### Package Management

**Tool**: UV (modern Python package manager)
**Why**: Faster than pip, better dependency resolution, built-in virtual environments

### Setting Up a Project

```bash
# Navigate to a project directory
cd experimental-notebooks/rags/infinite-retreival

# UV automatically handles environment and dependencies
uv run python src/quick_demo.py

# For Jupyter notebooks
uv run jupyter notebook notebooks/infini_retri_demo.ipynb
```

### Python Version

- **Minimum**: Python 3.11+
- **Specified in**: All `pyproject.toml` files via `requires-python = ">=3.11"`

### Dependency Management

#### Standard Dependencies Pattern

```toml
[project]
name = "project-name"
version = "0.1.0"
description = "Project description"
requires-python = ">=3.11"
dependencies = [
    # Jupyter/Notebook support
    "ipykernel>=6.30.1",
    "jupyter>=1.1.1",

    # Visualization
    "matplotlib>=3.10.5",
    "plotly>=6.2.0",
    "seaborn>=0.13.2",

    # Core ML libraries
    "torch>=2.8.0",
    "transformers>=4.55.0",
    "numpy>=2.3.2",
]

[project.optional-dependencies]
gpu = [
    # GPU-specific packages (excluded on macOS)
    "unsloth; sys_platform != 'darwin'",
    "vllm; sys_platform != 'darwin'",
    "bitsandbytes; sys_platform != 'darwin'",
]
```

#### Key Points

- **Minimum versions specified**: Ensures compatibility while allowing patches
- **Platform-aware**: GPU dependencies excluded on macOS
- **Optional extras**: GPU enhancements as `[project.optional-dependencies]`
- **Lock files**: `uv.lock` for reproducibility

---

## Coding Conventions

### Python Style Guide

#### File Naming

- **Modules/Classes**: `snake_case` (e.g., `infini_retri.py`)
- **Demo scripts**: Descriptive with `_demo` suffix (e.g., `quick_demo.py`, `harder_demo.py`)
- **Test files**: Not currently used (experimental repo)

#### Import Organization

```python
# 1. Standard library imports
import warnings
from typing import List, Dict, Tuple, Optional

# 2. Third-party libraries (grouped by purpose)
# Core ML frameworks
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Scientific computing
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Local imports (if any)
# from .utils import helper_function
```

**Order**: System imports → External libraries → Type hints → Local imports

#### Type Hints

**Always use comprehensive type hints**:

```python
def segment_and_score(self, long_text: str, query: str) -> List[Dict]:
    """Process text and return scored segments."""
    pass

def calculate_score(
    self,
    attention_data: Dict,
    query: str,
    threshold: float = 0.5
) -> float:
    """Calculate relevance score with optional threshold."""
    pass

# Optional parameters
def analyze(self, text: str, query: Optional[str] = None) -> Dict:
    """Analyze text with optional query."""
    pass
```

#### Class Structure

```python
class ExampleClass:
    """
    Clear class description.

    Key features:
    1. Feature one
    2. Feature two
    3. Feature three
    """

    def __init__(self, param1: str, param2: int = 512):
        """
        Initialize the class.

        Args:
            param1: Description of param1
            param2: Description of param2 (default: 512)
        """
        self.param1 = param1
        self.param2 = param2

    # Core functionality methods
    def primary_method(self, input_data: str) -> Dict:
        """Main processing method."""
        pass

    # Helper/utility methods
    def _helper_method(self, data: List) -> List:
        """Internal helper (prefixed with _)."""
        pass

    # Visualization/analysis methods
    def visualize(self, data: Dict) -> None:
        """Visualization method."""
        pass
```

**Method organization**:
1. `__init__` and setup methods
2. Core public methods
3. Helper/utility methods (prefixed with `_` if private)
4. Visualization/analysis methods

#### Resource Management

```python
# Always use context managers for resources
with torch.no_grad():
    outputs = self.model(**inputs)

# Device detection
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self.model.to(self.device)

# Explicit model evaluation mode
self.model.eval()
```

#### Error Handling

```python
# Suppress warnings when intentional
import warnings
warnings.filterwarnings('ignore')  # Document why

# Graceful fallbacks
try:
    result = expensive_operation()
except OutOfMemoryError:
    print("Falling back to CPU...")
    result = cpu_operation()
```

---

## Documentation Standards

### Module-Level Documentation

Every Python file should start with a module docstring:

```python
"""
ModuleName: Brief description

Detailed explanation of what this module does.
Based on [Paper Name] by [Authors] if applicable.

Key concepts:
1. Concept one
2. Concept two
"""
```

### Class Documentation

```python
class InfiniRetri:
    """
    One-line description of the class.

    Longer description explaining the purpose, approach, and key insights.

    Key insights from the paper:
    1. Insight one
    2. Insight two
    3. Insight three

    Attributes:
        model_name: Description
        window_size: Description
    """
```

### Function/Method Documentation

```python
def segment_and_score(self, long_text: str, query: str) -> List[Dict]:
    """
    Brief description of what this does.

    More detailed explanation of the algorithm or approach:
    1. Step one
    2. Step two
    3. Step three

    Args:
        long_text: Description of parameter
        query: Description of parameter

    Returns:
        List of dictionaries containing:
        - 'text': Segment text
        - 'score': Relevance score
        - 'start_idx': Starting position

    Example:
        >>> segments = model.segment_and_score(text, "Find the password")
        >>> top_segment = max(segments, key=lambda x: x['score'])
    """
    pass
```

### README Structure

Each major project should have a README.md with:

```markdown
# Project Name: Brief Description

One-paragraph overview of what this project does and why it matters.

## 📄 Paper Summary (if applicable)

**Key Problem**: What problem does this solve?

**Solution**: How does it solve it?

**Main Achievement**: Key results or metrics.

## 🧠 Key Insights

1. Insight one
2. Insight two
3. Insight three

## 🏗️ How It Works

### Step 1: Description
Details...

### Step 2: Description
Details...

## 📁 Project Structure

```
project/
├── src/
├── notebooks/
└── README.md
```

## 🚀 Running the Demo

### Quick Setup
```bash
# Commands to run
```

## 🔬 What This Demonstrates

- Point one
- Point two
- Point three

## 💡 Why This Matters

Explanation of significance and applications.

## 📚 References

- Paper citation
- Links to resources
```

### Inline Comments

**Comment the "why", not the "what"**:

```python
# Good: Explains reasoning
# Use deeper layers for clearer attention patterns (per paper findings)
layer_idx = len(attentions) - 1

# Bad: States the obvious
# Set layer_idx to length of attentions minus 1
layer_idx = len(attentions) - 1
```

```python
# Good: Provides context
# Skip very short segments - insufficient context for meaningful scoring
if len(segment_tokens) < 50:
    break

# Bad: Redundant
# If segment is less than 50 tokens, break
if len(segment_tokens) < 50:
    break
```

### Markdown Formatting

#### Use emoji for visual organization (in documentation only):

- 📄 Papers/Documents
- 🔬 Research/Experiments
- 🚀 Setup/Getting Started
- 💡 Key Insights
- ✅ Successes
- 🎯 Goals/Objectives
- 🛠️ Tools/Implementation

#### Code blocks with language specification:

```markdown
```python
def example():
    pass
```
```

---

## Working with Notebooks

### Notebook Naming Conventions

1. **Sequential/Tutorial**: Use numbered prefix
   - `02. llm_finetuning_hf.ipynb`
   - Helps with ordering in file browsers

2. **Descriptive names**: Clear indication of content
   - `infini_retri_demo.ipynb`
   - `linear_algebra_intro.ipynb`

3. **Analysis notebooks**: Suffix with `_analysis`
   - `infini_retri_paper_analysis.ipynb`

### Notebook Structure Pattern

```
1. Title & Overview (Markdown)
   - Clear H1 title
   - Problem statement
   - What this notebook demonstrates

2. Setup & Imports (Code)
   - Path setup if needed
   - Library imports
   - Configuration

3. Conceptual Sections (Alternating Markdown + Code)
   - Markdown: Explain the concept
   - Code: Demonstrate with implementation
   - Output: Show results
   - Repeat for each concept

4. Visualization & Analysis (Code + Markdown)
   - Visual outputs (plots, heatmaps)
   - Interpretation of results
   - Comparison with baselines

5. Summary & Key Takeaways (Markdown)
   - What we learned
   - Important insights
   - Next steps or further reading
```

### Cell Organization Best Practices

```python
# Cell 1: Imports (one cell for all imports)
import torch
import numpy as np
from transformers import AutoModel

# Cell 2: Configuration (separate cell for config)
MODEL_NAME = "gpt2"
WINDOW_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cell 3: Single concept demonstration
# Demonstrate attention patterns
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
outputs = model(**inputs)
print(f"Attention shape: {outputs.attentions[0].shape}")

# Cell 4: Visualization (separate from computation)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix)
plt.title("Attention Patterns")
plt.show()
```

### Markdown Cells in Notebooks

```markdown
# Main Section Title

Brief introduction to this section and what we'll explore.

## Subsection: Specific Concept

Detailed explanation with:
- Key points in bullet form
- **Bold** for emphasis
- `code snippets` for technical terms

### Implementation Details

More specific information about how we'll implement this.

> **Note**: Important callouts in blockquotes
```

### Running Notebooks

```bash
# From project directory
uv run jupyter notebook notebooks/example.ipynb

# Or start Jupyter Lab
uv run jupyter lab
```

---

## Project Organization Patterns

### Pattern 1: Simple Experiment (Single Notebook)

```
topic-name/
├── pyproject.toml
├── uv.lock
└── experiment_notebook.ipynb
```

**Use when**: Exploring a concept, quick experiments, tutorials

**Example**: `maths-for-dl/linear_algebra_intro.ipynb`

### Pattern 2: Implementation Project (Source + Notebooks)

```
project-name/
├── README.md              # Comprehensive project documentation
├── pyproject.toml         # Dependencies
├── uv.lock                # Lock file
├── src/                   # Python implementations
│   ├── main_class.py      # Core functionality
│   ├── utils.py           # Helper functions
│   ├── quick_demo.py      # Simple demonstration
│   └── harder_demo.py     # Advanced demonstration
└── notebooks/             # Interactive notebooks
    ├── demo.ipynb         # Interactive walkthrough
    └── paper_analysis.ipynb  # Research analysis
```

**Use when**: Implementing papers, reusable components, complex projects

**Example**: `rags/infinite-retreival/`

### Pattern 3: Topic Collection (Multiple Related Notebooks)

```
topic-area/
├── README.md
├── initial-research/
│   ├── concept1.ipynb
│   └── concept2.ipynb
└── advanced-topics/
    └── advanced.ipynb
```

**Use when**: Exploring a research area with multiple approaches

**Example**: `reinforcement-learning/`

### Choosing the Right Pattern

| Pattern | Use When | Examples |
|---------|----------|----------|
| **Simple** | Learning, tutorials, single concepts | Math foundations, quick tests |
| **Implementation** | Paper implementations, reusable code | InfiniRetri, custom models |
| **Collection** | Research area exploration | RL algorithms, various approaches |

---

## Common Workflows

### Adding a New Experiment

1. **Choose directory structure** based on patterns above

2. **Create project directory**:
   ```bash
   mkdir -p experimental-notebooks/new-topic
   cd experimental-notebooks/new-topic
   ```

3. **Initialize with pyproject.toml**:
   ```bash
   # Create pyproject.toml
   cat > pyproject.toml << 'EOF'
   [project]
   name = "new-topic"
   version = "0.1.0"
   description = "Description here"
   requires-python = ">=3.11"
   dependencies = [
       "ipykernel>=6.30.1",
       "jupyter>=1.1.1",
       "torch>=2.8.0",
       # Add other dependencies
   ]
   EOF
   ```

4. **Install dependencies**:
   ```bash
   uv sync
   ```

5. **Create initial notebook or source files**

6. **Add README.md** following the standard structure

### Implementing a Research Paper

1. **Create implementation project** using Pattern 2:
   ```bash
   mkdir -p experimental-notebooks/category/paper-name/{src,notebooks}
   ```

2. **Add README with paper summary** including:
   - Paper citation
   - Key problem and solution
   - Main achievements
   - How to run demos

3. **Implement in `src/`**:
   - Main class in `paper_name.py`
   - Demo scripts (`quick_demo.py`, `harder_demo.py`)

4. **Create notebooks/** for:
   - Interactive demonstrations
   - Paper analysis and insights
   - Visualizations

5. **Document thoroughly**:
   - Module docstrings with paper reference
   - Method docstrings explaining paper concepts
   - README with comprehensive explanation

### Creating Demonstrations

**Quick Demo** (`quick_demo.py`):
```python
"""
Quick demonstration of [Feature].

Shows basic functionality with simple example.
"""

def main():
    print("=" * 60)
    print("Quick Demo: [Feature Name]")
    print("=" * 60)

    # Setup
    print("\n1. Setting up...")
    model = initialize_model()

    # Simple example
    print("\n2. Running simple example...")
    result = model.process(simple_input)

    # Show results
    print("\n3. Results:")
    print(f"Output: {result}")

    print("\n" + "=" * 60)
    print("Demo complete!")

if __name__ == "__main__":
    main()
```

**Harder Demo** (`harder_demo.py`):
```python
"""
Advanced demonstration of [Feature].

Shows challenging scenario and compares with baseline.
"""

def main():
    print("=" * 60)
    print("Advanced Demo: [Feature Name]")
    print("=" * 60)

    # Setup
    print("\n1. Setting up challenging scenario...")

    # Baseline comparison
    print("\n2. Running baseline approach...")
    baseline_result = baseline_method()

    # New approach
    print("\n3. Running new approach...")
    new_result = new_method()

    # Comparison
    print("\n4. Comparison:")
    print(f"Baseline: {baseline_result}")
    print(f"New method: {new_result}")
    print(f"Improvement: {calculate_improvement()}%")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

### Writing Documentation

1. **Start with module docstring** explaining purpose
2. **Add class docstrings** with key insights
3. **Document all public methods** with Args/Returns
4. **Create README.md** following standard structure
5. **Update parent README** if needed to link to new project

### Testing Code

While this is primarily an experimental repository, when creating demos:

1. **Create simple test case first** (`quick_demo.py`)
2. **Create challenging test case** (`harder_demo.py`)
3. **Compare with baseline** to validate improvements
4. **Document expected vs. actual results** in README

---

## Key Files Reference

### Essential Reading

| File | Purpose | Key Learnings |
|------|---------|---------------|
| `experimental-notebooks/rags/infinite-retreival/src/infini_retri.py` | Well-documented class implementation | Type hints, docstrings, class structure |
| `experimental-notebooks/rags/infinite-retreival/README.md` | Comprehensive project README | Documentation structure, emoji usage |
| `complete_finetuning_guide.md` | Educational guide | Tutorial writing, concept explanation |
| `experimental-notebooks/rags/infinite-retreival/src/quick_demo.py` | Simple demo pattern | Demo script structure |

### Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python project dependencies and metadata |
| `uv.lock` | Locked dependency versions for reproducibility |
| `.gitignore` | Git ignore patterns (IDE, checkpoints, env files) |

---

## Best Practices for AI Assistants

### When Adding New Code

1. **Check existing patterns first**: Look at similar implementations in the repo
2. **Follow the established structure**: Use Pattern 1, 2, or 3 as appropriate
3. **Use comprehensive type hints**: Always include parameter and return types
4. **Document thoroughly**: Module, class, and method docstrings required
5. **Create demos**: At minimum a `quick_demo.py` showing basic usage
6. **Add README**: For any new major project or implementation

### When Modifying Existing Code

1. **Preserve existing style**: Match indentation, naming, and structure
2. **Update docstrings**: If changing functionality, update documentation
3. **Test with demos**: Run existing demo scripts to verify changes
4. **Update README**: Reflect any significant changes in documentation

### When Creating Documentation

1. **Use emoji sparingly**: Only in README/markdown, not in code or docstrings
2. **Follow README structure**: Use the standard sections shown above
3. **Include code examples**: Show actual usage from the implementation
4. **Link to papers**: Always cite source papers for implementations
5. **Explain "why"**: Focus on insights and understanding, not just mechanics

### When Writing Notebooks

1. **Narrative structure**: Tell a story from problem → solution → results
2. **One concept per cell**: Don't cram multiple ideas into one code cell
3. **Alternate explanation and code**: Markdown → Code → Output pattern
4. **Include visualizations**: Use plots, heatmaps, and diagrams
5. **Add clear section headers**: Use markdown headers to organize content
6. **Show progress**: Include print statements showing what's happening
7. **Compare approaches**: Show baseline vs. new method when applicable

### Resource Considerations

1. **Device detection**: Always support both CPU and GPU
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

2. **Memory management**: Use `torch.no_grad()` for inference
3. **Model size**: Default to smaller models (GPT-2) for demos
4. **Graceful degradation**: Provide fallbacks when resources unavailable

### Common Pitfalls to Avoid

❌ **Don't**:
- Create files without reading existing patterns
- Skip type hints or docstrings
- Use inconsistent naming (camelCase vs snake_case)
- Put reusable code directly in notebooks
- Create demos without clear output/comparison
- Write READMEs without following the structure
- Commit notebooks with large outputs (clear before committing)

✅ **Do**:
- Study existing implementations first
- Follow established patterns and conventions
- Use comprehensive documentation
- Separate reusable code (`src/`) from demos (`notebooks/`)
- Create clear, progressive demonstrations
- Include paper citations and references
- Test demos before committing

### Git Workflow

1. **Branch naming**: Follow pattern `claude/add-[feature]-[ID]`
2. **Commit messages**: Clear, descriptive messages
3. **Before committing**:
   - Clear notebook outputs if large
   - Ensure demos run successfully
   - Update relevant README files
4. **Push to correct branch**: Verify branch name matches required pattern

---

## Development Checklist

When creating a new project or major feature, use this checklist:

### Setup Phase
- [ ] Created appropriate directory structure (Pattern 1, 2, or 3)
- [ ] Added `pyproject.toml` with dependencies
- [ ] Ran `uv sync` to install dependencies
- [ ] Created initial file structure

### Implementation Phase
- [ ] Added module docstring with paper reference (if applicable)
- [ ] Implemented class with comprehensive docstrings
- [ ] Used type hints for all methods
- [ ] Followed import organization pattern
- [ ] Used proper resource management (context managers, device detection)

### Documentation Phase
- [ ] Created README.md with standard sections
- [ ] Added docstrings to all public methods
- [ ] Included code examples in docstrings
- [ ] Added inline comments for complex logic
- [ ] Updated parent directory README if needed

### Demo/Testing Phase
- [ ] Created `quick_demo.py` for basic functionality
- [ ] Created `harder_demo.py` for advanced scenarios (if applicable)
- [ ] Created interactive notebook in `notebooks/`
- [ ] Tested all demos successfully
- [ ] Added visualizations where appropriate

### Finalization Phase
- [ ] Cleared notebook outputs (if large)
- [ ] Verified all dependencies in `pyproject.toml`
- [ ] Checked that code follows established patterns
- [ ] Reviewed documentation for clarity
- [ ] Ready to commit and push

---

## Questions & Troubleshooting

### "Which directory structure should I use?"

- **Single notebook exploration**: Pattern 1 (simple)
- **Paper implementation with reusable code**: Pattern 2 (implementation)
- **Multiple related experiments**: Pattern 3 (collection)

### "What dependencies should I include?"

Check `experimental-notebooks/maths-for-dl/pyproject.toml` for the most comprehensive list. At minimum:
- `ipykernel`, `jupyter` for notebooks
- `torch`, `transformers` for ML
- `matplotlib`, `seaborn` for visualization
- `numpy` for numerical computing

### "How detailed should documentation be?"

Follow the **progressive disclosure** principle:
- **Module docstring**: High-level overview
- **Class docstring**: Key concepts and insights
- **Method docstring**: Specific functionality
- **README**: Comprehensive explanation with examples

### "Should I create a demo script or notebook?"

**Both, if possible**:
- **Demo scripts** (`quick_demo.py`): Quick verification, CI/CD friendly
- **Notebooks**: Interactive exploration, visualizations, education

### "How do I run the code?"

```bash
# For Python scripts
uv run python path/to/script.py

# For notebooks
uv run jupyter notebook path/to/notebook.ipynb

# UV handles environment and dependencies automatically
```

---

## Conclusion

This repository emphasizes:
1. **Clear organization**: Structured by topic with consistent patterns
2. **Comprehensive documentation**: At module, class, and method levels
3. **Educational focus**: Explain concepts, not just code
4. **Research implementation**: Paper-based work with proper citations
5. **Experimentation-friendly**: Easy to add new projects and ideas

When in doubt, look at `experimental-notebooks/rags/infinite-retreival/` as a reference implementation that demonstrates all best practices.

---

**For updates or questions about this guide, refer to the git history or check the README.md files throughout the repository.**
