# Conversational Financial QA Prototype

## Executive Summary
This prototype demonstrates a dual-mode financial question-answering system using the [ConvFinQA](https://github.com/czyssrs/ConvFinQA) dataset, capable of extracting insights from financial documents, including tables. It supports two inference modes:

- **Cloud Mode**: DeepSeek LLM with ~75% accuracy and 3-4s response time.
- **Local Mode**: GGUF-quantised Mistral-7B running on CPU with ~50% accuracy and 20-30s latency.

The system employs a unified interface with automatic context formatting and flexible prompting. Evaluation using Exact Match (EM) with 5% tolerance ensures precise numerical accuracy while preventing hallucinations. Key challenges include multi-step reasoning errors and implicit temporal references, with future improvements focused on fine-tuning Mistral and integrating RAG for enhanced document retrieval.

## Key Features
✅ Dual inference modes ([DeepSeek](https://platform.deepseek.com/) / [Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF))

✅ Common interface for both models

✅ Flexible prompting allowing to add more examples or update the prompts to address various types of questions.

✅ Automatic context formatting  

✅ Metric-driven performance analysis

## Implementation Approach
1. **Hybrid Architecture**: Unified interface for DeepSeek/Mistral models; a balance between accuracy (DeepSeek) and privacy (Mistral).
2. **Local Mode**:
   - GGUF-quantised Mistral-7B
   - llama.cpp CPU optimisation
   - 4-bit quantisation (6GB RAM usage)
3. **Cloud Mode**:
   - DeepSeek API integration
   - Token-based cost tracking
4. **Metric Selection**
   We chose Exact Match (EM) with 5% accuracy tolerance for the numerical values returned by the LLM. E.g. if the LLM responded with the value `80.34` and the expected answer is `80.3`, the LLM response is calculated as EM, since `80.4` differs from `80.3` by 4.98%.

## Setup

### Prerequisites
- Python 3.10+
- 8GB RAM (for local mode)
- [DeepSeek API key](https://platform.deepseek.com/) (cloud mode)

### Installation
```bash
git clone https://github.com/yourusername/conv-fin-qa
cd conv-fin-qa

# Base dependencies
pip install -r requirements.txt

# Local mode extras (CPU-only)
pip install llama-cpp-python

# Get Mistral-7B quantised model (Q4_K_M)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -O models/mistral-7b-q4.gguf
```

## Configuration

1. Create `.env` file in project root
2. Populate it with your paths/credentials.
3. Directory structure should match the project structure:
```text
.
├── clients/
│   └── main_client.json        # Client whitelist/access rules
├── models/                     # (For local mode)
│   └── mistral-7b...gguf       # Quantised Mistral model
├── data/
├── src/
│   └──conv_fin_qa/             # Core Python package
|       ├── __main__.py         # entry point
|           ...
├── .env                        # Environment configuration
└── requirements.txt
```

## Usage

### Cloud Mode (DeepSeek)
```bash
export DEEPSEEK_API_KEY=your_key_here
python -m conv_fin_qa --mode deepseek
```

### Local Mode (Mistral CPU)
```bash
python -m conv_fin_qa --mode mistral
```

Example session:
```bash
Please enter document ID: dummy/page_114

Here is the context for the document dummy/page_114:
| Metric                        | 2008     | 2007     |
| Diluted Earnings Per Share    | 0.75     | 1.09     |

Question: What is the percentage change in diluted EPS from 2007 to 2008?
Answer: -31.19%
```

## Evaluation Methodology
We use **Exact Match (EM) with a 5% tolerance** as our primary metric because:
- Financial answers require precise numerical correctness but often involve rounding.
- The 5% tolerance accounts for small numerical discrepancies (e.g., 80.34 vs. 80.3), ensuring reasonable matches while preventing hallucinations.

## Key Findings
- **Accuracy Trade-off:** DeepSeek achieves ~75% accuracy due to its larger model size and cloud-based resources but incurs API costs. Mistral, with ~50% accuracy, is a privacy-focused, offline alternative.
- **Latency vs. Cost:** DeepSeek provides faster responses (3-4s) but costs ~$0.10/1000 questions, while Mistral takes 20-30s but operates cost-free locally.
- **Failure Patterns:** Both models struggle with multi-step reasoning and implicit temporal references, with Mistral exhibiting a higher error rate in these scenarios.

## Performance Comparison
| Metric               | DeepSeek (Cloud) | Mistral (Local CPU) |
|----------------------|------------------|---------------------|
| Latency per question  | 3-4s             | 20-30s               |
| Cost                  | ~$0.10/1000 Qs  | Free                |
| Accuracy*             | 75% EM           | 50% EM              |

*Measured on 100-sample validation set

## Limitations
We encountered the following limitations during development:
- **Table Parsing Dependency:** Accuracy is highly dependent on the quality of table parsing, leading to errors when headers are inconsistent or misaligned.
- **Temporal Reference Ambiguity:** Difficulty in resolving implicit temporal expressions such as “last year” or “previous quarter” without explicit date mapping.
- **Multi-step Reasoning Errors:** Higher error rates on questions requiring chained operations, especially in Mistral due to its smaller context window.


## Future Improvements
- **Fine-tuning Mistral:** Training Mistral on financial documents to enhance domain-specific accuracy and improve performance on numerical reasoning tasks.
- **Retrieval-Augmented Generation (RAG):** Integrating RAG to improve document retrieval and better handle multi-step context understanding.
- **Verbosity Control with Prompt Refinement:** Enhancing prompts by providing contextual examples and using chain-of-thought prompting to reduce over-explanations and improve concise, accurate answers.
