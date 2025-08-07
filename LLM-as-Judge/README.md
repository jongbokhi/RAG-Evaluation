# RAG Evaluation Framework with LLM-as-Judge

Production-ready evaluation system for RAG applications using LLM-as-Judge methodology with automated synthetic dataset generation.

## Overview

Comprehensive RAG evaluation framework that generates synthetic test datasets from PDF documents using RAGAS, then evaluates RAG system performance across four metrics using GPT-4.1-mini as an automated judge.

## Architecture

**Data Preparation:**
```
PDF Documents → RAGAS → Synthetic Dataset (Questions + Reference Answers)
```

**Evaluation Pipeline:**
```
LangSmith Dataset → ask_question() → RAG Answer + Retrieved Docs
                                           ↓
                                    LLM-as-Judge Evaluators (GPT-4.1-mini)
                                           ↓
                                   Correctness | Groundedness | Helpfulness | Retrieval Relevance
```

**Tech Stack**: 
- LangChain 0.3.26
- LangSmith 0.4.4  
- RAGAS 0.3.0
- FAISS 1.11.0
- OpenAI GPT-4.1-mini

## Evaluation Metrics  
**[reference link](https://github.com/langchain-ai/openevals?tab=readme-ov-file#correctness-rag)**
1. **Correctness** - Measures how accurate a generated answer is to a ground-truth answer
2. **Groundedness** - Measures the extent that the generated response agrees with the retrieved context
3. **Helpfulness** - Measures how well the generated response addresses the initial user input
4. **Retrieval Relevance** - Measures how relevant retrieved context is to an input query

## Project Structure

```
LLM-as-Judge/              # LLM-as-Judge evaluation system
├── LLM-as-judge.ipynb     # Main evaluation pipeline
├── RAGPDF.py              # RAG system implementation
├── evaluator/             # 4-metric evaluation modules
│   ├── correctness.py
│   ├── groundedness.py
│   ├── helpfulness.py
│   └── retrievalrelevance.py
RAGAS/                                    # RAGAS-based evaluation & dataset generation
├── RAG-Evaluation-Using RAGAS.ipynb      # RAGAS evaluation pipeline
└── Sythetic dataset - Using RAGAS.ipynb  # Synthetic data generation
```
