# Feynman

*Learn any technical topic from first principles — taught by research papers, not blog posts.*

Feynman is an agentic research assistant that takes you from zero to deep understanding of any technical topic. You type a topic, the agent autonomously fetches and reads relevant papers from ArXiv, builds an ordered concept curriculum, then teaches you interactively — checking genuine comprehension at each step before moving on, and re-explaining with a different analogy if you don't get it.

Inspired by Richard Feynman's principle: **if you can't explain it simply, you don't understand it yet.**

---

## Demo

> *Demo GIF coming after first recorded session — record with `brew install licecap`*

---

## Architecture

```mermaid
flowchart TD
    A([🧑 User enters topic]) --> B

    subgraph Turn 1 — Research
        B[research_node\nFetch ArXiv papers\nEnrich with Semantic Scholar citations\nParse PDFs → FAISS index\nExtract ordered concept curriculum]
    end

    B --> C([❓ Assessment questions sent to user])
    C --> D

    subgraph Turn 2 — Calibrate
        D[assess_node\nInfer user level\nbeginner · intermediate · advanced] --> E
        E[explain_node\nRAG retrieval from FAISS\nFeynman-style explanation\nCites papers by title\nEnds with Socratic question]
    end

    E --> F([💬 Explanation + question sent to user])
    F --> G

    subgraph Turn 3+ — Teach
        G{User message intent}
        G -->|Answers question| H[check_node\nEvaluate understanding\nnot just recall]
        G -->|go deeper / tell me more| I[drill_node\nFetch extra targeted papers\nDeep paper-grounded explanation]
        H -->|understood| J{More concepts?}
        H -->|gaps remain| E
        J -->|yes| E
        J -->|no| K[summarize_node\nFeynman summary\nthe final test]
        I --> F
    end

    K --> L([✅ Session complete])
```

---

## Eval Results

============================================================
TOPIC:                  Vision Transformers
CONCEPTS EVALUATED:     6
AVG FEYNMAN SCORE:      4.00/5  (target ≥ 3.8)
AVG HALLUCINATION RATE: 18.9%  (target < 5%)
============================================================

| Concept | Feynman Score | Hallucination Rate |
|---|---|---|
| Vision Transformer (ViT) Architecture | 4/5 | 40.0% |
| Self-Supervised Learning (SSL) in Vision | 4/5 | 26.3% |
| Attention Mechanisms and Locality Bias | 4/5 | 7.1% |
| Computational Efficiency and Model Scaling | 4/5 | 12.5% |
| Cross-Domain and Multi-Modal Adaptation | 4/5 | 9.5% |
| Task-Specific Architectural Optimization | 4/5 | 17.6% |

**Targets:** avg Feynman score ≥ 3.8 (Achieved) · hallucination rate < 5% (WIP)

---

## Design Decisions

### Why LangGraph over a simple prompt chain
A chain is linear: A → B → C → done. Teaching requires cycles: explain → check → re-explain → check again. LangGraph's `StateGraph` models this as a first-class directed graph with conditional edges, so the retry loop is structural rather than bolted on with `if` statements. The typed `AgentState` TypedDict also makes debugging explicit — you can inspect exactly what changed after each node.

### Why ArXiv + Semantic Scholar over web search
Web search returns a mix of blog posts, Stack Overflow answers, YouTube transcripts, and marketing copy — high noise, unverifiable claims, no ground truth. ArXiv papers are peer-reviewed, citable, and written by the people who actually built the things being explained. Semantic Scholar adds citation counts, which are a reliable proxy for foundational importance: a paper cited 3,000 times is almost certainly more important to understand than one cited 12 times. The constraint also makes the product's claims auditable — every explanation must cite a paper by title.

### Why sentence-transformers locally vs. API embeddings
`all-MiniLM-L6-v2` runs on-device (MPS on Apple Silicon, CPU elsewhere) with no API call, no latency, and no per-token cost. At the scale of a single session (~1,200 chunks from 10 papers), local inference is instantaneous. OpenAI or Gemini embeddings would add ~$0.01/session in cost, an extra network round-trip per search, and a hard dependency on an external service that would break HuggingFace deployment if the key weren't set. The model is small enough (80 MB) to embed directly in the Space.

### The check_node retry loop — measuring understanding
"Understood" is evaluated by asking the LLM whether the user's answer demonstrates that they can *apply or extend* the concept, not just recall the wording used in the explanation. The prompt explicitly says "credit correct intuition even with imperfect terminology." This is the core Feynman mechanic: you only advance when you've actually understood, not when you've produced the right-sounding words. When the check fails, `reexplain=True` is set in state, and `explain_node` switches to the `RE_EXPLAIN_CONCEPT` prompt which requires a completely different analogy — not a slower repetition of the same framing.

---

## Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph `StateGraph` |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Paper retrieval | ArXiv API + Semantic Scholar API |
| PDF parsing | PyMuPDF (`fitz`) |
| Token counting | tiktoken |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (MPS) |
| Vector store | FAISS `IndexFlatIP` |
| Interface | Gradio 6 |
| Deployment | HuggingFace Spaces (Docker) |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/Feynman_Researcher
cd Feynman_Researcher

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your Google API key to .env:  GOOGLE_API_KEY=AIza...

python app.py
# Open http://localhost:7860
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

---

## Limitations

**Works well**
- Topics with a large ArXiv presence: deep learning, NLP, computer vision, RL
- Foundational ML papers (transformers, diffusion, GNNs) — well-cited, well-parsed PDFs
- Intermediate learners with some ML background — the calibration is most accurate here

**Works less well**
- Very new topics (< 6 months old) — too few papers to build a solid curriculum
- Highly mathematical topics — PDF parsing loses LaTeX equations, so proofs are missing from context
- Interdisciplinary topics (e.g. "ML for drug discovery") — concept ordering is less reliable when papers span very different fields
- Two-column PDF layouts — PyMuPDF extracts them as garbled text; section detection fails

---

## Project Structure

```
feynman/
├── agent/
│   ├── graph.py        # LangGraph StateGraph definition
│   ├── nodes.py        # 6 node functions (research/assess/explain/check/drill/summarize)
│   ├── prompts.py      # All prompt templates
│   └── state.py        # AgentState TypedDict
├── retrieval/
│   ├── arxiv_client.py      # ArXiv API wrapper
│   ├── semantic_scholar.py  # Semantic Scholar enrichment
│   ├── pdf_parser.py        # PyMuPDF + tiktoken chunking
│   └── vector_store.py      # FAISS + sentence-transformers
└── eval/
    └── citation_checker.py  # Hallucination rate measurement
app.py              # Gradio interface
notebooks/
└── eval_feynman.ipynb  # Feynman score + hallucination eval
```
