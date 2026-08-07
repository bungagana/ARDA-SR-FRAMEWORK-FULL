# ARDA-SR

**Adaptive Retrieval-Decision Architecture with Dual-Draft Arbitration and Scenario Reasoning**

This is the research codebase behind ARDA-SR — a RAG framework for public-sector QA over
Indonesian transmigration/regional-governance documents, built from three coordinated modules.
The system and its experiments were developed here first; the paper *"ARDA-SR: Entropy-Based
Routing and Dual-Draft Arbitration for False Refusal Reduction and Scenario Reasoning in
Government Question Answering"* reports the results this codebase produces.

| Module | Role |
|---|---|
| **AQR** — Adaptive Query Router | Entropy-based zero-shot routing into 4 response modes (m1 direct / m2 retrieval / m3 hybrid / m4 scenario) |
| **DDA** — Dual-Draft Arbitrator | Generates a parametric draft and a retrieval-grounded draft, arbitrates between them with a 4-dimensional utility function to reduce false refusals |
| **SR** — Scenario Reasoning | Structures policy-scenario queries into comparable alternatives, scored by expected utility |

This repository contains everything needed to rebuild the knowledge base, regenerate the QA test
set, rerun all 10 baselines + ARDA-SR, and reproduce the ablation study, the cross-backbone
robustness check, and the sensitivity analysis — the full experiment program this codebase runs,
which the paper then reports on.

## Requirements

- Python 3.10+
- API keys for **Gemini**, **Claude (Anthropic)**, and **OpenAI** (see below) — the pipeline calls
  all three by design, not interchangeably
- ~1 GB free disk for the embedding model + FAISS index (downloaded/built on first run)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## API keys

```bash
# Windows (PowerShell)
copy .env.example .env
# macOS/Linux
cp .env.example .env
```

Then fill in `.env`:

| Variable | Used for | Required |
|---|---|---|
| `GEMINI_API_KEY` | Generation, AQR routing, retrieval-grounded drafts, DDA, SR, all baselines | Yes |
| `ANTHROPIC_API_KEY` | QA ground-truth dataset generation (`claude-haiku-4-5`) | Yes |
| `OPENAI_API_KEY` | Independent LLM-as-judge (`gpt-5.4-mini`) — deliberately a different model family from the generator, to avoid self-evaluation bias | Yes |
| `HF_TOKEN` | Hosted Hugging Face backbones for `cross/run_cross_backbone.py` only | Only for the cross-backbone check |

On Windows, run scripts with UTF-8 forced (source documents include non-ASCII text):

```powershell
$env:PYTHONUTF8="1"
```

## Running the pipeline

Run in order. Each step reads the previous step's output; steps 3–5 are safe to re-run/resume.

```bash
# 1. Build the knowledge base (extract, chunk, embed, index)
python 01_build_kb.py
# -> kb/faiss_index/, kb/bm25_index.pkl, kb/chunks.json

# 2. Generate the 1,000-item QA test set (5 categories x 200) with Claude Haiku
python 02_generate_qa_claude.py
# -> data/qa_dataset.json, data/qa_dataset.csv, data/qa_generation_stats.json

# 2.5 (optional but recommended) Derive should_be_answerable labels from
#     corpus evidence via embedding similarity (Section 2.4.1, Stage 1).
#     Stage 2 (3 human annotators) is a manual review step outside this repo.
python verify_answerability.py --threshold 0.45

# 3. Run all 10 baselines + ARDA-SR on the full test set
python 03_run_experiment.py
#   --method arda_sr     run a single method only
#   --smoke               sanity check on the first 10 queries
#   --skip-judge           skip the GPT LLM-judge call (faster, no answer-quality scores)
# -> results/{method}_results.json, results/all_metrics.json

# 4. Ablation study (V0 Base RAG -> +AQR -> +Hybrid retrieval -> +DDA -> +SR)
python 04_ablation.py
# -> results/ablation_metrics.json

# 5. Statistical analysis, tables, and figures
python 05_analyze_results.py
# -> outputs/ (summary tables, significance tests, figures)
```

Optional robustness/analysis scripts:

```bash
# Cross-backbone check on smaller open models (Qwen2.5-1.5B, Phi-3-mini) — see cross/README.md
python cross/run_cross_backbone.py --smoke

# Sensitivity analysis: perturb delta / beta / alpha / lambda around their
# fixed defaults and check metric stability (Fig. 6)
python Sensitivity_analysis.py --dry-run   # preview the plan, no API calls
python Sensitivity_analysis.py
```

## Default parameters

These are the fixed values this codebase uses to produce its results (`config.py`), also reported
in the paper's methodology section:

| Parameter | Value | Meaning |
|---|---:|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation / routing backbone |
| `CLAUDE_QA_MODEL` | `claude-haiku-4-5` | QA dataset generation |
| `GPT_JUDGE_MODEL` | `gpt-5.4-mini` | Independent LLM-as-judge |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 820 / 80 | Target characters per chunk |
| `ENTROPY_THRESHOLD` (τ_H) | 1.05 bits | AQR hybrid-routing threshold (52.5% of H_max = log2(4) = 2.0 bits) |
| `HYBRID_ALPHA` (α) | 0.60 | Dense-vs-BM25 weight in hybrid retrieval |
| `TOP_K` | 5 | Retrieved evidence chunks |
| `DDA` β weights | (0.30, 0.25, 0.20, 0.25) | Relevance, faithfulness, coverage, risk |
| `DDA_DECISION_MARGIN` (δ) | 0.05 | Utility margin before preferring one draft over merging |
| `SR_LAMBDA` (λ_SR) | 0.50 | Risk-aversion coefficient in expected-utility scoring |

## Repository structure

```text
PDF/
├── arda_sr/               # AQR, DDA, SR, hybrid retrieval, end-to-end pipeline
├── baselines/              # 10 baseline RAG methods (Standard/Hybrid/HyDE/Adaptive/CRAG/
│                            #   ReAct/Self-RAG/FLARE/IRCoT/LLM-only)
├── evaluation/             # Metrics, LLM-as-judge, statistical tests
├── utils/                  # Document parsing, KB building, LLM clients (Gemini/Claude/GPT)
├── cross/                  # Cross-backbone robustness check (see cross/README.md)
├── figures/                # Figure-generation scripts
├── kb/                     # Built knowledge base (FAISS + BM25 index) — generated, gitignored
├── data/                   # QA test set (tracked — the fixed set used for all reported results)
├── results/                # Per-method result JSONs + aggregated metrics (tracked — evidence
│                            #   backing the paper's tables)
├── outputs/                # Generated tables/figures from 05_analyze_results.py
├── documents.zip           # Source corpus (PDF/DOCX government documents)
├── 01_build_kb.py … 05_analyze_results.py
├── Sensitivity_analysis.py
├── verify_answerability.py
├── config.py                # All fixed hyperparameters and model names
└── requirements.txt
```

## Inspecting results without re-running everything

`results/*.json` and `outputs/*` are committed to this repository — the actual output this
codebase produced, corresponding to the paper's Table 5 (main comparison), Table 6 (ablation),
and the statistical tests — so they can be inspected directly without spending API budget.
Re-running `01`–`05` from scratch will call the Gemini/Claude/OpenAI APIs for all 1,000 test
queries across 11 methods and will incur real cost and take a non-trivial amount of time — use
`--smoke` first to sanity-check your setup.
