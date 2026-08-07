"""Central configuration for all ARDA-SR experiments."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
KB_DIR      = BASE_DIR / "kb"
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_ZIP    = BASE_DIR / "documents.zip"

for d in [KB_DIR, DATA_DIR, RESULTS_DIR, OUTPUTS_DIR]:
    d.mkdir(exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = "gemini-2.5-flash"   # generation & routing — per paper Section 2.4.3 backbone
# NOTE: judging is done by GPT_JUDGE_MODEL (see below) via GPTJudgeClient, not Gemini —
# there is no separate Gemini judge model anymore (removed dead GEMINI_JUDGE_MODEL constant).
TEMPERATURE         = 0.0                  # deterministic generation
MAX_OUTPUT_TOKENS   = 2048
REQUEST_DELAY_S     = 1.2                  # seconds between API calls (rate limit)
MAX_RETRIES         = 5

# ── Claude (QA generation only — per paper Section 2.4.1: "claude-haiku-4-5") ──
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_QA_MODEL     = "claude-haiku-4-5"

# ── OpenAI (LLM-as-judge only — replaces Gemini-as-its-own-judge conflict of
# interest; generation/retrieval/DDA/SR all stay on Gemini) ──────────────────
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
GPT_JUDGE_MODEL      = "gpt-5.4-mini"
GPT_JUDGE_REASONING_EFFORT = "low"  # judging is a short structured-JSON scoring task

# ── Embeddings ────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM    = 384

# ── Chunking ──────────────────────────────────────────────────────────────

CHUNK_SIZE    = 820
CHUNK_OVERLAP = 80

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K          = 5
HYBRID_ALPHA   = 0.6   # weight for dense (semantic); (1-α) for BM25

# ── AQR ──────────────────────────────────────────────────────────────────
ENTROPY_THRESHOLD = 1.05  # τ_H; above this → hybrid path

# ── DDA ──────────────────────────────────────────────────────────────────
DDA_BETA_SEARCH = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
DDA_VALIDATION_SPLIT = 0.2   # for separate development data only; never tune on final test data
DDA_DECISION_MARGIN = 0.05   # utility gap required before preferring the retrieval draft

# ── SR ───────────────────────────────────────────────────────────────────
SR_LAMBDA        = 0.5    # risk-aversion parameter λ_SR
SR_NUM_SCENARIOS = 3      # scenarios to generate per policy query

# ── QA Generation ─────────────────────────────────────────────────────────
QA_CATEGORIES = ["DK", "FR", "CR", "AR", "PS"]
QA_TARGET_PER_CATEGORY  = 200
QA_GENERATE_PER_CATEGORY = 250   # generate extra for filtering

QA_CATEGORY_DESC = {
    "DK": "Direct Knowledge — conceptual questions answerable without document retrieval",
    "FR": "Factual Retrieval — single-document fact lookups (statistics, specific area data)",
    "CR": "Complex Reasoning — multi-hop questions requiring info from 2+ documents",
    "AR": "Ambiguous Routing — borderline queries where retrieval need is non-obvious",
    "PS": "Policy Scenario — multi-alternative policy decisions requiring trade-off analysis",
}

# ── Evaluation ────────────────────────────────────────────────────────────
JUDGE_SCALE_MAX = 5        # Likert scale 1–5
BOOTSTRAP_SAMPLES = 1000
WILCOXON_ALPHA    = 0.05

# ── IRCoT ─────────────────────────────────────────────────────────────────
IRCOT_MAX_ITERATIONS = 3

# ── ReAct ─────────────────────────────────────────────────────────────────
REACT_MAX_STEPS = 5

# ── Experiment ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

BASELINE_NAMES = [
    "llm_only",
    "standard_rag",
    "hybrid_rag",
    "hyde_rag",
    "adaptive_rag",
    "crag",
    "react",
    "selfrag",
    "flare",
    "ircot",
    "arda_sr",
]

ABLATION_VARIANTS = ["V0_base_rag", "V1_aqr", "V2_hybrid", "V3_dda", "V4_sr"]
