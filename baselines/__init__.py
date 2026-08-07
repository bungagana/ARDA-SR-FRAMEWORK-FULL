from .base import BasePipeline
from .llm_only import LLMOnlyPipeline
from .standard_rag import StandardRAGPipeline
from .hybrid_rag import HybridRAGPipeline
from .hyde_rag import HyDERAGPipeline
from .adaptive_rag import AdaptiveRAGPipeline
from .crag import CRAGPipeline
from .react import ReActPipeline
from .selfrag import SelfRAGPipeline
from .flare import FLAREPipeline
from .ircot import IRCoTPipeline

ALL_BASELINES = {
    "llm_only":    LLMOnlyPipeline,
    "standard_rag": StandardRAGPipeline,
    "hybrid_rag":  HybridRAGPipeline,
    "hyde_rag":    HyDERAGPipeline,
    "adaptive_rag": AdaptiveRAGPipeline,
    "crag":        CRAGPipeline,
    "react":       ReActPipeline,
    "selfrag":     SelfRAGPipeline,
    "flare":       FLAREPipeline,
    "ircot":       IRCoTPipeline,
}

__all__ = ["BasePipeline", "ALL_BASELINES"]
