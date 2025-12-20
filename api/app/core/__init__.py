"""Core services for JR AutoRAG backend."""

from .ab_testing import ABTestingFramework, Experiment, ExperimentVariant
from .batch_processor import BatchProcessor, BatchResult
from .cache import CacheManager, EmbeddingCache, QueryCache, get_cache_manager
from .chunking import Chunk, ChunkingStrategy, FixedChunker, RecursiveChunker, SemanticChunker, get_chunker
from .compression import CitedPassage, CompressedContext, ContextCompressor
from .config_store import ConfigStore
from .documents import DocumentStore
from .document_processors import DocumentProcessor, ExtractedTable, ListExtractor, TableExtractor
from .evaluator import EvaluationScore, HeuristicEvaluator, LLMJudge
from .gatherer import Gatherer
from .hierarchy import DocumentTree, HierarchicalRetriever, HierarchyBuilder, HierarchyNode
from .hybrid_retrieval import HybridConfig, HybridRetrievalEngine, RetrievalResult
from .ingest import IngestPipeline
from .memory import ConversationContext, ConversationMemory, ConversationTurn
from .metadata_enricher import DocumentMetadata, MetadataEnricher
from .metrics import EvaluationResult, LatencyMetrics, MetricsCalculator, MetricsStore, RetrievalMetrics
from .multimodal import ExtractedImage, ImageType, MultimodalProcessor
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import ProviderFactory
from .reflection import AnswerQuality, ReflectionResult, SelfReflector
from .retrieval import RetrievalEngine
from .router import QueryRouter, RetrievalStrategy, RoutingDecision
from .smart_planner import PlanStep, QueryAnalysis, QueryType, RetrievalPlan, SmartPlanner
from .telemetry import TelemetryStore
from .tools import CalculatorTool, DateTimeTool, Tool, ToolRegistry, ToolResult
from .tracing import DetailedTrace, TraceSpan, Tracer, get_tracer
from .vector_store import ChromaVectorStore, InMemoryVectorStore, VectorStore, get_vector_store

__all__ = [
    "ConfigStore",
    "DocumentStore",
    "Planner",
    "ProviderFactory",
    "RetrievalEngine",
    "IngestPipeline",
    "Gatherer",
    "Orchestrator",
    "TelemetryStore",
    # Phase 1: Hybrid retrieval
    "Chunk",
    "ChunkingStrategy",
    "FixedChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "get_chunker",
    "HybridRetrievalEngine",
    "HybridConfig",
    "RetrievalResult",
    # Phase 2: Smart planning and compression
    "SmartPlanner",
    "QueryType",
    "QueryAnalysis",
    "PlanStep",
    "RetrievalPlan",
    "ContextCompressor",
    "CompressedContext",
    "CitedPassage",
    # Phase 3: Agentic components
    "QueryRouter",
    "RetrievalStrategy",
    "RoutingDecision",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "DateTimeTool",
    "SelfReflector",
    "ReflectionResult",
    "AnswerQuality",
    "ConversationMemory",
    "ConversationTurn",
    "ConversationContext",
    # Phase 4: Document processing
    "DocumentProcessor",
    "TableExtractor",
    "ListExtractor",
    "ExtractedTable",
    "MetadataEnricher",
    "DocumentMetadata",
    "HierarchyBuilder",
    "DocumentTree",
    "HierarchyNode",
    "HierarchicalRetriever",
    "MultimodalProcessor",
    "ExtractedImage",
    "ImageType",
    # Phase 5: Evaluation and observability
    "MetricsStore",
    "MetricsCalculator",
    "RetrievalMetrics",
    "LatencyMetrics",
    "EvaluationResult",
    "ABTestingFramework",
    "Experiment",
    "ExperimentVariant",
    "LLMJudge",
    "HeuristicEvaluator",
    "EvaluationScore",
    "Tracer",
    "TraceSpan",
    "DetailedTrace",
    "get_tracer",
    # Phase 6: Production readiness
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "get_vector_store",
    "CacheManager",
    "EmbeddingCache",
    "QueryCache",
    "get_cache_manager",
    "BatchProcessor",
    "BatchResult",
]
