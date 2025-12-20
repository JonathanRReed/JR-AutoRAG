"""Core services for JR AutoRAG backend."""

from .config_store import ConfigStore
from .documents import DocumentStore
from .gatherer import Gatherer
from .ingest import IngestPipeline
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import ProviderFactory
from .retrieval import RetrievalEngine
from .telemetry import TelemetryStore

# Phase 1: Hybrid retrieval
from .chunking import (
    Chunk,
    ChunkingStrategy,
    FixedChunker,
    SemanticChunker,
    RecursiveChunker,
    get_chunker,
)
from .hybrid_retrieval import (
    HybridRetrievalEngine,
    HybridConfig,
    RetrievalResult,
)

# Phase 2: Smart planning and compression
from .smart_planner import (
    SmartPlanner,
    QueryType,
    QueryAnalysis,
    PlanStep,
    RetrievalPlan,
)
from .compression import (
    ContextCompressor,
    CompressedContext,
    CitedPassage,
)

# Phase 3: Agentic components
from .router import (
    QueryRouter,
    RetrievalStrategy,
    RoutingDecision,
)
from .tools import (
    Tool,
    ToolResult,
    ToolRegistry,
    CalculatorTool,
    DateTimeTool,
)
from .reflection import (
    SelfReflector,
    ReflectionResult,
    AnswerQuality,
)
from .memory import (
    ConversationMemory,
    ConversationTurn,
    ConversationContext,
)

# Phase 4: Document processing
from .document_processors import (
    DocumentProcessor,
    TableExtractor,
    ListExtractor,
    ExtractedTable,
)
from .metadata_enricher import (
    MetadataEnricher,
    DocumentMetadata,
)
from .hierarchy import (
    HierarchyBuilder,
    DocumentTree,
    HierarchyNode,
    HierarchicalRetriever,
)
from .multimodal import (
    MultimodalProcessor,
    ExtractedImage,
    ImageType,
)

# Phase 5: Evaluation and observability
from .metrics import (
    MetricsStore,
    MetricsCalculator,
    RetrievalMetrics,
    LatencyMetrics,
    EvaluationResult,
)
from .ab_testing import (
    ABTestingFramework,
    Experiment,
    ExperimentVariant,
)
from .evaluator import (
    LLMJudge,
    HeuristicEvaluator,
    EvaluationScore,
)
from .tracing import (
    Tracer,
    TraceSpan,
    DetailedTrace,
    get_tracer,
)

# Phase 6: Production readiness
from .vector_store import (
    VectorStore,
    InMemoryVectorStore,
    ChromaVectorStore,
    get_vector_store,
)
from .cache import (
    CacheManager,
    EmbeddingCache,
    QueryCache,
    get_cache_manager,
)
from .batch_processor import (
    BatchProcessor,
    BatchResult,
)

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
