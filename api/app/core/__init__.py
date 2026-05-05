"""Core services for JR AutoRAG backend."""

from .ab_testing import ABTestingFramework, Experiment, ExperimentVariant

# v2.0 Abstention rules
from .abstention import AbstentionConfig, AbstentionReason, AbstentionResult, AbstentionRules, get_abstention_rules
from .adaptive_gate import AdaptiveGate, GateDecision, GateResult
from .artifact_builder import ArtifactBuilder, ArtifactState, ArtifactStatus, BuildProgress, get_artifact_builder
from .audit import AuditAction, AuditEntry, AuditLog, get_audit_log

# Enterprise modules
from .auth import APIKey, APIKeyAuth, get_auth
from .auto_weights import AutoHybridWeights, AutoWeightConfig, WeightProfile, get_auto_weights
from .batch_processor import BatchProcessor, BatchResult

# v2 Binary Quantization modules
from .binary_quantization import (
    BQ_VERSION,
    BQConfig,
    batch_float32_to_binary,
    binary_to_bits,
    estimate_storage_savings,
    float32_to_binary,
    get_binary_dimension,
    hamming_distance,
    validate_dimension,
)
from .binary_vector_store import (
    IndexStats,
    MilvusChunk,
    MilvusConfig,
    MilvusSearchResult,
    MilvusVectorStore,
    get_milvus_store,
    is_milvus_available,
)
from .bq_hybrid_retrieval import BQHybridRetrievalEngine
from .bq_retrieval import (
    BQRetrievalConfig,
    BQRetrievalService,
    RetrievalDebug,
    RetrievalModeV2,
    RetrievalTimings,
    RetrievedChunk,
    get_bq_retrieval_service,
)

# vNext Expansion modules (Guarantees G1-G4)
from .cache import CacheManager, EmbeddingCache, QueryCache, RetrievalMode, get_cache_manager
from .chunking import Chunk, ChunkingStrategy, FixedChunker, RecursiveChunker, SemanticChunker, get_chunker

# v2.0 Phase 4: Observability & Security
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    get_circuit_breaker,
    get_circuit_breaker_registry,
)
from .citation_formatter import RichCitation, format_rich_context, generate_reference_section, validate_answer_citations
from .citation_verifier import CitationCheck, CitationVerifier, VerificationResult
from .compression import CitedPassage, CompressedContext, ContextCompressor
from .config_migration import V2_NEW_FIELDS, ConfigMigrator, MigrationResult, get_config_migrator, migrate_config
from .config_store import ConfigStore

# v2.0 Phase 3: Ingestion enhancements
from .contextual_enrichment import ContextualEnricher, EnrichedChunk, EnrichmentConfig, get_contextual_enricher
from .cost_tracker import (
    CostEstimate,
    CostLatencyTracker,
    LatencyBreakdown,
    RequestMetrics,
    TokenUsage,
    get_cost_tracker,
)
from .document_processors import DocumentProcessor, ExtractedTable, ListExtractor, TableExtractor
from .documents import DocumentStore
from .evaluator import EvaluationScore, HeuristicEvaluator, LLMJudge
from .evidence_validator import EvidenceValidator, ValidationResult, create_strict_validator

# v2.0 Phase 5: Polish & Docs
from .feature_flags import (
    V2_FEATURE_FLAGS,
    FeatureFlag,
    FeatureFlagRegistry,
    RiskLevel,
    get_feature_flags,
    is_feature_enabled,
)
from .flare import FLAREConfig, FLAREGenerator, FLAREResult, FLAREStep
from .gatherer import Gatherer

# Web search disabled for offline-only operation
# from .web_search import WebSearch, WebResult, WebSearchProvider, get_web_search
from .graph_rag import Community, Entity, EntityType, GraphRAG, Relationship
from .hierarchy import DocumentTree, HierarchicalRetriever, HierarchyBuilder, HierarchyNode
from .hybrid_retrieval import HybridConfig, HybridRetrievalEngine, RetrievalResult
from .ingest import IngestPipeline
from .local_first import BackendResolution, LocalFirstPolicyError, LocalFirstRegistry
from .memory import ConversationContext, ConversationMemory, ConversationTurn
from .metadata_enricher import DocumentMetadata, MetadataEnricher
from .metrics import EvaluationResult, LatencyMetrics, MetricsCalculator, MetricsStore, RetrievalMetrics
from .multi_granularity import (
    GranularChunk,
    GranularityLevel,
    MultiGranularityConfig,
    MultiGranularityIndexer,
    get_multi_granularity_indexer,
)
from .multimodal import ExtractedImage, ImageType, MultimodalProcessor
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import ProviderFactory
from .rate_limiter import RateLimitConfig, RateLimiter, TokenBucket, get_rate_limiter
from .reflection import AnswerQuality, ReflectionResult, SelfReflector
from .retrieval import RetrievalEngine
from .retrieval_cascade import CascadeConfig, CascadeResult, CascadeStage, RetrievalCascade, get_retrieval_cascade
from .retrieval_evaluator import EvaluationResult as RetrievalEvalResult

# SOTA Agentic components
from .retrieval_evaluator import KnowledgeStrip, RetrievalEvaluator, RetrievalVerdict
from .router import QueryRouter, RetrievalStrategy, RoutingDecision

# v2.0 Phase 2: Agentic loops
from .self_rag import CriticResult, SelfRAGConfig, SelfRAGCritic, get_self_rag_critic
from .smart_planner import PlanStep, QueryAnalysis, QueryType, RetrievalPlan, SmartPlanner
from .structured_data_parser import (
    StructuredDataConfig,
    StructuredDataFormat,
    StructuredDataParser,
    StructuredRecord,
    get_structured_data_parser,
)
from .structured_generator import ClaimSlot, ClaimType, StructuredGenerator, StructuredMemo, StructuredPick
from .telemetry import TelemetryStore
from .tools import CalculatorTool, DateTimeTool, Tool, ToolRegistry, ToolResult
from .trace_export import TraceBundle, create_trace_bundle, summarize_steps
from .tracing import DetailedTrace, Tracer, TraceSpan, get_tracer
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
    "BQHybridRetrievalEngine",
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
    "LocalFirstRegistry",
    "LocalFirstPolicyError",
    "BackendResolution",
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
    # v2 Binary Quantization
    "BQConfig",
    "BQ_VERSION",
    "validate_dimension",
    "float32_to_binary",
    "binary_to_bits",
    "hamming_distance",
    "batch_float32_to_binary",
    "get_binary_dimension",
    "estimate_storage_savings",
    "MilvusConfig",
    "MilvusChunk",
    "MilvusSearchResult",
    "MilvusVectorStore",
    "IndexStats",
    "is_milvus_available",
    "get_milvus_store",
    "RetrievalModeV2",
    "RetrievalTimings",
    "RetrievalDebug",
    "RetrievedChunk",
    "BQRetrievalConfig",
    "BQRetrievalService",
    "get_bq_retrieval_service",
    "CacheManager",
    "EmbeddingCache",
    "QueryCache",
    "get_cache_manager",
    "BatchProcessor",
    "BatchResult",
    # SOTA Agentic components
    "RetrievalEvaluator",
    "RetrievalVerdict",
    "RetrievalEvalResult",
    "KnowledgeStrip",
    "AdaptiveGate",
    "GateDecision",
    "GateResult",
    # GraphRAG
    "GraphRAG",
    "Entity",
    "Relationship",
    "Community",
    "EntityType",
    # FLARE
    "FLAREGenerator",
    "FLAREConfig",
    "FLAREResult",
    "FLAREStep",
    # Enterprise modules
    "APIKeyAuth",
    "APIKey",
    "get_auth",
    "AuditLog",
    "AuditEntry",
    "AuditAction",
    "get_audit_log",
    "RateLimiter",
    "TokenBucket",
    "RateLimitConfig",
    "get_rate_limiter",
    # Citation fidelity
    "RichCitation",
    "format_rich_context",
    "generate_reference_section",
    "validate_answer_citations",
    # Evidence validation (10/10)
    "EvidenceValidator",
    "ValidationResult",
    "create_strict_validator",
    # Structured generation
    "ClaimType",
    "ClaimSlot",
    "StructuredPick",
    "StructuredMemo",
    "StructuredGenerator",
    # vNext Expansion (G1-G4 Guarantees)
    "RetrievalMode",
    "CitationVerifier",
    "CitationCheck",
    "VerificationResult",
    "TraceBundle",
    "create_trace_bundle",
    "summarize_steps",
    "ArtifactBuilder",
    "ArtifactStatus",
    "ArtifactState",
    "BuildProgress",
    "get_artifact_builder",
    "AbstentionConfig",
    "AbstentionReason",
    "AbstentionResult",
    "AbstentionRules",
    "get_abstention_rules",
    "AutoHybridWeights",
    "AutoWeightConfig",
    "WeightProfile",
    "get_auto_weights",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    "get_circuit_breaker",
    "get_circuit_breaker_registry",
    "V2_NEW_FIELDS",
    "ConfigMigrator",
    "MigrationResult",
    "get_config_migrator",
    "migrate_config",
    "ContextualEnricher",
    "EnrichedChunk",
    "EnrichmentConfig",
    "get_contextual_enricher",
    "CostEstimate",
    "CostLatencyTracker",
    "LatencyBreakdown",
    "RequestMetrics",
    "TokenUsage",
    "get_cost_tracker",
    "V2_FEATURE_FLAGS",
    "FeatureFlag",
    "FeatureFlagRegistry",
    "RiskLevel",
    "get_feature_flags",
    "is_feature_enabled",
    "GranularChunk",
    "GranularityLevel",
    "MultiGranularityConfig",
    "MultiGranularityIndexer",
    "get_multi_granularity_indexer",
    "CascadeConfig",
    "CascadeResult",
    "CascadeStage",
    "RetrievalCascade",
    "get_retrieval_cascade",
    "CriticResult",
    "SelfRAGConfig",
    "SelfRAGCritic",
    "get_self_rag_critic",
    "StructuredDataConfig",
    "StructuredDataFormat",
    "StructuredDataParser",
    "StructuredRecord",
    "get_structured_data_parser",
]
