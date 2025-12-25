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
from .bq_hybrid_retrieval import BQHybridRetrievalEngine
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
# v2 Binary Quantization modules
from .binary_quantization import (
    BQConfig, BQ_VERSION, validate_dimension, float32_to_binary,
    binary_to_bits, hamming_distance, batch_float32_to_binary,
    get_binary_dimension, estimate_storage_savings,
)
from .binary_vector_store import (
    MilvusConfig, MilvusChunk, MilvusSearchResult, MilvusVectorStore,
    IndexStats, is_milvus_available, get_milvus_store,
)
from .bq_retrieval import (
    RetrievalModeV2, RetrievalTimings, RetrievalDebug, RetrievedChunk,
    BQRetrievalConfig, BQRetrievalService, get_bq_retrieval_service,
)
# SOTA Agentic components
from .retrieval_evaluator import RetrievalEvaluator, RetrievalVerdict, EvaluationResult as RetrievalEvalResult, KnowledgeStrip
from .adaptive_gate import AdaptiveGate, GateDecision, GateResult
# Web search disabled for offline-only operation
# from .web_search import WebSearch, WebResult, WebSearchProvider, get_web_search
from .graph_rag import GraphRAG, Entity, Relationship, Community, EntityType
from .flare import FLAREGenerator, FLAREConfig, FLAREResult, FLAREStep
# Enterprise modules
from .auth import APIKeyAuth, APIKey, get_auth
from .audit import AuditLog, AuditEntry, AuditAction, get_audit_log
from .rate_limiter import RateLimiter, TokenBucket, RateLimitConfig, get_rate_limiter
from .citation_formatter import RichCitation, format_rich_context, generate_reference_section, validate_answer_citations
from .evidence_validator import EvidenceValidator, ValidationResult, create_strict_validator
from .structured_generator import ClaimType, ClaimSlot, StructuredPick, StructuredMemo, StructuredGenerator
# vNext Expansion modules (Guarantees G1-G4)
from .cache import RetrievalMode
from .citation_verifier import CitationVerifier, CitationCheck, VerificationResult
from .trace_export import TraceBundle, create_trace_bundle, summarize_steps
from .artifact_builder import (
    ArtifactBuilder, ArtifactStatus, ArtifactState, BuildProgress, get_artifact_builder
)
# v2.0 Abstention rules
from .abstention import AbstentionRules, AbstentionConfig, AbstentionResult, AbstentionReason, get_abstention_rules
# v2.0 Phase 2: Agentic loops
from .self_rag import SelfRAGCritic, SelfRAGConfig, CriticResult, get_self_rag_critic
from .retrieval_cascade import RetrievalCascade, CascadeConfig, CascadeResult, CascadeStage, get_retrieval_cascade
from .auto_weights import AutoHybridWeights, AutoWeightConfig, WeightProfile, get_auto_weights
# v2.0 Phase 3: Ingestion enhancements
from .contextual_enrichment import ContextualEnricher, EnrichmentConfig, EnrichedChunk, get_contextual_enricher
from .multi_granularity import MultiGranularityIndexer, MultiGranularityConfig, GranularChunk, GranularityLevel, get_multi_granularity_indexer
from .structured_data_parser import StructuredDataParser, StructuredDataConfig, StructuredRecord, StructuredDataFormat, get_structured_data_parser
# v2.0 Phase 4: Observability & Security
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry, CircuitState, get_circuit_breaker, get_circuit_breaker_registry
from .cost_tracker import CostLatencyTracker, TokenUsage, CostEstimate, LatencyBreakdown, RequestMetrics, get_cost_tracker
# v2.0 Phase 5: Polish & Docs
from .feature_flags import FeatureFlagRegistry, FeatureFlag, RiskLevel, V2_FEATURE_FLAGS, get_feature_flags, is_feature_enabled
from .config_migration import ConfigMigrator, MigrationResult, V2_NEW_FIELDS, get_config_migrator, migrate_config

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
]
