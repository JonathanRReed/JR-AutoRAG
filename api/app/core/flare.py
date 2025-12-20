"""FLARE: Forward-Looking Active Retrieval during generation.

This module provides confidence-triggered retrieval during text generation:
- Monitor generation confidence in real-time
- Retrieve additional context when confidence drops
- Re-generate with enhanced context
- Seamless integration with streaming generation

Based on: Active Retrieval Augmented Generation
Paper: https://arxiv.org/abs/2305.06983
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers import LLMProvider
    from .gatherer import EvidenceChunk
    from .hybrid_retrieval import HybridRetrievalEngine


@dataclass
class FLAREConfig:
    """Configuration for FLARE generation."""
    confidence_threshold: float = 0.3  # Trigger retrieval below this
    max_retrievals: int = 3  # Maximum mid-generation retrievals
    lookahead_tokens: int = 50  # Tokens to generate for lookahead
    retrieval_top_k: int = 3  # Chunks to retrieve per trigger
    min_sentence_length: int = 20  # Minimum chars before considering retrieval


@dataclass
class FLAREStep:
    """A single step in FLARE generation."""
    text: str
    confidence: float
    triggered_retrieval: bool = False
    retrieved_chunks: list[str] = field(default_factory=list)


@dataclass
class FLAREResult:
    """Result of FLARE generation."""
    answer: str
    steps: list[FLAREStep]
    total_retrievals: int
    total_chunks_used: int


class FLAREGenerator:
    """Forward-Looking Active Retrieval during generation.
    
    FLARE monitors generation confidence and triggers retrieval
    when the model is uncertain. Key steps:
    
    1. Generate a sentence
    2. Estimate confidence (via heuristics or logprobs)
    3. If low confidence, use generated text as retrieval query
    4. Inject new context and regenerate sentence
    5. Continue until complete
    
    This implementation uses heuristic confidence estimation since
    most local LLMs don't expose logprobs.
    """
    
    # Patterns that suggest uncertainty
    UNCERTAINTY_PATTERNS = [
        r'\b(maybe|perhaps|possibly|might|could be|I think|I believe)\b',
        r'\b(not sure|uncertain|unclear|unknown)\b',
        r'\b(approximately|around|about|roughly)\b',
        r'\?$',  # Questions indicate uncertainty
    ]
    
    # Patterns that suggest confidence
    CONFIDENCE_PATTERNS = [
        r'\b(definitely|certainly|clearly|obviously|according to)\b',
        r'\[\d+\]',  # Citations indicate grounded statements
        r'\b(research shows|studies indicate|data shows)\b',
    ]
    
    def __init__(self, config: FLAREConfig | None = None) -> None:
        self.config = config or FLAREConfig()
        self._uncertainty_re = [re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_PATTERNS]
        self._confidence_re = [re.compile(p, re.IGNORECASE) for p in self.CONFIDENCE_PATTERNS]
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence of generated text via heuristics.
        
        Returns a score 0-1 where:
        - 0.0 = very uncertain
        - 1.0 = very confident
        """
        if not text.strip():
            return 0.5
        
        # Count uncertainty indicators
        uncertainty_count = sum(1 for p in self._uncertainty_re if p.search(text))
        
        # Count confidence indicators
        confidence_count = sum(1 for p in self._confidence_re if p.search(text))
        
        # Base confidence
        base = 0.5
        
        # Adjust based on patterns
        uncertainty_penalty = 0.15 * uncertainty_count
        confidence_bonus = 0.2 * confidence_count
        
        # Short sentences are less confident
        if len(text) < 50:
            uncertainty_penalty += 0.1
        
        # Very long sentences might be rambling
        if len(text) > 300:
            uncertainty_penalty += 0.05
        
        confidence = base - uncertainty_penalty + confidence_bonus
        return max(0.0, min(1.0, confidence))
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_retrieval_query(self, sentence: str, original_query: str) -> str:
        """Extract a query from the generated sentence for retrieval.
        
        Combines the uncertain statement with the original query
        to find relevant context.
        """
        # Remove uncertainty markers
        clean = sentence
        for p in self._uncertainty_re:
            clean = p.sub('', clean)
        
        # Extract key terms (nouns, verbs likely)
        key_terms = re.findall(r'\b[A-Za-z]{4,}\b', clean)
        
        if key_terms:
            # Combine with original query
            combined = f"{original_query} {' '.join(key_terms[:5])}"
            return combined
        
        return original_query
    
    async def generate_with_flare(
        self,
        query: str,
        initial_context: str,
        provider: "LLMProvider",
        retriever: "HybridRetrievalEngine",
        document_ids: list[str] | None = None,
    ) -> FLAREResult:
        """Generate answer with active retrieval on uncertainty.
        
        Args:
            query: User's question
            initial_context: Initial retrieved context
            provider: LLM provider for generation
            retriever: Retrieval engine for active retrieval
            document_ids: Optional document filter
        
        Returns:
            FLAREResult with complete answer and step details
        """
        steps: list[FLAREStep] = []
        current_context = initial_context
        retrieval_count = 0
        all_chunk_ids: set[str] = set()
        
        # System prompt for RAG generation
        system_prompt = """You are a precise RAG assistant. Answer based on the provided context.
Use citations like [1], [2] when referencing sources.
If uncertain about something, state it clearly."""
        
        # Generate initial response
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{current_context}\n\nQuestion: {query}"},
        ]
        
        try:
            full_response = await provider.chat(messages)
        except Exception as e:
            return FLAREResult(
                answer=f"Generation error: {e}",
                steps=[],
                total_retrievals=0,
                total_chunks_used=0,
            )
        
        # Split into sentences for confidence checking
        sentences = self._split_sentences(full_response)
        
        final_answer_parts = []
        
        for i, sentence in enumerate(sentences):
            confidence = self._estimate_confidence(sentence)
            
            step = FLAREStep(
                text=sentence,
                confidence=confidence,
            )
            
            # Check if retrieval needed
            if (confidence < self.config.confidence_threshold and 
                retrieval_count < self.config.max_retrievals and
                len(sentence) >= self.config.min_sentence_length):
                
                # Generate retrieval query from uncertain sentence
                retrieval_query = self._extract_retrieval_query(sentence, query)
                
                # Retrieve additional context
                try:
                    results = retriever.query(
                        retrieval_query,
                        top_k=self.config.retrieval_top_k,
                        document_ids=document_ids,
                    )
                    
                    if results:
                        step.triggered_retrieval = True
                        step.retrieved_chunks = [r.chunk_id for r in results]
                        all_chunk_ids.update(step.retrieved_chunks)
                        retrieval_count += 1
                        
                        # Add new context
                        new_context = "\n\n".join([
                            f"[Additional Context {j+1}]: {r.chunk_text[:500]}"
                            for j, r in enumerate(results)
                        ])
                        
                        # Regenerate this sentence with enhanced context
                        regen_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": (
                                f"Context:\n{current_context}\n\n"
                                f"{new_context}\n\n"
                                f"Question: {query}\n\n"
                                f"Continue the answer after: {' '.join(final_answer_parts)}\n"
                                f"Provide just the next sentence with citations."
                            )},
                        ]
                        
                        try:
                            regenerated = await provider.chat(regen_messages)
                            # Take just the first sentence
                            regen_sentences = self._split_sentences(regenerated)
                            if regen_sentences:
                                sentence = regen_sentences[0]
                                step.text = sentence
                                step.confidence = self._estimate_confidence(sentence)
                        except Exception:
                            pass  # Keep original sentence
                        
                        # Update context for future sentences
                        current_context = f"{current_context}\n\n{new_context}"
                
                except Exception:
                    pass  # Continue with original sentence
            
            steps.append(step)
            final_answer_parts.append(sentence)
        
        final_answer = " ".join(final_answer_parts)
        
        return FLAREResult(
            answer=final_answer,
            steps=steps,
            total_retrievals=retrieval_count,
            total_chunks_used=len(all_chunk_ids),
        )
    
    async def generate_streaming_with_flare(
        self,
        query: str,
        initial_context: str,
        provider: "LLMProvider",
        retriever: "HybridRetrievalEngine",
        document_ids: list[str] | None = None,
        on_token: Any = None,
    ) -> FLAREResult:
        """Generate with FLARE and streaming output.
        
        Similar to generate_with_flare but yields tokens as they're generated.
        Retrieval triggers may cause brief pauses in the stream.
        """
        # For simplicity, this implementation generates fully then streams
        # A full implementation would integrate with streaming generation
        result = await self.generate_with_flare(
            query, initial_context, provider, retriever, document_ids
        )
        
        # Stream the result if callback provided
        if on_token:
            for char in result.answer:
                on_token(char)
        
        return result


__all__ = [
    "FLAREConfig",
    "FLAREStep",
    "FLAREResult",
    "FLAREGenerator",
]
