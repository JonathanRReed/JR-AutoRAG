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
from .uncertainty_monitor import (
    ConfidenceSignal,
    UncertaintyMonitor,
    UNCERTAINTY_PATTERNS,
)


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
    confidence_signal: ConfidenceSignal | None = None


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

    This implementation now leverages the shared UncertaintyMonitor so that
    logprob- or entropy-based signals (when available) can be blended with the
    heuristic cues.
    """

    def __init__(
        self,
        config: FLAREConfig | None = None,
        monitor: UncertaintyMonitor | None = None,
    ) -> None:
        self.config = config or FLAREConfig()
        threshold = self.config.confidence_threshold
        self._monitor = monitor or UncertaintyMonitor(threshold=threshold)
        self._uncertainty_re = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]

    def _estimate_confidence(self, text: str) -> ConfidenceSignal:
        """Estimate confidence of generated text."""

        return self._monitor.estimate(text)
    
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
        system_prompt: str | None = None,
        answer_instruction: str | None = None,
        continue_instruction: str | None = None,
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
        system_prompt = system_prompt or """You are a precise RAG assistant. Answer based on the provided context.
Use citations like [1], [2] when referencing sources.
If uncertain about something, state it clearly."""

        answer_instruction = answer_instruction or (
            "Answer ONLY using the provided context and include bracketed citations like [1]."
        )
        continue_instruction = continue_instruction or "Provide just the next sentence with bracketed citations."
        
        # Generate initial response
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{current_context}\n\nQuestion: {query}\n\n"
                    f"{answer_instruction}"
                ),
            },
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
            signal = self._estimate_confidence(sentence)
            confidence_value = signal.aggregate
            step = FLAREStep(
                text=sentence,
                confidence=confidence_value,
                confidence_signal=signal,
            )
            
            # Check if retrieval needed
            if (
                self._monitor.should_trigger(signal) and 
                retrieval_count < self.config.max_retrievals and
                len(sentence) >= self.config.min_sentence_length
            ):
                
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
                                f"{continue_instruction}"
                            )},
                        ]
                        
                        try:
                            regenerated = await provider.chat(regen_messages)
                            # Take just the first sentence
                            regen_sentences = self._split_sentences(regenerated)
                            if regen_sentences:
                                sentence = regen_sentences[0]
                                signal = self._estimate_confidence(sentence)
                                step.text = sentence
                                step.confidence = signal.aggregate
                                step.confidence_signal = signal
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
        system_prompt: str | None = None,
        answer_instruction: str | None = None,
        continue_instruction: str | None = None,
    ) -> FLAREResult:
        """Generate with FLARE and streaming output.
        
        Similar to generate_with_flare but yields tokens as they're generated.
        Retrieval triggers may cause brief pauses in the stream.
        """
        # For simplicity, this implementation generates fully then streams
        # A full implementation would integrate with streaming generation
        result = await self.generate_with_flare(
            query,
            initial_context,
            provider,
            retriever,
            document_ids,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            continue_instruction=continue_instruction,
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
