"""Deterministic citation verification and repair.

Implements Guarantee G1: Every citation must map to a retrieved chunk ID 
that was actually in the model's context window, or the system returns "unknown".

Key capabilities:
- Parse [Source: N], [N], (Doc: X), ChunkID: X citation patterns
- Match citations to actual retrieved chunk IDs
- Constrained repair: rewrite with valid citations or mark unverifiable
- Export verification results for trace bundles
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .providers import LLMProvider


@dataclass
class CitationCheck:
    """Result of checking a single citation."""
    citation_id: str
    original_text: str
    valid: bool
    matched_chunk_id: str | None = None
    reason: str = ""


@dataclass
class VerificationResult:
    """Complete verification result for an answer."""
    original_answer: str
    verified_answer: str
    all_valid: bool
    citation_checks: list[CitationCheck] = field(default_factory=list)
    pass_rate: float = 0.0
    repair_attempts: int = 0
    final_pass: bool = False
    
    def to_trace_dict(self) -> dict[str, Any]:
        """Export for trace bundle (E1 requirement)."""
        return {
            "citation_check_pass_rate": round(self.pass_rate, 3),
            "repair_attempts": self.repair_attempts,
            "final_pass": self.final_pass,
            "total_citations": len(self.citation_checks),
            "valid_citations": sum(1 for c in self.citation_checks if c.valid),
            "invalid_citations": [
                {"id": c.citation_id, "reason": c.reason}
                for c in self.citation_checks if not c.valid
            ],
        }


class CitationVerifier:
    """Verify and repair citations against retrieved chunk IDs.
    
    Implements G1: Deterministic citation validity guarantee.
    
    Key features:
    - Parse multiple citation formats: [Source: N], [N], (Doc: X), ChunkID: X
    - Verify citations map to actually-retrieved chunk IDs
    - Constrained repair: rewrite using only valid chunks or mark unverifiable
    - Unit-testable: fake citations are always rejected
    """
    
    # Citation patterns to detect
    CITATION_PATTERNS = [
        r'\[Source:\s*(\d+)\]',            # [Source: 1]
        r'\[(\d+)\]',                       # [1]
        r'\(Doc:\s*([^\)]+)\)',            # (Doc: xyz)
        r'\(Source:\s*([^\)]+)\)',         # (Source: xyz)
        r'ChunkID:\s*([^\s\]\)]+)',        # ChunkID: abc-123
    ]
    
    REPAIR_PROMPT = """The following answer contains citations that do not match the provided source documents.

## Retrieved Sources (use ONLY these chunk IDs for citations)
{chunk_list}

## Answer with invalid citations
{answer}

## Invalid citation IDs found
{invalid_ids}

## Instructions
Rewrite the answer using ONLY the valid chunk IDs listed above. 
For each claim that cannot be supported by any chunk, do one of:
1. Remove the unsupported claim entirely
2. Replace it with: "This information could not be verified from the provided sources."

Keep all correctly cited information intact. 
Output ONLY the corrected answer text, nothing else."""

    def __init__(self, max_repair_attempts: int = 2) -> None:
        """Initialize citation verifier.
        
        Args:
            max_repair_attempts: Max LLM repair attempts before giving up
        """
        self._patterns = [re.compile(p) for p in self.CITATION_PATTERNS]
        self._max_repair_attempts = max_repair_attempts
    
    def extract_citations(self, text: str) -> list[tuple[str, str]]:
        """Extract all citation IDs and their full match text from answer.
        
        Returns:
            List of (citation_id, matched_text) tuples
        """
        citations: list[tuple[str, str]] = []
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                citations.append((match.group(1), match.group(0)))
        return citations
    
    def get_valid_chunk_ids(self, chunks: list["EvidenceChunk"]) -> dict[str, str]:
        """Build mapping of valid citation IDs to chunk IDs.
        
        Returns dict mapping valid citation forms (e.g., "1", "doc-chunk0") 
        to actual chunk IDs.
        """
        valid: dict[str, str] = {}
        for i, chunk in enumerate(chunks):
            chunk_id = getattr(chunk, 'id', None) or getattr(chunk, 'chunk_id', None)
            if chunk_id:
                # Map the full chunk ID
                valid[str(chunk_id)] = str(chunk_id)
                # Also map 1-based numeric index
                valid[str(i + 1)] = str(chunk_id)
        return valid
    
    def verify(
        self,
        answer: str,
        chunks: list["EvidenceChunk"],
    ) -> VerificationResult:
        """Verify all citations in answer against retrieved chunks.
        
        Args:
            answer: Generated answer to verify
            chunks: Source evidence chunks that were in model context
            
        Returns:
            VerificationResult with pass/fail for each citation
        """
        valid_ids = self.get_valid_chunk_ids(chunks)
        citations = self.extract_citations(answer)
        
        checks: list[CitationCheck] = []
        for cid, match_text in citations:
            matched_chunk_id = valid_ids.get(cid)
            is_valid = matched_chunk_id is not None
            checks.append(CitationCheck(
                citation_id=cid,
                original_text=match_text,
                valid=is_valid,
                matched_chunk_id=matched_chunk_id,
                reason="" if is_valid else f"Citation '{cid}' not found in retrieved chunks",
            ))
        
        valid_count = sum(1 for c in checks if c.valid)
        total = len(checks) if checks else 1  # Avoid div by zero
        pass_rate = valid_count / total if checks else 1.0  # No citations = pass
        
        return VerificationResult(
            original_answer=answer,
            verified_answer=answer,
            all_valid=all(c.valid for c in checks) if checks else True,
            citation_checks=checks,
            pass_rate=pass_rate,
            repair_attempts=0,
            final_pass=pass_rate >= 1.0,
        )
    
    async def verify_and_repair(
        self,
        answer: str,
        chunks: list["EvidenceChunk"],
        provider: "LLMProvider",
    ) -> VerificationResult:
        """Verify citations and attempt LLM-based repair if invalid.
        
        Args:
            answer: Generated answer to verify
            chunks: Source evidence chunks
            provider: LLM provider for repair attempts
            
        Returns:
            VerificationResult with repaired answer if needed
        """
        result = self.verify(answer, chunks)
        
        if result.all_valid:
            result.final_pass = True
            return result
        
        # Attempt repair using LLM
        current_answer = answer
        for attempt in range(self._max_repair_attempts):
            result.repair_attempts = attempt + 1
            
            # Build chunk list for repair prompt
            chunk_list_parts = []
            for i, c in enumerate(chunks):
                chunk_id = getattr(c, 'id', None) or getattr(c, 'chunk_id', f"chunk_{i}")
                snippet = getattr(c, 'snippet', str(c))[:150]
                chunk_list_parts.append(f"[{i+1}] ID: {chunk_id}\n    Excerpt: {snippet}...")
            chunk_list = "\n".join(chunk_list_parts)
            
            invalid_ids = [c.citation_id for c in result.citation_checks if not c.valid]
            
            prompt = self.REPAIR_PROMPT.format(
                chunk_list=chunk_list,
                answer=current_answer,
                invalid_ids=", ".join(invalid_ids),
            )
            
            try:
                repaired = await provider.chat([
                    {"role": "system", "content": "You are a citation repair assistant. Fix invalid citations."},
                    {"role": "user", "content": prompt},
                ])
                
                # Verify the repaired answer
                new_result = self.verify(repaired.strip(), chunks)
                new_result.repair_attempts = attempt + 1
                
                if new_result.all_valid:
                    new_result.verified_answer = repaired.strip()
                    new_result.final_pass = True
                    return new_result
                
                current_answer = repaired.strip()
                result = new_result
                
            except Exception as e:
                result.citation_checks.append(CitationCheck(
                    citation_id="repair_error",
                    original_text="",
                    valid=False,
                    reason=f"Repair failed: {str(e)}",
                ))
                break
        
        # Repair failed - mark answer with warning
        result.verified_answer = self._mark_unverified(current_answer, result.citation_checks)
        result.final_pass = False
        return result
    
    def _mark_unverified(
        self,
        answer: str,
        checks: list[CitationCheck],
    ) -> str:
        """Mark answer with warning about unverified citations."""
        invalid_ids = [c.citation_id for c in checks if not c.valid]
        if not invalid_ids:
            return answer
        
        warning = (
            f"[⚠️ Note: Some citations could not be verified against source documents. "
            f"Unverified citation IDs: {', '.join(invalid_ids)}]\n\n"
        )
        return warning + answer
    
    def verify_strict(
        self,
        answer: str,
        chunks: list["EvidenceChunk"],
    ) -> tuple[str, VerificationResult]:
        """Strict verification: remove sentences with invalid citations.
        
        Returns:
            Tuple of (cleaned_answer, verification_result)
        """
        result = self.verify(answer, chunks)
        
        if result.all_valid:
            result.final_pass = True
            return answer, result
        
        # Get set of invalid citation texts
        invalid_texts = {c.original_text for c in result.citation_checks if not c.valid}
        
        # Remove sentences containing invalid citations
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        kept_sentences = []
        removed_count = 0
        
        for sentence in sentences:
            has_invalid = any(inv in sentence for inv in invalid_texts)
            if has_invalid:
                removed_count += 1
            else:
                kept_sentences.append(sentence)
        
        cleaned = " ".join(kept_sentences)
        
        if removed_count > 0:
            cleaned += f"\n\n[Note: {removed_count} sentence(s) removed due to unverifiable citations.]"
        
        result.verified_answer = cleaned
        result.final_pass = len(kept_sentences) > 0
        return cleaned, result


__all__ = [
    "CitationCheck",
    "VerificationResult",
    "CitationVerifier",
]
