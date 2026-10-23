"""Versioned, local text embedding contracts. No imports of optional ML runtimes.

Model cards and pinned configs were checked on 2026-10-21. These are practical
presets, not a claim that leaderboard quality transfers to every user corpus.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os

import numpy as np

DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2"
LEGACY_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


@dataclass(frozen=True)
class EmbeddingProfile:
    model_id: str
    revision: str
    label: str
    dimensions: int
    max_tokens: int
    query_prompt: str
    document_prompt: str = ""
    text_only: bool = False
    contract_version: int = 1


EMBEDDING_PROFILES = {
    "microsoft/harrier-oss-v1-0.6b": EmbeddingProfile(
        "microsoft/harrier-oss-v1-0.6b", "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        "Harrier 0.6B · quality alternative", 1024, 32768,
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    ),
    "google/embeddinggemma-2": EmbeddingProfile(
        "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "EmbeddingGemma 2 · lightweight text", 768, 8192,
        "task: search result | query: ", "title: none | text: ", True,
    ),
    "Qwen/Qwen3-Embedding-4B": EmbeddingProfile(
        "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b",
        "Qwen3 Embedding 4B · high memory", 2560, 32768,
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    ),
}


def embedding_fingerprint(model_id: str) -> str:
    profile = EMBEDDING_PROFILES.get(model_id)
    if profile is None:
        return model_id  # Preserve existing custom and legacy index identities.
    payload = {**asdict(profile), "normalization": "l2", "dtype": "float32"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def encode_query(embedder, texts, **kwargs):
    """Use explicit retrieval-query formatting, retaining legacy adapters."""
    # A new Sentence Transformers release may add prompt-aware encode_query
    # to legacy models. Do not silently change old index/query contracts.
    if isinstance(embedder, ProfiledEmbedder):
        return embedder.encode_query(texts, **kwargs)
    return embedder.encode(texts, **kwargs)


class ProfiledEmbedder:
    """Document-default adapter shared by ingestion, chunking and RAPTOR.

Queries must call encode_query. Explicit prompts prevent model defaults from
silently changing document/query roles. Fail closed on invalid vectors instead
of caching NaNs, mixing dimensions, or silently truncating long source text.
"""

    def __init__(self, model, profile: EmbeddingProfile):
        self.model = model
        self.profile = profile

    def encode(self, texts, **kwargs):
        return self._encode(texts, self.profile.document_prompt, **kwargs)

    def encode_query(self, texts, **kwargs):
        return self._encode(texts, self.profile.query_prompt, **kwargs)

    def _encode(self, texts, prompt, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return np.empty((0, self.profile.dimensions), dtype=np.float32)
        tokenizer = getattr(self.model, "tokenizer", None)
        if callable(tokenizer):
            limit = min(self.profile.max_tokens, getattr(self.model, "max_seq_length", self.profile.max_tokens))
            for text in items:
                tokens = tokenizer(prompt + text, truncation=False, add_special_tokens=True)["input_ids"]
                if len(tokens) > limit:
                    raise ValueError(f"Input exceeds {limit} tokens; split the source or shorten the query")
        kwargs.pop("prompt_name", None)
        kwargs.pop("prompt", None)
        kwargs.setdefault("batch_size", 4)
        kwargs.update(prompt=prompt, normalize_embeddings=True, convert_to_numpy=True)
        vectors = np.asarray(self.model.encode(items, **kwargs), dtype=np.float32)
        if vectors.shape != (len(items), self.profile.dimensions):
            raise ValueError("Embedding dimension or batch size does not match the selected profile")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Embedding model returned non-finite or zero vectors")
        vectors = vectors / norms
        return vectors[0] if single else vectors


def load_profiled_embedder(factory, model_id: str):
    """Load a pinned built-in model without executing remote model code."""
    profile = EMBEDDING_PROFILES[model_id]
    options = dict(device=os.environ.get("AUTORAG_EMBEDDING_DEVICE") or None, revision=profile.revision, trust_remote_code=False,
                   model_kwargs={"dtype": "float32"})
    if profile.text_only:
        options["config_kwargs"] = {"vision_config": None, "audio_config": None}
    return ProfiledEmbedder(factory(model_id, **options), profile)
