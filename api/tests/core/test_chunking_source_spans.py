"""Every quoted chunk must be an exact slice of its indexed source text."""

import pytest

from app.core.chunking import get_chunker


@pytest.mark.parametrize("strategy", ["fixed", "semantic", "recursive", "late"])
@pytest.mark.parametrize("text", [
    "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.",
    "  First sentence.\nSecond sentence.  Third sentence.  ",
    "Repeated paragraph.\n\nRepeated paragraph.\n\nLast paragraph.",
    "Header\r\n\r\nA paragraph with café and 東京.\r\nAnother line.",
    "abcdefghij" * 10,
    "\n\t  ",
    "",
])
def test_chunk_text_and_offsets_round_trip_to_source(strategy, text):
    chunks = get_chunker(strategy, target_size=24, overlap=0).chunk(text)

    for chunk in chunks:
        assert 0 <= chunk.start_char <= chunk.end_char <= len(text)
        assert text[chunk.start_char:chunk.end_char] == chunk.text
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("strategy", ["recursive", "late"])
def test_long_separator_free_text_is_bounded_without_duplicates(strategy):
    text = "0123456789" * 20
    chunks = get_chunker(strategy, target_size=24, overlap=0).chunk(text)

    assert all(len(chunk.text) <= 24 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(previous.end_char == current.start_char for previous, current in zip(chunks, chunks[1:]))


@pytest.mark.parametrize("strategy", ["recursive", "late"])
def test_recursive_splitting_preserves_all_non_whitespace_characters(strategy):
    text = "First sentence.\nSecond sentence. Third sentence.\n\n" + "X" * 65
    chunks = get_chunker(strategy, target_size=20, overlap=0).chunk(text)

    assert "".join("".join(chunk.text.split()) for chunk in chunks) == "".join(text.split())
    assert all(a.end_char <= b.start_char for a, b in zip(chunks, chunks[1:]))


@pytest.mark.parametrize("strategy", ["recursive", "late"])
def test_repeated_text_offsets_remain_monotonic(strategy):
    text = "echo echo echo\n\necho echo echo\n\necho echo echo"
    chunks = get_chunker(strategy, target_size=15, overlap=0).chunk(text)

    assert len(chunks) == 3
    assert all(a.end_char <= b.start_char for a, b in zip(chunks, chunks[1:]))
    assert [text[c.start_char:c.end_char] for c in chunks] == ["echo echo echo"] * 3


def test_hard_split_overlap_does_not_add_a_contained_tail():
    text = "abcdefghijklmnopqr"
    chunks = get_chunker("recursive", target_size=10, overlap=2).chunk(text)
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 10), (8, 18)]


def test_trimming_whitespace_does_not_create_a_contained_overlap_chunk():
    text = "abcdefghij" + "\t" * 5 + "klmnopqrstuvwxyz"
    chunks = get_chunker("recursive", target_size=10, overlap=5).chunk(text)
    assert all(not (previous.start_char <= current.start_char
                    and current.end_char <= previous.end_char)
               for previous, current in zip(chunks, chunks[1:]))
    covered = {offset for chunk in chunks for offset in range(chunk.start_char, chunk.end_char)}
    assert all(offset in covered for offset, char in enumerate(text) if not char.isspace())


@pytest.mark.parametrize("strategy", ["fixed", "semantic", "recursive", "late"])
@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_chunk_size_is_rejected_instead_of_losing_source(strategy, size):
    with pytest.raises(ValueError, match="positive"):
        get_chunker(strategy, target_size=size).chunk("Useful source evidence.")


@pytest.mark.parametrize("size", [0, -1])
def test_public_retrieval_config_rejects_nonpositive_chunk_size(size):
    from pydantic import ValidationError
    from app.schemas.config import RetrievalDefaults

    with pytest.raises(ValidationError):
        RetrievalDefaults(chunk_size=size)


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 11), (10, -1), (400, 400)])
def test_recursive_invalid_overlap_is_rejected(size, overlap):
    from app.core.chunking import RecursiveChunker

    with pytest.raises(ValueError, match="overlap"):
        RecursiveChunker(target_size=size, overlap=overlap)


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 11), (10, -1), (400, 400)])
def test_public_recursive_config_rejects_invalid_overlap(size, overlap):
    from pydantic import ValidationError
    from app.schemas.config import RetrievalDefaults

    with pytest.raises(ValidationError, match="chunk_overlap"):
        RetrievalDefaults(chunking_strategy="recursive", chunk_size=size, chunk_overlap=overlap)


def test_recursive_default_and_explicit_small_nonoverlapping_sizes_are_valid():
    from app.core.chunking import RecursiveChunker

    assert RecursiveChunker().chunk("0123456789" * 100)
    chunks = RecursiveChunker(target_size=10, overlap=0).chunk("0123456789" * 10)
    assert len(chunks) == 10
    assert "".join(chunk.text for chunk in chunks) == "0123456789" * 10


def test_recursive_small_size_with_oversized_default_overlap_has_clear_error():
    from app.core.chunking import RecursiveChunker

    with pytest.raises(ValueError, match="overlap.*smaller.*target_size"):
        RecursiveChunker(target_size=10)
