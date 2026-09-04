from __future__ import annotations

import pytest

from app.core.documents import DocumentStore


def test_document_store_rejects_duplicate_titles_by_default(tmp_path):
    store = DocumentStore(path=tmp_path / "documents.db")
    original = store.add(
        title="Victim Report", text="original", metadata={"owner": "victim"}
    )

    with pytest.raises(ValueError, match="Document title already exists"):
        store.add(
            title="  victim report  ", text="attacker", metadata={"owner": "attacker"}
        )

    stored = store.get(original.id)
    assert stored is not None
    assert stored.id == original.id
    assert stored.text == "original"
    assert stored.metadata["owner"] == "victim"
    assert len(store.list()) == 1


def test_document_store_can_explicitly_replace_duplicate_titles(tmp_path):
    store = DocumentStore(path=tmp_path / "documents.db")
    original = store.add(
        title="Victim Report", text="original", metadata={"owner": "victim"}
    )

    replacement = store.add(
        title="  victim report  ",
        text="replacement",
        metadata={"owner": "trusted-updater"},
        on_duplicate="replace",
    )

    assert replacement.id == original.id
    stored = store.get(original.id)
    assert stored is not None
    assert stored.text == "replacement"
    assert stored.metadata["owner"] == "trusted-updater"
    assert len(store.list()) == 1
