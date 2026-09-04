from app.core.persistence import IndexMetadata, IndexPersistence


def test_save_and_load_trees_json(tmp_path):
    persistence = IndexPersistence(base_path=tmp_path)
    metadata = IndexMetadata(
        corpus_version="v1",
        config_hash="abc1234",
        chunk_count=2,
        created_at=1000.0,
        model_name="test-model",
    )
    trees_data = {
        "doc1": {
            "root_ids": ["node1"],
            "nodes": {
                "node1": {
                    "node_id": "node1",
                    "text": "Sample summary node",
                    "level": 1,
                    "children_ids": ["leaf1", "leaf2"],
                }
            },
        }
    }

    saved_path = persistence.save_trees("test_index", trees_data, metadata)
    assert saved_path.suffix == ".json"
    assert saved_path.exists()

    loaded_trees, loaded_metadata = persistence.load_trees("test_index")
    assert loaded_trees == trees_data
    assert loaded_metadata is not None
    assert loaded_metadata.corpus_version == "v1"
    assert loaded_metadata.config_hash == "abc1234"


def test_load_trees_nonexistent(tmp_path):
    persistence = IndexPersistence(base_path=tmp_path)
    loaded_trees, loaded_metadata = persistence.load_trees("nonexistent_index")
    assert loaded_trees is None
    assert loaded_metadata is None
