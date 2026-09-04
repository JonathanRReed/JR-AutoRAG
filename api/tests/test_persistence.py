import json
from app.core.persistence import ArtifactPersistence, IndexMetadata, IndexType


def test_save_and_load_trees_json(tmp_path):
    persistence = ArtifactPersistence(base_path=tmp_path)
    index_name = "test_index"

    tree_data = {
        "doc_1": {
            "root_id": "node_0",
            "nodes": {
                "node_0": {
                    "id": "node_0",
                    "text": "Root summary",
                    "children": ["node_1"],
                    "level": 1,
                },
                "node_1": {
                    "id": "node_1",
                    "text": "Leaf chunk",
                    "children": [],
                    "level": 0,
                },
            },
        }
    }

    metadata = IndexMetadata(
        index_name=index_name,
        index_type=IndexType.RAPTOR,
        document_count=1,
        total_chunks=2,
    )

    saved_path = persistence.save_trees(index_name, tree_data, metadata)

    # Confirm file saved is .json
    assert saved_path.suffix == ".json"
    assert saved_path.exists()

    # Verify file contents are valid JSON
    with open(saved_path, "r") as f:
        loaded_json_content = json.load(f)
    assert loaded_json_content == tree_data

    # Test load_trees
    loaded_trees, loaded_meta = persistence.load_trees(index_name)
    assert loaded_trees == tree_data
    assert loaded_meta is not None
    assert loaded_meta.index_name == index_name
    assert loaded_meta.document_count == 1


def test_load_trees_nonexistent(tmp_path):
    persistence = ArtifactPersistence(base_path=tmp_path)
    loaded_trees, loaded_meta = persistence.load_trees("nonexistent")
    assert loaded_trees is None
    assert loaded_meta is None
