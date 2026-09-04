from unittest.mock import Mock
from app.core.async_ingest import AsyncIngestManager, JobStore, IngestJob


def test_submit_sync():
    mock_pipeline = Mock()
    manager = AsyncIngestManager(pipeline=mock_pipeline)

    # Mock the job store create method
    mock_job = IngestJob(id="test_id", title="Test Job")
    manager._job_store = Mock(spec=JobStore)
    manager._job_store.create.return_value = mock_job

    title = "Test Job"
    content = b"Test content"
    metadata = {"key": "value"}

    result = manager.submit_sync(title=title, content=content, metadata=metadata)

    manager._job_store.create.assert_called_once_with(title, content, metadata)
    assert result == mock_job
