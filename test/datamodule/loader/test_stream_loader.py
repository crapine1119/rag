import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from datamodule.loader.stream_loader import StreamLoader  # Adjust import based on actual module location


def test_stream_loader():
    loader = StreamLoader()
    test_url = "https://arxiv.org/pdf/2305.05726"

    mock_document = Document(page_content="Test Content", metadata={"source": test_url})

    with patch.object(loader, "load", return_value=iter([mock_document])):
        result = list(loader.stream(test_url))

        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].page_content == "Test Content"
        assert result[0].metadata["source"] == test_url


if __name__ == "__main__":
    test_stream_loader()
