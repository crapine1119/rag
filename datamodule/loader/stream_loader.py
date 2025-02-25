from functools import partial
from typing import Iterator

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, ArxivLoader
from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader


class StreamLoader:
    def __init__(self):
        self._web_loader = partial(WebBaseLoader)
        self._google_loader = partial(GoogleDriveLoader)
        self._pdf_loader = partial(PyPDFLoader)
        self._arxiv_loader = partial(ArxivLoader, load_max_docs=1, load_all_available_meta=True)

    def stream(self, stream_filepath: str, **kwargs) -> Iterator[Document]:
        if stream_filepath.startswith("https://arxiv.org/"):
            _, arxiv_id = stream_filepath.split("arxiv.org/pdf/")
            loader = self._arxiv_loader(query=arxiv_id)
        elif stream_filepath.endswith(".pdf"):
            loader = self._pdf_loader(stream_filepath)
        elif stream_filepath.startswith("https"):
            loader = self._web_loader(stream_filepath)
        elif stream_filepath.startswith("https://drive.google.com"):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return loader.lazy_load()
