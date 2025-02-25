from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from overrides import overrides


class CustomChormaForMetadata(Chroma):
    def __init__(self, **chroma_kwargs):
        super().__init__(**chroma_kwargs)

    @overrides
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        if type(self).add_texts != VectorStore.add_texts:
            if "ids" not in kwargs:
                ids = [doc.id for doc in documents]

                # If there's at least one valid ID, we'll assume that IDs
                # should be used.
                if any(ids):
                    kwargs["ids"] = ids

            texts = [doc.page_content for doc in documents]
            metadatas = [self._preprocess_metadata(doc.metadata) for doc in documents]
            return self.add_texts(texts, metadatas, **kwargs)
        msg = f"`add_documents` and `add_texts` has not been implemented " f"for {self.__class__.__name__} "
        raise NotImplementedError(msg)

    def _preprocess_metadata(self, metadata: dict) -> dict:
        processed_metadata = {}
        for k, v in metadata.items():
            if v is None:
                processed_metadata[k] = "None"
            elif isinstance(v, list):
                processed_metadata[k] = str(v)
            else:
                processed_metadata[k] = v
        return processed_metadata
