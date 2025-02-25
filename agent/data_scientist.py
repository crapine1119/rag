from typing import Any

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter

from datamodule.loader.stream_loader import StreamLoader


class DataScientist:
    def __init__(self, splitter: TextSplitter, vectorstore: VectorStore, tester: Any = None):
        self._loader = StreamLoader()
        self._splitter = splitter
        self._vectorstore = vectorstore
        self._tester = tester

    def read(self, filepath: str, **kwargs):
        lazy_docs = self._loader.stream(filepath, **kwargs)
        docs: list[Document] = self._splitter.split_documents([doc for doc in lazy_docs])
        self._vectorstore.add_documents(docs)
        print("Documents are stored to vectorDB")


if __name__ == "__main__":
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from vectorstore.chroma import CustomChormaForMetadata

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Specify the character chunk size
        chunk_overlap=200,  # "Allowed" Overlap across chunks
        length_function=len,  # Function used to evaluate the chunk size (here in terms of characters)
    )
    embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    # vectordb = FAISS(
    #     embedding_function=embedding,
    #     index=faiss.IndexFlatL2(384),
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )

    vectordb = CustomChormaForMetadata(embedding_function=embedding)

    data_scientist = DataScientist(splitter=text_splitter, vectorstore=vectordb)
    data_scientist.read("https://arxiv.org/pdf/2305.05726")
