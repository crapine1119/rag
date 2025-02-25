from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


class MLResearcher:
    def __init__(self, vectorstore: VectorStore):
        self._vectorstore = vectorstore

    def get_retriever(self, search_type: str = "mmr", **search_kwargs) -> BaseRetriever:
        return self._vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def answer(self, query: str, search_type: str = "mmr", **search_kwargs):
        retriever = self.get_retriever(search_type, **search_kwargs)
        docs = retriever.invoke(query)

        # for d in map(lambda x: x.page_content, docs):
        #     print("==" * 100)
        #     print(d)
