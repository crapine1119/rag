from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

from agent.data_scientist import DataScientist
from agent.ml_researcher import MLResearcher
from vectorstore.chroma import CustomChormaForMetadata

if __name__ == "__main__":

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
    ml_researcher = MLResearcher(vectorstore=vectordb)

    data_scientist.read("https://arxiv.org/pdf/2305.05726")
    ml_researcher.answer("Few-/Zero-shot Semantic Segmentation", "mmr", **{"k": 10, "fetch_k": 20, "lambda_mult": 0.1})
