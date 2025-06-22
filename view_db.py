from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def view_db_content(PERSIST_DIR: str) -> None:
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    collection = vectorstore._collection
    results = collection.get(include=["documents", "metadatas","embeddings"])

    print("\n✅ Stored Chunks in ChromaDB:\n")
    for i, (doc, meta, emb) in enumerate(zip(results["documents"], results["metadatas"], results["embeddings"])):
        print(f"--- Chunk {i+1} ---")
        print(f"Content:\n{doc}")
        print(f"Metadata: {meta}")
        print(f"Embedding (length {len(emb)}): {emb[:5]}...")  # Show only first 5 values for brevity
        print()


if __name__ == "__main__":
    persist_directory = "C:\kanish\KanishWorks\chroma_db"
    view_db_content(persist_directory)