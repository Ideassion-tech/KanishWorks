import os
import torch
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])



def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return splitter.split_text(text)


def embedding_chunks(file_path: str, persist_directory: str) -> None:
    text=load_document(file_path)
    chunks=chunk_document(text)
    print(f"✅ Loaded and chunked into {len(chunks)} parts.")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    vector_store = Chroma.from_texts(
        texts = chunks, 
        embedding = embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        persist_directory=persist_directory
    )
    vector_store.persist()
    print(f"\n✅ Embeddings stored in {persist_directory}.\n")

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
        print(f"Embedding (length {len(emb)}): {emb[:5]}...") 
        print()

if __name__ == "__main__":
    file_path = "RMI_Supercars_Coming_Light_Vehicle_Revolution_1993.pdf"
    persist_directory = "C:\kanish\KanishWorks\chroma_db"
    embedding_chunks(file_path, persist_directory)
    print("✅ Embedding process completed successfully.")
    view_db_content(persist_directory)
