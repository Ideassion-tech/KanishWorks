
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

model = SentenceTransformer("BAAI/bge-base-en")
client = chromadb.Client()
collection = client.create_collection(
    name="my_collection_complete",
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_search": 100,
            "ef_construction": 100,
            "max_neighbors": 16,
            "num_threads": 4
        },
        "embedding_function": model
    }
)

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def chunk_text(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]


def embed_chunks(text_chunks):
    return model.encode(text_chunks).tolist()

def store_in_chroma(chunks, embeddings):
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )

def query_chroma(prompt, k=5):
    query_embedding = model.encode([prompt])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results["documents"][0]

if __name__ == "__main__":
    # Step 1: Load
    pdf_path = "RMI_Supercars_Coming_Light_Vehicle_Revolution_1993.pdf" 
    documents = load_pdf(pdf_path)

    # Step 2: Chunk
    chunks = chunk_text(documents)
    print(f"Loaded and chunked into {len(chunks)} parts.")

    embeddings = embed_chunks(chunks)
    print(f"Embedded {len(embeddings)} chunks.")
 
    store_in_chroma(chunks, embeddings)
    print("Stored embeddings in ChromaDB.")

    query = "What is main aerodynamic feature of the car?"
    top_chunks = query_chroma(query)

    print("\nTop 5 Results:\n")
    for i, chunk in enumerate(top_chunks):
        print(f"Result {i+1}:\n{chunk}\n{'-'*50}")