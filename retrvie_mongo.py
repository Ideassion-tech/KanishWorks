from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient
import mongo_det


embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

#connecting to MongoDB
client=MongoClient(mongo_det.mongodb_con_string)
collection = client[mongo_det.db_name][mongo_det.collection_name]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection, 
    index_name=mongo_det.index_name
)

query = "what are supercars"    
k=3
docs = vector_store.max_marginal_relevance_search(query, K=k)
model= SentenceTransformer("BAAI/bge-base-en")
qestion_embedding = model.encode(query)
for i in range(0,k):
    print(i)
    print("\n|----------------------------------|\n")   
    print(docs[i].page_content)
    cos_scores = util.cos_sim(qestion_embedding, docs[i].metadata['embedding'])
    print(f"Cosine Similarity Score: {cos_scores.item()}\n")
    print("\n|----------------------------------|\n")