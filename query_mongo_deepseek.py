from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient
import mongo_det
import deepseek
import llama

embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

#connecting to MongoDB
client=MongoClient(mongo_det.mongodb_con_string)
collection = client[mongo_det.db_name][mongo_det.collection_name]

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection, 
    index_name=mongo_det.index_name
)

query = str(input("Enter your query: "))    
k=3
docs = vector_store.max_marginal_relevance_search(query, K=k)
model= SentenceTransformer("BAAI/bge-base-en")
qestion_embedding = model.encode(query).tolist()
for i in range(0,k):
    print(i+1)
    print("\n|----------------------------------|\n")
    print(docs[i].page_content)
    cos_scores = util.cos_sim(qestion_embedding, docs[i].metadata['embedding'])
    print(f"Cosine Similarity Score: {cos_scores.item()}\n")
    print("\n|----------------------------------|\n")

required_chunks = [docs[i].page_content for i in range(k)]

content = "\n\n".join(required_chunks)

prompt =f'''

user QUESTION: {query}
content: {content}
Answer the users question asked by user according to the content given here. just give the answer, do not repeat the question or say anything else.
'''


response_deepseek = deepseek.deepseek_ai(prompt)
print("Deepseek Response:")
print(response_deepseek)
response_embedding_deepseek = model.encode(response_deepseek).tolist()
cos_scores_deepseek = util.cos_sim(qestion_embedding, response_embedding_deepseek)
print(f"Cosine Similarity Score for response: {cos_scores_deepseek.item()}\n")


response_llama = llama.llama_ai(prompt)
print("Llama Response:")
print(response_deepseek)
response_embedding_Llama = model.encode(response_llama).tolist()
cos_scores_llama = util.cos_sim(qestion_embedding, response_embedding_Llama)
print(f"Cosine Similarity Score for response: {cos_scores_llama.item()}\n")