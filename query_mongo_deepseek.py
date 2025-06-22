from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer, util
import torch
from pymongo import MongoClient
import mongo_det
import requests
from pydantic import BaseModel

class UserInput(BaseModel):
    content: str

def deepseek_ai(user_input: UserInput) -> str:
    API_KEY = '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'  
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",       
        "X-Title": "Terminal DeepSeek CLI"
    }

    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            deepseektext = UserInput(content=result)
            return deepseektext.content
        else:
            return f"[Error {response.status_code}]: {response.text}"
    except Exception as e:
        return f"[Exception]: {str(e)}"


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
qestion_embedding = model.encode(query).tolist()
for i in range(0,k):
    print(i)
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


response = deepseek_ai(prompt)
print(response)

response_embedding = model.encode(response).tolist()
cos_scores = util.cos_sim(qestion_embedding, response_embedding)
print(f"Cosine Similarity Score for response: {cos_scores.item()}\n")