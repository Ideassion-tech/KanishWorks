from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import chromadb
from langchain.vectorstores import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import requests
from pydantic import BaseModel

import os
import glob

class UserInput(BaseModel):
    content: str

def llama_ai(user_input: UserInput) -> str:
    API_KEY = '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'  # 🔐 Replace with your actual key
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Rasa RAG Car Assistant"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
        "messages": [
            {
                "role": "user",
                "content": user_input.content
            }
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"[Error {response.status_code}]: {response.text}"
    except Exception as e:
        return f"[Exception]: {str(e)}"

mongodb_con_string='mongodb+srv://kanish:oRlOexAtRg7Atm3a@interncluster.sm2acna.mongodb.net/?retryWrites=true&w=majority&appName=InternCluster'
db_name = 'supercar'
index_name = 'supercar_index'
collection_name = 'supercar_collection'

def query_mongo(query):
    embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

    #connecting to MongoDB
    client=MongoClient(mongodb_con_string)
    collection = client[db_name][collection_name]

    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=collection, 
        index_name=index_name
    )
    docs = vector_store.max_marginal_relevance_search(query,k=5)
    return docs

class ActionRAGAnswer(Action):
    def name(self) -> Text:
        return "action_rag_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
 

        
        
        query = tracker.latest_message.get("text")
        docs = query_mongo(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""You are a helpful car assistant. Use the following context to answer the user's question.

Context:
{context}

User question: {query}

Answer:

Do not answer if the context does not contain any information related to the question. Just say: I don't know."""

        user_input = UserInput(content=prompt)
        response = llama_ai(user_input)
        dispatcher.utter_message(text=response)
        return []
