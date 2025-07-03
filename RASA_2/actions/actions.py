# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient

mongodb_con_string='mongodb+srv://kanish:oRlOexAtRg7Atm3a@interncluster.sm2acna.mongodb.net/?retryWrites=true&w=majority&appName=InternCluster'
db_name_super = 'supercar'
collection_name_super = 'supercar_collection'
index_name_super = 'supercar_index'

db_name_a = 'Cars'
collection_name_a = 'americanCars'
index_name_a = 'vector_index'

import requests
from pydantic import BaseModel

class UserInput(BaseModel):
    content: str
    
def llama_ai(user_input: UserInput) -> str:
    API_KEY = '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'  
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",       
        "X-Title": "Terminal Llama CLI"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick:free",
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

class ActionSupercarInfor(Action):

    def name(self) -> Text:
        return "action_supercar_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    
        query = tracker.latest_message.get("text")

        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

        #connecting to MongoDB
        client=MongoClient(mongodb_con_string)
        collection = client[db_name_super][collection_name_super]

        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection, 
            index_name=index_name_super
        )   
        k=3
        docs = vector_store.max_marginal_relevance_search(query, K=k)
        model= SentenceTransformer("BAAI/bge-base-en")

        required_chunks = [docs[i].page_content for i in range(k)]

        content = "\n\n".join(required_chunks)

        prompt =f'''
        You are a SuperCar bot. You will answer questions about supercars.
        Answer the users question asked by user according to the content given here. 
        just give the answer, do not repeat the question or say anything else.
        Give the answer only if you get a content. If there is no content just say "I am not sure about that.".
        Make sure you answer from the context only. Dont use your own data or pull up from the internet.
        If it is required you can give the answers in bulletins or points.

        user QUESTION: {query}
        content: {content}
        Answer: 

        '''


        response_llama = llama_ai(prompt)
        qestion_embedding = model.encode(query)
        for line in response_llama.strip().splitlines():
            if line.strip():
                dispatcher.utter_message(text=line.strip())

        return []
    
class ActionAmericanCar(Action):
    def name(self) -> Text:
        return "action_car_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        query = tracker.latest_message.get("text")

        embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
        client=MongoClient(mongodb_con_string)
        collection = client[db_name_a][collection_name_a]

        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection, 
            index_name=index_name_a
        )   
        k=3
        docs = vector_store.max_marginal_relevance_search(query, K=k)
        model= SentenceTransformer("BAAI/bge-base-en")

        required_chunks = [docs[i].page_content for i in range(k)]

        content = "\n\n".join(required_chunks)
        prompt =f'''
        You are a American Car bot. You will answer questions about cars.
        Answer the users question asked by user according to the content given here. 
        just give the answer, do not repeat the question or say anything else.
        Give the answer only if you get a content. If there is no content just say "Iam yet to train on that information".
        Make sure you answer from the context only. Dont use your own data or pull up from the internet.
        If it is required you can give the answers in bulletins or points.
        Make sure the answer is like how humans would interact.
        
        user QUESTION: {query}
        content: {content}
        Answer: 

        '''


        response_llama = llama_ai(prompt)
        qestion_embedding = model.encode(query)
        for line in response_llama.strip().splitlines():
            if line.strip():
                dispatcher.utter_message(text=line.strip())

        return []