from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import mongo_det

#file loading
filepath="RMI_Supercars_Coming_Light_Vehicle_Revolution_1993.pdf"
loader=PyPDFLoader(filepath)
pages=loader.load()

#file chunking
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators= ["\n\n", "\n", ".", "?", "!", " ", ""]
)

docs=text_splitter.split_documents(pages)

embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

#connecting to MongoDB
client=MongoClient(mongo_det.mongodb_con_string)
collection = client[mongo_det.db_name][mongo_det.collection_name]

collection.delete_many({})  # Clear existing data

#storing

vector_store = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, 
    collection=collection, 
    index_name=mongo_det.index_name
)