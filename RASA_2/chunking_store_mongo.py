from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import os
from pypdf import PdfReader
from langchain.schema import Document
db_name_a = 'Cars'
collection_name_a = 'americanCars'
index_name_a = 'vector_index'
mongodb_con_string='mongodb+srv://kanish:oRlOexAtRg7Atm3a@interncluster.sm2acna.mongodb.net/?retryWrites=true&w=majority&appName=InternCluster'

#file loading
pdf_folder=f"RASA_2\docs2"


text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators= ["\n\n", "\n", ".", "?", "!", " ", ""]
)

docs=[]

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Reading {filename}")
        reader = PdfReader(pdf_path)

        # Extract text from all pages
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Split the text into chunks
        chunks = text_splitter.split_text(full_text)
        
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"source_file": filename}
            ))

print(docs)
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

#connecting to MongoDB
client=MongoClient(mongodb_con_string)
collection = client[db_name_a][collection_name_a]

collection.delete_many({})  # Clear existing data

#storing

vector_store = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, 
    collection=collection, 
    index_name=index_name_a
)