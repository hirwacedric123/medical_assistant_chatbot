from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os
import pinecone
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"



#Creating Embeddings for Each of The Text Chunks & storing
docsearch= PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embeddings = embeddings,
)
