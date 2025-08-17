import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

def query_vector(vector, top_k=3):
    return index.query(
        #namespace="default",
        vector=vector, 
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
