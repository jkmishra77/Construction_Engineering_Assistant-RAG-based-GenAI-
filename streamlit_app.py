# streamlit_app.py
import os
from dotenv import load_dotenv
import streamlit as st
from dotenv import load_dotenv
from backend.llm import llm
from pinecone import Pinecone

from backend.query import query_vector
from backend.embedding import embeddings
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

st.set_page_config(page_title="GenAI Chat with RAG", layout="wide")
st.title("  Engineering Assistatnt (RAG-enabled GenAI Chatbot)")
st.subtitle(" In this Demoyou acn chat with my CV you can compy and paste questions like" \
"1. tell me the Professional Summary"
"2. What are Projects (Live Demos)"
"3. what the Skills & Expertise in LLMs & Generative AI"
"4. what the Awards & Certifications Dr Mishra awarded  ")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

# Input box
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.chat_message("user").write(user_input)

    # Step 1: Embed user query
    embedded_vector = embeddings.embed_query(user_input)

    # Step 2: Query Pinecone
    results = query_vector(embedded_vector, top_k=3)
    #print(results)
    context_chunks = [
    match.metadata["text"]
    for match in results.matches
    if match.metadata and "text" in match.metadata
]
    context_text = "\n".join(context_chunks)

    # Step 3: Construct full prompt
    full_prompt = f"Context:\n{context_text}\n\nUser: {user_input}"
    st.session_state.messages.append(HumanMessage(content=full_prompt))

    # Step 4: Get LLM response
    response = llm.invoke(st.session_state.messages)
    st.session_state.messages.append(response)
    st.chat_message("assistant").write(response.content)
