from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

import os
groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0.5,
    model_name="llama3-8b-8192",
    groq_api_key=groq_key
)
