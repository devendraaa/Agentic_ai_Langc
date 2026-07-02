import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_groq import ChatGroq

def build_llm_runnable() -> RunnableLambda:

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    

    llm = ChatGroq(
    model="llama-3.3-70b-versatile"
   )
    
    def call_groq(prompt: str) -> str:
        response = llm.generate_content(prompt)
        if hasattr(response, "content"):
            return response.text
        
        return str(response)
    
    return RunnableLambda(call_groq)
