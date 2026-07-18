from langchain_groq import ChatGroq
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api ket not found in .env")

llm = ChatGroq(model="llama-3.3-70b-versatile")

class JudgeResponse(BaseModel):
    enough_context: bool
    reason: str
    missing_info: str

judge_llm = llm.with_structured_output(JudgeResponse)

response = judge_llm.invoke("""
Question:
What is AI?

Context:
Artificial Intelligence is the simulation of human intelligence by machines.

Is this enough context?
""")

print(response)