#stategraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class ChainState(TypedDict):
    job_des : str
    resume_summary : str
    cover_letter : str

def gen_resume(state: ChainState) -> ChainState:
    prompt = f"Generate a resume summary for the following job description: {state['job_des']}"
    state['resume_summary'] = llm.invoke(prompt).text
    return state

def gen_cover_letter(state: ChainState) -> ChainState:
    prompt = f"Generate a cover letter for the following job description: {state['job_des']} and resume summary: {state['resume_summary']}"
    state['cover_letter'] = llm.invoke(prompt).text
    return state

workflow = StateGraph(ChainState)
workflow.add_node("generate_resume_summary", gen_resume)
workflow.add_node("generate_cover_letter", gen_cover_letter)
workflow.add_edge("generate_resume_summary", "generate_cover_letter")

workflow.set_entry_point("generate_resume_summary")
workflow.set_finish_point("generate_cover_letter")

app = workflow.compile()

input_state = {"job_des": "we are a looking for a AI Engineer with experience in Python and Machine Learning and experiences in deploying AI models in Production."}

result = app.invoke(input_state)

print(result)