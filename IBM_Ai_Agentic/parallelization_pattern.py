from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from IPython.display import Image, display


load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class State(TypedDict):
    text: str
    spanish: str
    french: str
    japanese: str
    combined_output: str

def translate_to_spanish(state: State) -> State:
    prompt = f"Translate the following text to Spanish: {state['text']}"
    response = llm.invoke(prompt)
    return {
        "spanish": response.content
    }

def translate_to_french(state: State) -> State:
    prompt = f"Translate the following text to French: {state['text']}"
    response = llm.invoke(prompt)
    return {
        "french": response.content
    }    

def translate_to_japanese(state: State) -> State:
    prompt = f"Translate the following text to Japanese: {state['text']}"
    response = llm.invoke(prompt)
    return {
        "japanese": response.content
    }    

def aggregate_translations(state: State) -> dict:
    combined = f'Original text {state['text']}\n\n'
    combined += f'Translation to Spanish: {state['spanish']}\n\n'
    combined += f'Translation to French: {state['french']}\n\n'
    combined += f'Translation to Japanese: {state['japanese']}\n\n'
    return {
        "combined_output": combined
    }

workflow = StateGraph(State)
workflow.add_node("translate_to_spanish", translate_to_spanish)
workflow.add_node("translate_to_french", translate_to_french)
workflow.add_node("translate_to_japanese", translate_to_japanese)
workflow.add_node("aggregate_translations", aggregate_translations)

workflow.add_edge(START, "translate_to_spanish")
workflow.add_edge(START, "translate_to_french")
workflow.add_edge(START, "translate_to_japanese")

workflow.add_edge("translate_to_french", "aggregate_translations")
workflow.add_edge("translate_to_spanish", "aggregate_translations")
workflow.add_edge("translate_to_japanese", "aggregate_translations")    

workflow.add_edge("aggregate_translations", END)



app = workflow.compile()
png_data = app.get_graph().draw_mermaid_png()

with open("parallel_graph.png", "wb") as f:
    f.write(png_data)

print("Graph saved as parallel_graph.png")

input_text = {
    'text': "Hello, how are you, how can i help you today?"
}

result = app.invoke(input_text)

print(result)