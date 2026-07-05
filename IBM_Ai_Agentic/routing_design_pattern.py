from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


class RouteState(TypedDict, total=False):
    user_input: str
    task: str
    output: str


class Router(BaseModel):
    role: Literal["summarize", "translate"] = Field(
        description="The task requested by the user"
    )


llm_router = llm.with_structured_output(Router)


def router_node(state: RouteState) -> RouteState:

    router_prompt = f"""
    You are an AI task classifier.

    Decide whether the user wants to:

    - summarize a passage
    - translate text into French

    User input:
    {state['user_input']}
    """

    response = llm_router.invoke(router_prompt)

    print("Router response:", response)

    return {
        **state,
        "task": response.role
    }


def router(state: RouteState) -> str:
    return state["task"]


def summarize_node(state: RouteState) -> RouteState:

    prompt = f"""
    Summarize the following passage:

    {state['user_input']}
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "output": response.content
    }


def translate_node(state: RouteState) -> RouteState:

    prompt = f"""
    Translate the following passage into French:

    {state['user_input']}
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "output": response.content
    }


workflow = StateGraph(RouteState)


workflow.add_node("router_node", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)


workflow.add_edge(START, "router_node")


workflow.add_conditional_edges(
    "router_node",
    router,
    {
        "summarize": "summarize",
        "translate": "translate"
    }
)


workflow.add_edge("summarize", END)
workflow.add_edge("translate", END)


app = workflow.compile()


input_state = {
    "user_input": """
    Can you translate the following passage into French:
    'Hello, how are you?'
    """
}


result = app.invoke(input_state)

print(result)