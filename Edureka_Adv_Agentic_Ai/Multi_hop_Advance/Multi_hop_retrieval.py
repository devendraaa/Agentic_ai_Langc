from typing import TypedDict, List
from langchain_core.documents import Document
import ftfy
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
# from data_ingestion import call_data_inge
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api key not found in .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class RouterResponse(BaseModel):
    collection: Literal[
        "docker",
        "aws",
        "ai_agent",
        "langgraph"
        ]
    reason: str

router_llm = llm.with_structured_output(RouterResponse)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ai_agent_db = Chroma(
                    persist_directory="./db/ai_agent",
                    embedding_function=embeddings
)

docker_db = Chroma(
                    persist_directory="./db/docker",
                    embedding_function=embeddings
)

aws_db = Chroma(
                    persist_directory="./db/aws",
                    embedding_function=embeddings
)

langgraph_db = Chroma(
                    persist_directory="./db/langgraph",
                    embedding_function=embeddings
)

retrievers = {

    "docker": docker_db.as_retriever(
        search_kwargs={"k":2}
    ),

    "aws": aws_db.as_retriever(
        search_kwargs={"k":2}
    ),

    "ai_agent": ai_agent_db.as_retriever(
        search_kwargs={"k":2}
    ),

    "langgraph": langgraph_db.as_retriever(
        search_kwargs={"k":2}
    ),
}

class JudgeResponse(BaseModel):

    enough_context: bool = Field(description="True if the retrieved context is sufficient.")

    reason: str = Field(description="Explain why the context is or isn't sufficient.")

    missing_info: str = Field(description="Information still needed.")


class GraphState(TypedDict):

    question: str

    visited_collections: list[str]

    current_query: str

    documents: List[Document]

    answer: str

    retrieval_count: int

    max_hops: int

    enough_context: bool

    missing_info: str

    judge_reason: str

    collection: str


JUDGE_PROMPT = """
You are an expert evaluator.

Question:
{question}

Retrieved Context:
{context}

Determine whether the retrieved context is sufficient to answer the question.

If not, identify what information is still missing.

Return your evaluation.
"""

REWRITE_QUERY_PROMPT = """
    You are an expert search query writer.

    Original Question:
    {question}

    Retrieved Context:
    {context}

    Missing Information:
    {missing_info}

    Generate ONE improved search query to retrieve ONLY the missing information.

    Return ONLY the search query.
"""

ANSWER_PROMPT = """
        You are an enterprise AI assistant.

        Answer the question ONLY using the retrieved context.

        Question:
        {question}

        Context:
        {context}

        Provide a detailed answer.

        If the answer cannot be found,
        say you don't have enough information.
"""

ROUTER_PROMPT = """
    You are an expert routing agent.

    Available collections:

    1. docker
    - Docker
    - Docker Compose

    2. aws
    - EC2
    - IAM
    - S3
    - Lambda

    3. ai_agent
    - AI Agents
    - CrewAI
    - LangChain Agents
    - MCP

    4. langgraph
    - LangGraph
    - StateGraph
    - Nodes

    Question:
    {question}

    Already searched:
    {visited}

    Choose the BEST collection.

    If the required information is likely to be in a collection that has NOT been searched yet,
    prefer that collection.

    Return ONLY one collection name.
"""

def router_node(state: GraphState):

    prompt = ROUTER_PROMPT.format(
        question=state["current_query"],
        visited=", ".join(state["visited_collections"])
    )

    response = router_llm.invoke(prompt)
    print("Routing Reason:", response.reason)

    state["collection"] = response.collection

    if response.collection not in state["visited_collections"]:
        state["visited_collections"].append(response.collection)

    return state

def retrieve_node(state: GraphState):

    print("===============Retrieval Node===============")
    print(f"\nSearching: {state['current_query']}")

    retriver = retrievers[state["collection"]]

    docs = retriver.invoke(state["current_query"])

    print(f"Retrieved {len(docs)} documents")

    # state["documents"].extend(docs)
    state["documents"] = docs
    state["retrieval_count"] += 1
    print(f"Collection: {state['collection']}")

    return state

judge_llm = llm.with_structured_output(JudgeResponse)

def judge_node(state: GraphState):
    print("===============Judge Node===============")

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    context = ftfy.fix_text(context)

    print("content length :", len(context) )
    print("documents length: ", len(state["documents"]))

    prompt = JUDGE_PROMPT.format(
        question=state["question"],
        context=context,
        missing_info=state["missing_info"]
    )

    # response = llm.invoke(prompt)

    print("=" * 80)
    print(prompt)
    print("=" * 80)
    response = judge_llm.invoke(prompt)

    # decision = response.content.strip().upper()

    state["enough_context"] = response.enough_context
    state["judge_reason"] = response.reason
    state["missing_info"] = response.missing_info

    return state

def rewrite_query_node(state: GraphState):
    print("===============Retrieval Query Node===============")

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = REWRITE_QUERY_PROMPT.format(
        question=state["question"],
        context=context,
        missing_info=state['missing_info']
    )

    response = llm.invoke(prompt)

    state["current_query"] = response.content

    return state

def answer_node(state: GraphState):
    print("===============Answer Node===============")

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = ANSWER_PROMPT.format(

        question=state["question"],
        context=context
    )

    response = llm.invoke(prompt)
    print(response)
    print(response.content)
    state["answer"] = response.content

    return state

def route_after_judge(state: GraphState):
    print("===============Route After Node===============")

    # LLM says context is enough
    if state["enough_context"]:
        return "answer"

    # Stop infinite loops
    if state["retrieval_count"] >= state["max_hops"]:
        return "answer"

    return "rewrite"

builder = StateGraph(GraphState)

builder.add_node("retrieve",retrieve_node)

builder.add_node("judge",judge_node)

builder.add_node("rewrite",rewrite_query_node)

builder.add_node("answer",answer_node)

# builder.add_edge(START,"retrieve")
builder.add_node("router", router_node)

builder.add_edge(START, "router")

builder.add_edge("router", "retrieve")

builder.add_edge("retrieve", "judge")

builder.add_conditional_edges(

    "judge",

    route_after_judge,

    {
        "answer": "answer",
        "rewrite": "rewrite",
    }

)

builder.add_edge("rewrite", "router")
builder.add_edge("answer",END)

graph = builder.compile()


from IPython.display import Image, display

try:
    png = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png)
    print("graph save as graph.png")
except Exception as e:
    print("Error:",e)