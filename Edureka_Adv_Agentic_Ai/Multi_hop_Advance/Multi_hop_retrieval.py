from typing import TypedDict, List
from langchain_core.documents import Document
import ftfy
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Literal
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

load_dotenv()

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

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
        search_kwargs={"k":4}
    ),

    "aws": aws_db.as_retriever(
        search_kwargs={"k":4}
    ),

    "ai_agent": ai_agent_db.as_retriever(
        search_kwargs={"k":4}
    ),

    "langgraph": langgraph_db.as_retriever(
        search_kwargs={"k":4}
    ),
}

ALL_COLLECTIONS = [
    "docker",
    "aws",
    "ai_agent",
    "langgraph"
]

class JudgeResponse(BaseModel):

    enough_context: bool = Field(description="True if the retrieved context is sufficient.")

    reason: str = Field(description="Explain why the context is or isn't sufficient.")

    missing_info: str = Field(description="Information still needed.")


class GraphState(TypedDict):

    question: str

    remaining_collections: list[str]   # <-- ADD THIS

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
    You are evaluating retrieved knowledge.

    Question:
    {question}

    Context:
    {context}

    Your job:

    1. Decide whether the context is sufficient to answer the user's question.

    2. If the answer can be written with reasonable confidence,
    return enough_context=True.

    3. Only return enough_context=False if there is a significant missing concept
    that would materially improve the answer.

    Do not request additional information if the answer can already be written.
"""

class RewriteResponse(BaseModel):
    query: str

from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=RewriteResponse)

REWRITE_QUERY_PROMPT = """
    You are a search query optimizer.

    Original Question:
    {question}

    Missing Information:
    {missing_info}

    Generate exactly ONE search query.

    Rules:
    - Return only the search query.
    - No explanation.
    - No markdown.
    - No examples.
    - No additional text.
"""

ANSWER_PROMPT = """
You are an expert AI assistant.

If multiple retrieved chunks describe the same step,
combine them into one concise explanation.

Do not repeat the same workflow or instructions.

Remove redundant information before answering.

Answer the user's question using the retrieved information.

Rules:

- Write the answer entirely in your own words.
- Never copy complete sentences from the context.
- Ignore copyright notices, headers, page numbers, chapter titles, and incomplete fragments.
- If multiple documents provide complementary information, combine them into one coherent explanation.
- Structure the answer in logical steps.
- Do not mention the retrieved context.

Question:
{question}

Context:
{context}
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

    remaining = [
        c
        for c in ALL_COLLECTIONS
        if c not in state["visited_collections"]
    ]

    state["remaining_collections"] = remaining

    if not remaining:
        return state

    prompt = f"""
    You are a routing agent.

    Available collections:
    {", ".join(remaining)}

    IMPORTANT:
    You MUST choose exactly one collection from the Available collections list.
    Choosing any collection not in the list is an error.

    Question:
    {state["current_query"]}
    """

    print("=" * 50)
    print("Remaining before routing:", remaining)
    print("Visited before routing:", state["visited_collections"])

    response = router_llm.invoke(prompt)

    if response.collection not in remaining:
        raise Exception(
            f"""
    INVALID ROUTER OUTPUT

    Remaining : {remaining}

    Returned  : {response.collection}
    """
        )

    print("LLM selected:", response.collection)

    print("Routing Reason:", response.reason)

    state["collection"] = response.collection

    

    state["visited_collections"].append(response.collection)

    print("Visited:", state["visited_collections"])
    print("Remaining:", remaining)

    return state

def retrieve_node(state: GraphState):

    print("===============Retrieval Node===============")
    print(f"\nSearching: {state['current_query']}")

    retriver = retrievers[state["collection"]]

    docs = retriver.invoke(state["current_query"])

    existing_docs = {}

    for doc in state["documents"]:
        key = (
            doc.metadata.get("book"),
            doc.metadata.get("page"),
            doc.page_content
        )
        existing_docs[key] = doc

    for doc in docs:
        key = (
            doc.metadata.get("book"),
            doc.metadata.get("page"),
            doc.page_content
        )
        existing_docs[key] = doc

    state["documents"] = list(existing_docs.values())

    state["retrieval_count"] += 1
    print(f"Collection: {state['collection']}")

    return state

judge_llm = llm.with_structured_output(JudgeResponse)

def rerank_node(state: GraphState):

    print("===============Rerank Node===============")

    query = state["question"]

    docs = state["documents"]

    if len(docs) <= 1:
        return state

    # Create (query, document) pairs
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    # Score every document
    scores = reranker.predict(pairs)

    # Sort by score (highest first)
    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    print("\nTop Reranked Documents\n")

    for score, doc in ranked_docs:
        print(f"Score: {score:.4f}")
        print(doc.metadata)
        print("-" * 60)

    # Keep only the best 5
    best_docs = []

    seen_books = set()

    for score, doc in ranked_docs:

        book = doc.metadata["book"]

        if book not in seen_books:
            best_docs.append(doc)
            seen_books.add(book)

        if len(best_docs) == 5:
            break

        state["documents"] = best_docs

    return state

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

    
    response = judge_llm.invoke(prompt)

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
        missing_info=state['missing_info']
    )

    # rewrite_llm = llm.with_structured_output(RewriteResponse)

    # response = rewrite_llm.invoke(prompt)
    response = llm.invoke(prompt)
    # print("this is response text from rewrite query node: /n", response.content)

    state["current_query"] = response.content.strip()
    print(state["current_query"])

    return state

def answer_node(state: GraphState):
    print("===============Answer Node===============")

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    for i, doc in enumerate(state["documents"]):
        print("=" * 60)
        print(f"Chunk {i+1}")
        print(doc.metadata)
        print(doc.page_content[:500])

    prompt = ANSWER_PROMPT.format(

        question=state["question"],
        context=context
    )

    response = llm.invoke(prompt)
    state["answer"] = response.content

    return state

def route_after_judge(state: GraphState):

    if state["enough_context"]:
        return "rerank"

    if not state["remaining_collections"]:
        return "rerank"

    if state["retrieval_count"] >= state["max_hops"]:
        return "rerank"

    return "rewrite"

builder = StateGraph(GraphState)

builder.add_node("router", router_node)

builder.add_node("retrieve",retrieve_node)

builder.add_node("judge", judge_node)

builder.add_node("rerank",rerank_node)

builder.add_node("rewrite",rewrite_query_node)

builder.add_node("answer",answer_node)



# Adding egde to node--

builder.add_edge(START, "router")

builder.add_edge("router", "retrieve")

builder.add_edge("retrieve", "judge")

builder.add_conditional_edges(

    "judge",

    route_after_judge,

    {
        "rewrite": "rewrite",
        "rerank": "rerank",
    }

)

builder.add_edge("rewrite", "router")
builder.add_edge("rerank","answer")
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