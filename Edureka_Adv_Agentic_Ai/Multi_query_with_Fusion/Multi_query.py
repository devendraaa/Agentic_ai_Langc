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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

class GraphState(TypedDict):

    user_query : str

    retrieval_name : list[str]

    all_documents : list[Document]

    select_ret : list[str]

class RouterResponse(BaseModel):
    select_ret : List[
        Literal[
        "docker_vs",
        "ai_agent_vs"
        ]
    ]

    sel_ret : str

router_llm = llm.with_structured_output(RouterResponse)

router_prompt = """
    Available retrievers:

    - docker_vs
    - ai_agent_vs
    
    User query:

    {query}

    Return the most appropriate retriever(s).
"""

def Router_node(state: GraphState):

    prompt = router_prompt.format(
        query = state['user_query']
    )

    selecte_ret = router_llm.invoke(prompt)

    return selecte_ret

class Multi_query(BaseModel):
    query : list[str] = Field(description="return in string of multi query in list")

Multi_llm = llm.with_structured_output(Multi_query)

multi_prompt = """
       you are a Multi query generator engineer,
       {query}
       regenerate query and return this query in 3 robust query in different ways.
       Multi query must be in string type and clean.
    """

def Multi_query_generator(state: GraphState):

    m_prompt = multi_prompt.format(
        query = state["user_query"]
    )

    multi_query = Multi_llm.invoke(m_prompt)
    return multi_query

state = {
    "user_query" : input("enter user query")
}

m_query = Multi_query_generator(state)
print("new query generated base on simple one query", m_query)

docker_vs = Chroma(
    persist_directory="Multi_hop_Advance/db/docker",
    embedding_function=embeddings
)

ai_agent_vs = Chroma(
    persist_directory="Multi_hop_Advance/db/ai_agent",
    embedding_function=embeddings
) 

retrievers = {

    "docker_vs": docker_vs.as_retriever(
        search_kwargs={"k":4}
    ),

    "ai_agent_vs": ai_agent_vs.as_retriever(
        search_kwargs={"k":4}
    ),

}

for query in m_query.query:
    state["user_query"] = query

    router = Router_node(state=state)

    print(router)


all_results = []

for retriever_name in router.select_ret:

    docs = retrievers[retriever_name].invoke(query)

    all_results.append({

        "query": query,

        "retriever": retriever_name,

        "docs": docs

    })
