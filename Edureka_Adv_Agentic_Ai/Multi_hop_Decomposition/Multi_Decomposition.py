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

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api key not found in .env file")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# vector store object
docker = Chroma(
    persist_directory="db/docker",
    embedding_function=embeddings
)

ai_agent = Chroma(
    persist_directory="db/ai_agent",
    embedding_function=embeddings
)

aws = Chroma(
    persist_directory="db/aws",
    embedding_function=embeddings
)

# all retrival dictionary

retrival = {
    "docker": docker.as_retriever(
        search_kwargs = {"k":4}
    ),

    "ai_agent": ai_agent.as_retriever(
        search_kwargs ={"k":4}
    ),

    "aws": aws.as_retriever(
        search_kwargs ={"k":4}
    )
}

class GraphState(TypedDict, total=False):

    original_query : str
    user_query : str
    retrival_name : list[str]
    sub_queries : list[str]
    all_document : list[Document]
    select_ret : list[str]

class Decomposition_query(BaseModel):
    sub_query: list[str] = Field(
        description="Ordered sub-questions required to answer the original question"
    )

decom_llm = llm.with_structured_output(Decomposition_query)

decompose_prompt = """
You are an expert at decomposing complex questions for multi-hop retrieval.

User Query:
{query}

Generate an ordered list of sub-questions.

Rules:
1. Each sub-question must answer one unique information need.
2. Do not create overlapping or redundant questions.
3. Arrange the questions in logical dependency order.
4. Later questions may rely on information from earlier ones.
5. Do not generate paraphrases.
6. Generate the minimum number of questions needed.
7. Return only the list of sub-questions.
"""

def decompose_query(state : GraphState):

    prompt = decompose_prompt.format(
        query = state["user_query"]
        )

    response = decom_llm.invoke(prompt)

    state["sub_queries"] = response.sub_query
    print("decompose query updated to state")
    print(response.sub_query)

    return state

class RouterRespones(BaseModel):
    select_ret : List[
        Literal[
            "docker",
            "ai_agent",
            "aws"
        ]
    ]

router_llm = llm.with_structured_output(RouterRespones)

router_prompt = """
    Available retrievers:

    - docker_vs
    - ai_agent_vs
    - aws
    
    User query:

    {query}

    Return the most appropriate retriever(s).
"""

def Route_Query(state : GraphState):

    hop_result = []

    for hop, que in enumerate(state["sub_queries"], start=1):
        r_prompt = router_prompt.format(
            query = que
        )
        select_re = router_llm.invoke(r_prompt)

        print(select_re.select_ret)

        re = select_re.select_ret

        for i in re:

            ret = retrival[i]

            result = ret.invoke(que)

            hop_result.append({
                "hop": hop,
                "query": que,
                "selected_retrieval": re,
                "retriever": i,
                "docs" : result
            })
            
    return hop_result

state = {
    "original_query" : input("/n enter user query")
}

state["user_query"] = state["original_query"]

state = decompose_query(state)

all_retrival = Route_Query(state)

def show_retrieval_results(results):
    for retrieval in results:
        print("=" * 100)
        print(f"Retrieval #{retrieval['hop']}")
        print(f"Query     : {retrieval['query']}")
        print(f"All Retrieval : {retrieval['retriever']}")
        print(f"Selected Retriever : {retrieval['selected_retrieval']}")
        print(f"Documents : {len(retrieval['docs'])}")

        for i, doc in enumerate(retrieval["docs"], start=1):
            print(f"\n[{i}] {doc.metadata.get('book')} | Page {doc.metadata.get('page')}")
            print(doc.page_content[:150].replace("\n", " ") + "...")

show_retrieval_results(all_retrival)