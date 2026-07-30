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
from collections import defaultdict

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
    hop_result : list[dict]
    fused_result: list[dict]
    all_evidence: list[dict]
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

    - docker
    - ai_agent
    - aws
    
    User query:

    {query}

    Return the most appropriate retriever(s).
"""

def Route_Query(state : GraphState):

    hop_res = []

    for hop, que in enumerate(state["sub_queries"], start=1):
        r_prompt = router_prompt.format(
            query = que
        )
        select_re = router_llm.invoke(r_prompt)

        print(select_re.select_ret)

        re = select_re.select_ret

        retts = []

        for i in re:

            ret = retrival[i]

            result = ret.invoke(que)

            retts.append({
                "retriever" : i,
                "docs" : result
            })

        hop_res.append({
                "hop": hop,
                "query": que,
                "retrievers": retts
            })
        
    state["hop_result"] = hop_res
            
    return state



# state = decompose_query(state)

# all_retrival = Route_Query(state)

def rrf_fusion(state: GraphState, k=60):

    fused_results = []

    for hop in state["hop_results"]:

        retrievals = hop["retrievers"]

        # If only one retriever, no fusion needed
        if len(retrievals) == 1:

            docs = retrievals[0]["docs"]

            hop["rrf_docs"] = [
                {
                    "document": doc,
                    "score": None
                }
                for doc in docs
            ]

            fused_results.append(hop)
            continue

        rrf_scores = defaultdict(float)
        document_lookup = {}

        # Calculate RRF scores
        for retrieval in retrievals:

            docs = retrieval["docs"]

            for rank, doc in enumerate(docs, start=1):

                # Unique document identifier
                doc_id = (
                    doc.metadata.get("source", "")
                    + "_"
                    + str(doc.metadata.get("page", ""))
                    + "_"
                    + str(rank)
                )

                rrf_scores[doc_id] += 1 / (k + rank)

                document_lookup[doc_id] = doc

        # Sort by RRF score
        ranked_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        fused_docs = []

        for doc_id, score in ranked_docs:

            fused_docs.append(
                {
                    "document": document_lookup[doc_id],
                    "score": score,
                    "source": document_lookup[doc_id].metadata.get("source"),
                    "page": document_lookup[doc_id].metadata.get("page")
                }
            )

        hop["rrf_docs"] = fused_docs

        fused_results.append(hop)

    state["fused_result"] = fused_results

    return state

# f_result = rrf_fusion(all_retrival)

def print_rrf_results(rrf_results):

    for hop in rrf_results:

        print("=" * 100)
        print(f"Hop : {hop['hop']}")
        print(f"Query : {hop['query']}")

        print("\nRRF Ranking:\n")

        for rank, item in enumerate(hop["rrf_docs"], start=1):

            doc = item["document"]
            score = item["score"]

            score_text = "N/A" if score is None else f"{score:.6f}"

            print(f"[{rank}] Score : {score_text}")
            print(f"Source : {doc.metadata.get('source')}")
            print(f"Page   : {doc.metadata.get('page')}")
            print(doc.page_content[:200].replace("\n", " "))
            print("-" * 100)

def aggregate_evidence(state: GraphState):

    all_eviden = []

    for hop in state["fused_result"]:
        for item in hop["rrf_docs"]:
            all_eviden.append(
            {
                "hop": hop["hop"],
                "query": hop["query"],
                "document": item["document"],
                "score": item["score"]
            }
        )

    state["all_evidence"] = all_eviden

    return state
    # return {
    #     "hop_results": rrf_results,
    #     "all_evidence": all_evidence
    # }

# aggre_result = aggregate_evidence(f_result)

def show_retrieval_results(state: GraphState):

    for retrieval in state["all_evidence"]:
        print("=" * 100)
        print(f"Retrieval #{retrieval['hop']}")
        print(f"Query     : {retrieval['query']}")
        # print(f"All Retrieval : {retrieval['retriever']}")
        # print(f"Selected Retriever : {retrieval['selected_retrieval']}")
        print(f"Documents : {len(retrieval['document'])}")
        print(f"Score : {retrieval['score']}")

        for i, doc in enumerate(retrieval["document"], start=1):
            print(f"\n[{i}] {doc.metadata.get('book')} | Page {doc.metadata.get('page')}")
            print(doc.page_content[:150].replace("\n", " ") + "...")


workflow = StateGraph(GraphState)

workflow.add_node("decompose_query", decompose_query)

workflow.add_node("Route_Query", Route_Query)

workflow.add_node("rrf_fusion", rrf_fusion)

workflow.add_node("aggregate_evidence", aggregate_evidence)

workflow.add_node("show_retrieval_results", show_retrieval_results)

workflow.add_edge(START, "decompose_query")

workflow.add_edge("decompose_query", "Route_Query")

workflow.add_edge("Route_Query", "rrf_fusion")

workflow.add_edge("rrf_fusion", "aggregate_evidence")

workflow.add_edge("aggregate_evidence", "show_retrieval_results")

workflow.add_edge("show_retrieval_results", END)

graph = workflow.compile()

state = {
    "original_query" : input("/n enter user query")
}

state["user_query"] = state["original_query"]

from IPython.display import Image, display

try:
    png = graph.get_graph().draw_mermaid_png()
    with open("Multi_decomposition_graph.png", "wb") as f:
        f.write(png)
    print("graph save as graph.png")
except Exception as e:
    print("Error:",e)


