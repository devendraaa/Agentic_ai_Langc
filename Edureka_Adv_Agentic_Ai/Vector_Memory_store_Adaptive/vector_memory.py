from typing import TypedDict, List
from annotated_types import doc
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
import uuid
from datetime import datetime

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

DEBUG = False

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

few_shot = Chroma(
    persist_directory="db/db/few_shot",
    embedding_function=embeddings
)
# all retrival dictionary

working_memory_vs = Chroma(
    persist_directory="db/memory_vector",
    embedding_function=embeddings
)

retrival = {
    "docker": docker.as_retriever(
        search_kwargs = {"k":4}
    ),

    "ai_agent": ai_agent.as_retriever(
        search_kwargs ={"k":4}
    ),

    "aws": aws.as_retriever(
        search_kwargs ={"k":4}
    ),

    "few_shot": few_shot.as_retriever(
        search_kwargs ={"k":3}
    )
}

class GraphState(TypedDict, total=False):

    original_query : str
    user_query : str
    retrival_name : list[str]
    sub_queries : list[str]
    hop_results : list[dict]
    fused_result: list[dict]
    all_evidence: list[dict]
    select_ret : list[str]
    reranked_evidence: list[dict]

    query_type : str

    # memory layer
    conversation_id : str
    conversation_history : list
    # retrieval_memory : list[Document]
    turn : int
    hybrid_memory : str
    semantic_memory : list[Document]
    episodic_memory : list[Document]
    working_memory : list[Document]

    # few-shot-retrieval
    few_shot_examples : list[Document]

    # context builder
    context : str

    #prompt builder
    final_prompt : str
    final_answer : str

    #context compression
    compressed_evidence : list[dict]

    #memory orchestration
    memory_plan : list[dict]

    #query rewriter using memory layer
    rewritten_query : str 

QUERY_REWRITE_PROMPT = """
    You are an expert conversational query rewriting assistant.

    Your ONLY task is to rewrite the user's latest question into a complete,
    standalone question.

    You MUST use the conversation memory when necessary.

    Rules

    1. Produce a concise standalone retrieval query.

    2. Remove conversational words such as:

    - I want
    - I would like
    - also
    - please
    - as an engineer
    - as a beginner

    unless they change the meaning.

    3. Keep only information useful for retrieval.

    4. Never answer the question.

    5. Never decompose the question.

    6. Return exactly one rewritten question.

    Return ONLY the rewritten question.

    --------------------------------------------------

    Conversation Memory

    {memory}

    --------------------------------------------------

    Current User Question

    {question}

    --------------------------------------------------

    Standalone Question
"""

def query_rewriter(state: GraphState):

    prompt = QUERY_REWRITE_PROMPT.format(
        memory = state["hybrid_memory"],
        question = state["user_query"]
    )

    response = llm.invoke(prompt)

    state["rewritten_query"] = response.content.strip()

    print("\n Rewritten Query:")
    print(state["rewritten_query"])

    return state

MEMORY_KEYWORDS = {
    "continue",
    "previous",
    "earlier",
    "last",
    "remember",
    "again",
    "same",
    "before",
    "our"
}

PREFERENCE_KEYWORDS = {
    "prefer",
    "favorite",
    "usually",
    "always",
    "my",
    "preference"
}

def memory_orchestrator(state: GraphState):

    query = state["user_query"].lower()

    plan = {
        "working": True,
        "episodic": False,
        "semantic": False
    }

    if any(k in query for k in MEMORY_KEYWORDS):
        plan["episodic"] = True

    if any(k in query for k in PREFERENCE_KEYWORDS):
        plan["semantic"] = True

    state["memory_plan"] = plan

    print("\nMemory Plan")
    print(plan)

    return state

class Query_classifier(BaseModel):
    query_type : Literal[
        "simple",
        "multi_hop"
    ]

query_classifier_prompt = """
    You are an expert query classifier.

    Determine whether the user question is:

    simple
    - Can be answered with one retrieval.

    multi_hop
    - Requires multiple reasoning steps.
    - Needs information from multiple topics.
    - Requires decomposition.

    User Question:

    {query}

    Return only:
    simple
    or
    multi_hop
"""

query_classifier_llm = llm.with_structured_output(Query_classifier)

def update_memory(state: GraphState):

    memory_doc = Document(
        page_content=f"""
        User Query: {state['user_query']}

        Assistant Answer: {state['final_answer']}

        """.strip(),

        metadata={
            "memory_id": str(uuid.uuid4()),
            "conversation_id": state["conversation_id"],
            "turn": state["turn"],
            "role": "conversation",
            "timestamp": datetime.now().isoformat(),
            "memory_type": "working memory",
            "importance": 1.0
        }

    )

    working_memory_vs.add_documents([memory_doc])
    state["turn"] += 1
    print("\n Working Memory updated with new document")

    return state

def memory_retrieval(state: GraphState):

    docs = working_memory_vs.similarity_search(

        query = state["user_query"],
        k=4,
        filter = {
            "conversation_id": state["conversation_id"]
        }
    )

    state["working_memory"] = docs
    print("\n Working Memory Retrieval")

    for i , doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Document {i}:")
        print(doc.page_content)
        print("=" * 80)

    return state

def retrieval_fewshot_examples(state: GraphState):

    docs = retrival["few_shot"].invoke(state["user_query"])
    print("\n few-shot length of docs:", len(docs))

    state["few_shot_examples"] = docs

    # for doc in docs:
    #     print(doc.page_content)
    #     print("=" * 80)

    return state

def classify_query(state: GraphState):
    prompt = query_classifier_prompt.format(
        query = state["user_query"]
    )

    response = query_classifier_llm.invoke(prompt)

    state["query_type"] = response.query_type

    print("\n query type updated to state")
    print(response.query_type)

    return state

def query_router(state: GraphState):
    if state["query_type"] == "simple":
        print("\n simple query detected")
        return "simple"

    return "multi_hop"

class Decomposition_query(BaseModel):
    sub_query: list[str] = Field(
        description="Ordered sub-questions required to answer the original question"
    )

decom_llm = llm.with_structured_output(Decomposition_query)

decompose_prompt = """
    You are an expert Query Decomposition Assistant.

    Your job is to convert the user's question into the minimum number of
    independent retrieval queries.

    Rules:

    Before generating sub-queries,
    determine whether the current question depends on previous conversation.

    If yes,

    rewrite the question into a standalone question by explicitly including
    the missing entity from the conversation memory.

    Examples

    Conversation

    User:
    Explain AI Agent.

    Current Question

    How do I deploy to AWS?

    Rewrite

    How do I deploy an AI Agent to AWS?

    --------------------------------

    Conversation

    User:
    Teach Docker.

    Current Question

    How do I deploy it?

    Rewrite

    How do I deploy Docker containers?

    --------------------------------------------------

    Conversation Memory

    {memory}

    --------------------------------------------------

    Few-shot Examples

    {few_shot_examples}

    --------------------------------------------------

    Current User Question

    {query}
"""

def decompose_query(state: GraphState):

    few_shot_examples = ""

    for index, doc in enumerate(state["few_shot_examples"], start=1):

        few_shot_examples += f"""
            Example {index}

            Question:
            {doc.page_content}

            Sub Queries:
        """

        for i, q in enumerate(doc.metadata["sub_query"], start=1):
            few_shot_examples += f"\n{i}. {q}"

        few_shot_examples += "\n\n"

    prompt = decompose_prompt.format(
        query = state["rewritten_query"],
        memory = state["hybrid_memory"],
        few_shot_examples = few_shot_examples
    )

    # print("="*120)
    # print(prompt)
    # print("="*120)

    response = decom_llm.invoke(prompt)

    state["sub_queries"] = response.sub_query

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

        for i in list(dict.fromkeys(re)):

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
        
    state["hop_results"] = hop_res
            
    return state

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
                    doc.metadata["source"]
                    + "_"
                    + str(doc.metadata["page"])
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

    print("total length of aggregations: ", len(all_eviden))

    return state

def reranked_document(state: GraphState):
    pairs = []

    for evidence in state["all_evidence"]:
      
        pairs.append((
            state['user_query'],
            evidence['document'].page_content
        )
    )

    # print("cross encoder pairs:", pairs[0])
    scores = reranker.predict(pairs)

    # print("scores :", scores)
    reranked = []

    for evidence, score in zip(state["all_evidence"], scores):

        item = evidence.copy()

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    TOP_K = 2
    MIN_SCORE = -2.0

    filtered = [
        doc
        for doc in reranked
        if doc["rerank_score"] >= MIN_SCORE
    ]

    state["reranked_evidence"] = filtered[:TOP_K]

    print("after filtered :", len(filtered))

    return state

compression_prompt = """ You are an Evidence Extraction system.

    Extract ONLY the exact facts from the document that are useful for answering the user's question.

    Rules:

    1. Never rewrite technical steps.
    2. Never invent missing information.
    3. Never explain.
    4. Never summarize using outside knowledge.
    5. Copy only relevant facts from the document.
    6. Remove copyright, page numbers and headers.
    7. Preserve commands exactly as written.

    Return ONLY the information required to answer the user's question.

    Maximum:
    - 8 bullet points
    OR
    - 120 words

    Do not explain.
    Do not summarize unrelated content.

    User Question

    {question}

    Retrieved Document

    {document}

    Relevant Facts:   
                    """

def context_compression(state: GraphState):
    compressed = []

    for evidence in state["reranked_evidence"]:

        doc = evidence["document"]

        prompt = compression_prompt.format(
            question=state["user_query"],
            document=doc.page_content)
        
        response = llm.invoke(prompt)

        compressed.append({
            "hop": evidence["hop"],
            "query": evidence["query"],
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "score": evidence["score"],
            "compressed_text": response.content.strip()
        })

        state["compressed_evidence"] = compressed

    if not state["reranked_evidence"]:
        print("No compressed evidence found.")
        state["compressed_evidence"] = []
        return state

    return state

def context_builder(state: GraphState):

    MAX_CONTEXT_DOCS = 4

    MAX_CHARS = 500

    context = ""

    # -------------------------
    # Working Memory
    # -------------------------

    for doc in state["working_memory"]:

        context += f"""
            Conversation Memory

            {doc.page_content}

            ----------------------------------------

    """

    # -------------------------
    # Retrieved Documents
    # -------------------------

    context += "===== Retrieved Knowledge =====\n\n"
        # Document  = {i}
            # CrossEncoder Score:
        # {evidence["rerank_score"]:.3f}

    for i, evidence in enumerate(state["compressed_evidence"], start=1):

        # doc = evidence["document"]

        context += f"""

        Evidence {i}:

        Source:
        {evidence['source']}

        Page:
        {evidence['page']}

        Content:
        {evidence['compressed_text']}
                            """

    state["context"] = context

    if DEBUG:
        print("\n\n===== Context Builder Output =====\n")
        print(context)

    return state

def show_retrieval_results(state: GraphState):

    for rank, retrieval in enumerate(state["reranked_evidence"], start=1):

        doc = retrieval["document"]

        print("=" * 100)
        print(f"Rank :{rank}")
        print(f"Hop #{retrieval['hop']}")
        print(f"Query : {retrieval['query']}")
        print(f"RRF Score : {retrieval['score']}")
        print(f"CrossEncoder :{retrieval['rerank_score']}")

        print(f"Source : {doc.metadata.get('source')}")
        print(f"Page   : {doc.metadata.get('page')}")

        print(doc.page_content[:200].replace("\n", " "))
        print("-" * 100)

    return state

def memory_fusion(state: GraphState):

    memory_text = []

    # ======================================================
    # Working Memory
    # ======================================================

    working_docs = state.get("working_memory", [])

    if working_docs:

        memory_text.append(
            "================ WORKING MEMORY ================\n"
        )

        for i, doc in enumerate(working_docs, start=1):

            memory_text.append(
                f"""
        Memory {i}

        {doc.page_content}

        ----------------------------------------
        """
                    )

    # ======================================================
    # Episodic Memory
    # (Future)
    # ======================================================

    episodic_docs = state.get("episodic_memory", [])

    if episodic_docs:

        memory_text.append(
            "\n================ EPISODIC MEMORY ================\n"
        )

        for i, doc in enumerate(episodic_docs, start=1):

            memory_text.append(
                f"""
        Episode {i}

        {doc.page_content}

        ----------------------------------------
        """
                    )

    # ======================================================
    # Semantic Memory
    # (Future)
    # ======================================================

    semantic_docs = state.get("semantic_memory", [])

    if semantic_docs:

        memory_text.append(
            "\n================ SEMANTIC MEMORY ================\n"
        )

        for i, doc in enumerate(semantic_docs, start=1):

            memory_text.append(
                f"""
        Knowledge {i}

        {doc.page_content}

        ----------------------------------------
        """
                    )

    state["hybrid_memory"] = "\n".join(memory_text)

    print("=" * 100)
    print("HYBRID MEMORY")
    print("=" * 100)

    return state

ANSWER_PROMPT = """
    You are a Retrieval-Augmented AI Assistant.

    Conversation Memory is only for understanding
    the user's context and previous discussion.

    Do NOT use conversation memory as factual evidence.

    Use Retrieved Knowledge for all factual answers.

    If Retrieved Knowledge is empty,
    respond:

    "I don't have enough information."

    Rules:

    1. Use conversation memory only for conversational context.
    2. Use retrieved knowledge for factual answers.
    3. If the answer is not present in the retrieved knowledge, say:
    "I don't have enough information."
    4. Never invent facts.
    5. Never invent documents.

    ==================================================

    Conversation Memory

    {hybrid_memory}

    ==================================================

    Retrieved Knowledge

    {context}

    ==================================================

    Question

    {question}

    ==================================================

    Final Answer:
"""

def prompt_builder(state: GraphState):

    prompt = ANSWER_PROMPT.format(

        hybrid_memory=state["hybrid_memory"],

        context=state["context"],

        question=state["user_query"]

    )

    print("=" * 80)
    print("Prompt characters:", len(prompt))
    print("Approx tokens:", len(prompt) // 4)
    print("=" * 80)

    state["final_prompt"] = prompt
    return state

def generate_answer(state: GraphState):

    print("=" * 120)
    print("Generating Final Answer...")
    print("=" * 120)

    response = llm.invoke(state["final_prompt"])

    state["final_answer"] = response.content

    return state

def show_final_answer(state: GraphState):

    print("\n")
    print("=" * 120)
    print("FINAL ANSWER")
    print("=" * 120)

    print(state["final_answer"])

    return state

workflow = StateGraph(GraphState)

workflow.add_node("query_rewriter", query_rewriter)

workflow.add_node("memory_orchestrator", memory_orchestrator)

workflow.add_node("memory_retrieval", memory_retrieval)

workflow.add_node("memory_fusion",memory_fusion)

workflow.add_node("retrieval_fewshot_examples", retrieval_fewshot_examples)

workflow.add_node("decompose_query", decompose_query)

workflow.add_node("Route_Query", Route_Query)

workflow.add_node("rrf_fusion", rrf_fusion)

workflow.add_node("aggregate_evidence", aggregate_evidence)

workflow.add_node("reranked_document", reranked_document)

workflow.add_node("context_compression", context_compression)

workflow.add_node("context_builder", context_builder)

workflow.add_node("show_retrieval_results", show_retrieval_results)

workflow.add_node("update_memory", update_memory)

workflow.add_node("prompt_builder", prompt_builder)

workflow.add_node("generate_answer", generate_answer)

workflow.add_node("show_final_answer", show_final_answer)

# adding workflow edges

# workflow.add_edge(START, "classify_query")

# workflow.add_conditional_edges("classify_query", query_router,{
    # "simple": "Route_Query",
    # "multi_hop": "decompose_query"
# })

workflow.add_edge(START, "memory_retrieval")

workflow.add_edge("memory_retrieval", "memory_fusion")

workflow.add_edge("memory_fusion", "query_rewriter")

workflow.add_edge("query_rewriter", "retrieval_fewshot_examples")

workflow.add_edge("retrieval_fewshot_examples", "decompose_query")

workflow.add_edge("decompose_query", "Route_Query")

workflow.add_edge("Route_Query", "rrf_fusion")

workflow.add_edge("rrf_fusion", "aggregate_evidence")

workflow.add_edge("aggregate_evidence", "reranked_document")

workflow.add_edge("reranked_document", "context_compression")

workflow.add_edge("context_compression", "context_builder")

workflow.add_edge("context_builder", "show_retrieval_results")

workflow.add_edge("show_retrieval_results", "prompt_builder")

workflow.add_edge("prompt_builder", "generate_answer")

workflow.add_edge("generate_answer", "update_memory")

workflow.add_edge("update_memory", "show_final_answer")

workflow.add_edge("show_final_answer", END)

graph = workflow.compile()

conversation_id = str(uuid.uuid4())
turn = 1

while True:

    query = input("\n Enter User Query: ")

    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    state = {
        "conversation_id": conversation_id,
        "turn": turn,
        "original_query": query,
        "user_query": query
    }

    result = graph.invoke(state)
    turn += 1

from IPython.display import Image, display

try:
    png = graph.get_graph().draw_mermaid_png()
    with open("Multi_decomposition_graph.png", "wb") as f:
        f.write(png)
    print("graph save as graph.png")
except Exception as e:
    print("Error:",e)


