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

episodic_memory = Chroma(
    persist_directory="db/episodic_memory",
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
    episode: str
    episode_type : str
    importance_score : float
    store_episode : bool

    #query rewriter using memory layer
    rewritten_query : str 

    #intent classifier
    knowledge_retrieval: bool
    memory_update: bool

class IntentQuery(BaseModel):
    knowledge_retrieval: bool
    memory_update: bool

INTENT_PROMPT = """
    You are an Intent Classifier.

    Your task is to determine whether the user's message requires:

    1. Knowledge Retrieval
    2. Memory Update

    Either one or BOTH may be True.

    Examples

    User:
    What is Docker?

    knowledge_retrieval=True
    memory_update=False

    ----------------------------

    User:
    I completed Docker certification.

    knowledge_retrieval=False
    memory_update=True

    ----------------------------

    User:
    I completed Docker certification.
    What should I learn next?

    knowledge_retrieval=True
    memory_update=True

    ----------------------------

    User:
    Thanks!

    knowledge_retrieval=False
    memory_update=False

    ----------------------------

    User Message

    {query}
"""

intent_llm = llm.with_structured_output(IntentQuery)

def Intent_Classifier(state: GraphState):

    prompt = INTENT_PROMPT.format(
        query = state["user_query"]
    )

    response = intent_llm.invoke(prompt)

    state["knowledge_retrieval"] = response.knowledge_retrieval
    state["memory_update"] = response.memory_update

    print("intend classifier Knowledge Retrieval: ", state["knowledge_retrieval"])
    print("intend classifier Memory Retrieval: ", state["memory_update"])

    return state

def route_int_query(state: GraphState):

    if state["knowledge_retrieval"]:
        return "knowledge_pipeline"

    if state["memory_update"]:
        return "memory_pipeline"

    return "no_action"

def route_memory_update(state: GraphState):

    if state["memory_update"]:
        return "update_memory"

    return "finish"

def no_action(state: GraphState):

    state["final_answer"] = ""

    return state

class ImportantScore(BaseModel):

    importance: float = Field(
        description="Importance score between 0 and 1."
    )

    store: bool = Field(
        description="Whether the episode should be stored."
    )

    reason: str = Field(
        description="Reason for the score."
    )

IMPORTANCE_PROMPT = """
    You are an Episodic Memory Importance Evaluator.

    Your task is to decide whether the extracted episode
    should become long-term memory.

    Give a score between 0 and 1.

    Guidelines

    1. Long-term career goals

    0.9+

    2. Completed achievements

    0.9+

    3. Projects

    0.8+

    4. Stable preferences

    0.7+

    5. Learning progress

    0.6+

    6. Temporary questions

    0.1

    7. Greetings

    0.0

    Episode

    {episode}
"""

importance_llm = llm.with_structured_output(ImportantScore)

def importance_scorer(state: GraphState):

    if not state["episode"]:

        state["importance_score"] = 0.0
        state["store_episode"] = False

        return state

    prompt = IMPORTANCE_PROMPT.format(

        episode=state["episode"]

    )

    response = importance_llm.invoke(prompt)

    state["importance_score"] = response.importance

    state["store_episode"] = (
        response.store and response.importance >= 0.7
                )
    
    print("=" * 100)
    print("IMPORTANCE SCORER")

    print("Episode :", state["episode"])

    print("Importance :", response.importance)

    print("Store :", response.store)

    print("Reason :", response.reason)

    return state

def episodic_writer(state: GraphState):

    struc_episodic = [Document(
        page_content = state["episode"],

        metadata = {

            "conversation_id": state["conversation_id"],

            "episode_type": state["episode_type"],

            # "importance": state["importance_score"],

            "created_at": datetime.now().isoformat(),

            "memory_type": "episodic"

        }
    )]

    if state["store_episode"]:

        episodic_memory.add_documents(struc_episodic)

        print("episodic memory stored perfectly")

def episodic_mem_ret(state: GraphState):

    docs = episodic_memory.similarity_search(
        query = state["user_query"],
        k = 5
    )

    state["episodic_memory"] = docs

    print("\n" + "=" * 100)
    print("EPISODIC MEMORY RETRIEVAL")
    print("=" * 100)

    if not docs:
        print("No episodic memory found.")

    for i, doc in enumerate(docs, start=1):

        print(f"\nEpisode {i}:")
        print(doc.page_content)
        print("Type:", doc.metadata.get("episode_type"))
        print("-" * 100)

    return state

class Episode(BaseModel):

    should_store: bool = Field(
        description="whether the current conversation should be stored in episodic memory")

    episode: str = Field(
        description="A concise summary of the important expieriences from the current conversation that should be stored in episodic memory")

    episode_type: Literal[
        "achievement",
        "learning",
        "preference",
        "goal",
        "project",
        "other"]

episode_llm = llm.with_structured_output(Episode)

EPISODE_PROMPT = """
You are an Episodic Memory Extractor.

Your job is to determine whether the USER explicitly shared
long-term information that should be remembered.

IMPORTANT RULES

• Only extract facts explicitly stated by the user.
• Never infer.
• Never assume.
• Never guess.
• Never conclude that the user learned something unless they explicitly say so.
• Never conclude that the user completed something unless they explicitly say so.
• Never create memories from the assistant's answer.

The Assistant Answer is context only.
Do NOT extract facts from it.

Store information such as:

✓ Career
✓ Long-term goals
✓ Achievements
✓ Ongoing projects
✓ Stable preferences
✓ Skills explicitly claimed by the user

Do NOT store:

✗ Greetings
✗ Temporary factual questions
✗ One-off requests
✗ Information stated only by the assistant

Example

User:
I completed Docker certification.

Decision:
Store this conversation.

Summary:
User completed Docker certification.

----------------------------------------

Example

User:
What is Docker?

Decision:
Do not store this conversation.

----------------------------------------

Example

User:
I am an AWS developer.

Decision:
Store this conversation.

Summary:
User is an AWS developer.

----------------------------------------

User Query

{query}

----------------------------------------

Assistant Answer (Context Only)

{answer}

Analyze the conversation according to the rules above and produce the structured output.
"""

def memory_candidate(state: GraphState):

    prompt = EPISODE_PROMPT.format(
        query = state["user_query"],
        answer = state.get("final_answer", "")
    )

    response  = episode_llm.invoke(prompt)

    if response.should_store:

        state["episode"] = response.episode
        state["episode_type"] = response.episode_type

    else:
        state["episode"] = ""
        state["episode_type"] = ""

    print("=" * 100)
    print("EPISODIC MEMORY")

    print("Store :", response.should_store)

    print("Episode :", response.episode)

    print("Type :", response.episode_type)

    return state
  
QUERY_REWRITE_PROMPT = """
    You are an expert conversational query rewriting assistant.

    Your task is to rewrite the user's latest message into
    one complete standalone retrieval query.

    You have access to conversation memory.

    IMPORTANT RULES

    1. Resolve references using conversation memory.

    2. Preserve user-specific information when it changes
    what information should be retrieved.

    Examples of useful user-specific information:

    - career
    - skills
    - experience
    - certifications
    - goals
    - projects
    - stable preferences

    3. Remove filler words only when removing them does
    not change retrieval intent.

    4. Do NOT remove relevant background information.

    5. Never invent information that is not present
    in the conversation memory or current message.

    6. Never answer the question.

    7. Never decompose the question.

    8. Return exactly ONE standalone retrieval query.

    --------------------------------------------------

    Conversation Memory

    {memory}

    --------------------------------------------------

    Current User Message

    {question}

    --------------------------------------------------

    Standalone Retrieval Query:
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
            state['rewritten_query'],
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

    if not state["reranked_evidence"]:

        print("No compressed evidence found.")

        state["compressed_evidence"] = []

        return state

    for evidence in state["reranked_evidence"]:

        doc = evidence["document"]

        prompt = compression_prompt.format(
            question=state["rewritten_query"],
            document=doc.page_content
        )

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

    return state

def context_builder(state: GraphState):

    context = ""

    context += "===== Retrieved Knowledge =====\n\n"

    for i, evidence in enumerate(
        state["compressed_evidence"],
        start=1
    ):

        context += f"""

        Evidence {i}:

        Source:
        {evidence['source']}

        Page:
        {evidence['page']}

        Content:
        {evidence['compressed_text']}

        ----------------------------------------
        """

    state["context"] = context

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

    seen = set()

    # ======================================================
    # Working Memory
    # ======================================================

    working_docs = state.get("working_memory", [])

    if working_docs:

        memory_text.append(
            "================ WORKING MEMORY ================\n"
        )

        for i, doc in enumerate(working_docs, start=1):

            content = doc.page_content.strip()

            if content in seen:
                continue

            seen.add(content)

            memory_text.append(
                f"""
                    Memory {i}

                    {content}

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

            content = doc.page_content.strip()

            if content in seen:
                continue

            seen.add(content)

            memory_text.append(
                f"""
            Episode {i}

            {content}

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

            content = doc.page_content.strip()

            if content in seen:
                continue

            seen.add(content)

            memory_text.append(
                f"""
            Knowledge {i}

            {content}

            ----------------------------------------
            """
                        )

    state["hybrid_memory"] = "\n".join(memory_text)

    print("=" * 100)
    print("HYBRID MEMORY")
    print("=" * 100)

    print(state["hybrid_memory"])

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

    If retrieved knowledge is insufficient,

    DO NOT provide general advice.

    DO NOT speculate.

    Respond only:

    "I don't have enough information."

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

    print(state.get("final_answer", ""))

    return state

workflow = StateGraph(GraphState)

workflow = StateGraph(GraphState)

# ------------------------------------------------------------------
# Nodes
# ------------------------------------------------------------------

workflow.add_node("Intent_Classifier", Intent_Classifier)

# Knowledge Pipeline
workflow.add_node("memory_retrieval", memory_retrieval)
workflow.add_node("episodic_mem_ret", episodic_mem_ret)
workflow.add_node("memory_fusion", memory_fusion)
workflow.add_node("query_rewriter", query_rewriter)
workflow.add_node("retrieval_fewshot_examples", retrieval_fewshot_examples)
workflow.add_node("decompose_query", decompose_query)
workflow.add_node("Route_Query", Route_Query)
workflow.add_node("rrf_fusion", rrf_fusion)
workflow.add_node("aggregate_evidence", aggregate_evidence)
workflow.add_node("reranked_document", reranked_document)
workflow.add_node("context_compression", context_compression)
workflow.add_node("context_builder", context_builder)
workflow.add_node("show_retrieval_results", show_retrieval_results)
workflow.add_node("prompt_builder", prompt_builder)
workflow.add_node("generate_answer", generate_answer)

# Working Memory
workflow.add_node("update_memory", update_memory)

# Long-term Memory
workflow.add_node("memory_candidate", memory_candidate)
workflow.add_node("importance_scorer", importance_scorer)
workflow.add_node("episodic_writer", episodic_writer)

# Utility
workflow.add_node("no_action", no_action)
workflow.add_node("show_final_answer", show_final_answer)


# ------------------------------------------------------------------
# START
# ------------------------------------------------------------------

workflow.add_edge(START, "Intent_Classifier")


# ------------------------------------------------------------------
# Intent Router
# ------------------------------------------------------------------

workflow.add_conditional_edges(
    "Intent_Classifier",
    route_int_query,
    {
        "knowledge_pipeline": "memory_retrieval",
        "memory_pipeline": "memory_candidate",
        "no_action": "show_final_answer",
    },
)


# ------------------------------------------------------------------
# Knowledge Pipeline
# ------------------------------------------------------------------

workflow.add_edge("memory_retrieval", "episodic_mem_ret")

workflow.add_edge("episodic_mem_ret", "memory_fusion")

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


# ------------------------------------------------------------------
# After Working Memory
# Decide whether Episodic Memory should run
# ------------------------------------------------------------------

workflow.add_conditional_edges(
    "update_memory",
    route_memory_update,
    {
        "update_memory": "memory_candidate",
        "finish": "show_final_answer",
    },
)


# ------------------------------------------------------------------
# Memory-only Pipeline
# ------------------------------------------------------------------

workflow.add_edge("memory_candidate", "importance_scorer")

workflow.add_edge("importance_scorer", "episodic_writer")

workflow.add_edge("episodic_writer", "show_final_answer")


# ------------------------------------------------------------------
# END
# ------------------------------------------------------------------

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


