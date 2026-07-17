from typing import TypedDict, List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("api key not found in .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

path = "docker.pdf"

def document_load(path):
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        print("document load successfully")

        return documents
    except Exception as e:
        print(f"load documents Error : {e}")

def vector_st(documents):

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        print(len(chunks))
        print(chunks[2])

        if os.path.exists("./db"):
            vectorstore = Chroma(
                persist_directory="./db",
                embedding=embeddings
            )
        else:
            vectorstore = Chroma.from_documents(
                documents = chunks,
                embedding=embeddings,
                persist_directory="./db"
            )
        
        print("documents store to vectore store successfuly")

        return vectorstore
    
    except Exception as e:
        print(f"vector store Error : {e}")

text = document_load(path=path)
vectorsto = vector_st(text)
retriever = vectorsto.as_retriever(search_kwargs={"k": 4})

query = "what is docker"
ret = retriever.invoke(query)

for i in ret:
    print("="*60)
    print(i.page_content)

class GraphState(TypedDict):

    question: str

    current_query: str

    documents: List[Document]

    answer: str

    retrieval_count: int

    max_hops: int

    enough_context: bool





@tool
def retrieve_documents(query: str) -> str:
    """
    Search enterprise knowledge base.

    Use whenever information is needed.
    """

    docs = retriever.invoke(query)

    return docs


JUDGE_PROMPT = """
        You are an expert evaluator.

        Question:
        {question}

        Retrieved Context:
        {context}

        Determine whether the retrieved context contains enough information
        to completely answer the user's question.

        Respond ONLY with:

        YES

        or

        NO
"""


REWRITE_QUERY_PROMPT = """
        You are an expert search query writer.

        Original Question:
        {question}

        Current Retrieved Context:
        {context}

        The retrieved context is insufficient.

        Generate ONE improved search query that is likely to retrieve
        the missing information.

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


def retrieve_node(state: GraphState):

    print(f"\nSearching: {state['current_query']}")
    docs = retrieve_documents.invoke(
        state["current_query"]
    )

    state["documents"].extend(docs)
    state["retrieval_count"] += 1

    return state

def judge_node(state: GraphState):

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = JUDGE_PROMPT.format(

        question=state["question"],
        context=context
    )

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    state["enough_context"] = (
        decision == "YES"
    )

    return state

def rewrite_query_node(state: GraphState):

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = REWRITE_QUERY_PROMPT.format(
        question=state["question"],
        context=context
    )

    response = llm.invoke(prompt)

    state["current_query"] = response.content

    return state

def answer_node(state: GraphState):

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = ANSWER_PROMPT.format(

        question=state["question"],
        context=context
    )

    response = llm.invoke(prompt)
    state["answer"] = response.content

    return state

def route_after_judge(state: GraphState):

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

builder.add_edge(START,"retrieve")

builder.add_conditional_edges(

    "judge",

    route_after_judge,

    {
        "answer": "answer",
        "rewrite": "rewrite",
    }

)

builder.add_edge("rewrite","retrieve")
builder.add_edge("answer",END)

# graph = builder.compile()


# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass