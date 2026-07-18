from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from pathlib import Path

# path = "docker.pdf"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)

pdf_folder = Path("all_pdf")
print(pdf_folder)

def create_collection(pdf_folder, persist_dir: str, embedding):
    documents = []
    try:
        for pdf in pdf_folder.glob("*.pdf"):
            print(f"name of pdf : {pdf.name}")
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()

            for doc in docs:
                doc.metadata["book"] = pdf.stem

            documents.extend(docs)

            print(len(documents))
            print("document load successfully")

        splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap=150
                )

        chunks = splitter.split_documents(documents)       
        print("document chunks created: ",len(chunks))

        if not os.path.exists(persist_dir):
            vectorstore = Chroma.from_documents(
                documents = chunks,
                persist_directory=persist_dir,
                embedding=embedding
            )
        else:
            print("already vector db created for this pdf")

        return vectorstore
        
    except Exception as e:
        print(f"load documents Error : {e}")

ai_agent_vs = create_collection(Path("ai_agent"), "./db/ai_agent", embeddings)

docker_vs = create_collection(Path("aws"), "./db/aws", embeddings)

aws_vs = create_collection(Path("docker"), "./db/docker", embeddings)

langgraph_vs = create_collection(Path("langgraph"), "./db/langgraph", embeddings)