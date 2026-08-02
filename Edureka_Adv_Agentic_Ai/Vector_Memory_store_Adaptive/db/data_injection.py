from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from pathlib import Path
import ftfy
import json
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2" 
)

with open("few_shot_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

documents = []

for example in examples:

    documents.append(
        Document(
            page_content = example["question"],

            metadata={
                "id": example["id"],

                #store decomposition in metedata
                "sub_query": example["sub_queries"],
                "category": example["metadata"]["category"],
                "complexity": example["metadata"]["complexity"],
                "domains": example["metadata"]["domains"],
            }
        )
    )

    # -----------------------
# Create Chroma
# -----------------------

vectorstore = Chroma.from_documents(

    documents=documents,

    embedding=embeddings,

    persist_directory="db/few_shot"

)

print(f"Stored {len(documents)} examples.")