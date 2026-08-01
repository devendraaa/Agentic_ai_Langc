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

with open("data_injection.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

documents = []

for example in examples:

    page_content = f""" 
                    Question: {example['question']}
                    sub queries
                        """

    for i, query in enumerate(example['sub_queries'], start=1):
        page_content += f"\n{i}. {query}"

    documents.append(
        Document(
            page_content=page_content.strip(),

            metadata={
                "id": example["id"],
                "category":example["metadata"]["category"],
                "complexity":example["metadata"]["complexity"],
                "domains": ",".join(example["metadata"]["domains"]),
            }
        )
    )