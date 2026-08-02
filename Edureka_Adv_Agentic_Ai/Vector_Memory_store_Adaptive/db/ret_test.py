from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

few_shot = Chroma(
    persist_directory="db/few_shot",
    embedding_function=embeddings
)

retriever = few_shot.as_retriever(
    search_kwargs={"k": 3}
)

query = "How can I deploy an AI Agent using Docker on AWS?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print("=" * 80)
    print(f"Example {i}")
    print("Question:", doc.page_content)
    print("Metadata:", doc.metadata)