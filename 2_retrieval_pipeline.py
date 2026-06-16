from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import shutil
import subprocess


persist_directory = "db/chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Explain about windows?"
# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.3})

relevant_docs = retriever.invoke(query)
print(f"\nQuery: {query}")
print("------context retrieved------")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, answer the question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a concise and accurate answer based on the above documents."""

model = ChatOllama(
    model="llama2"
)

messages=[
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input),
]

result=model.invoke(messages)
print("------Answer------")
print(result.content)