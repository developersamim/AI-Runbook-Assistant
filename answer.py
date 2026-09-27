import os
from pathlib import Path
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(override=False)

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "langchain-chunks-index"
PINECONE_NAMESPACE = "personal"

RETRIEVAL_K = 10
# ONNX-based (no torch) so the app fits in Render's 512MB; same vectors as HuggingFace all-MiniLM-L6-v2
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the personal assistant of Samim Ahmed.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""



llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
    )

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(h["content"] for h in history if h["role"] == "user")
    return prior + "\n" + question

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
    