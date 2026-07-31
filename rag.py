from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import (
    OLLAMA_BASE_URL, EMBEDDING_MODEL, CHAT_MODEL,
    CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K
)

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about files on the user's computer.
Use ONLY the context below to answer. Mention the file path(s) where the answer was found, but list each unique file path only ONCE, even if it appears in multiple context chunks.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

def ask_question(question: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    results = retriever.invoke(question)

    if not results:
        return "I could not find anything relevant. Try indexing a folder first.", []

    context_parts = []
    sources = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)
    response = llm.invoke(prompt)

    return response.content, sources

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        answer, sources = ask_question(q)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {sources}")
