from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from .get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant. Use only the following context to answer the question.

Context:
{context}

---

Question: {question}
"""

def query_rag(question: str) -> dict:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(question, k=5)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=question
    )

    model = Ollama(model="deepseek-r1:1.5b")
    answer = model.invoke(prompt)

    sources = [doc.metadata for doc, _ in results]
    return {"response": answer, "sources": sources}
