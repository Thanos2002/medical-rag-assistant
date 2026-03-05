import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import dotenv_values

config = dotenv_values(".env")  # add this near the top of the file


load_dotenv()

CHROMA_PATH = "chroma_db"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def build_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""You are a helpful medical research assistant.
Use the following context from research papers to answer the question.
If you don't know the answer based on the context, say "I don't have enough information in the provided documents."

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        api_key=config["OPENAI_API_KEY"], 
        base_url="https://openrouter.ai/api/v1",
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

if __name__ == "__main__":
    print("[INFO] Building RAG chain...")
    chain, retriever = build_rag_chain()

    query = "Based on the Home EEG data provided in Figure 1, which specific sleep metric demonstrated a statistically significant reduction in Alzheimer’s Disease (AD) patients compared to controls, and why would relying on the Pittsburgh Sleep Quality Index (PSQI) or actigraphy (WASO) be insufficient for predicting this specific physiological change?"
    print(f"\n[INFO] Query: {query}")

    answer = chain.invoke(query)
    print(f"\n[INFO] Answer:\n{answer}")

    # Show sources separately
    print("\n[INFO] Sources:")
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] Page {doc.metadata.get('page', '?')} — {doc.metadata.get('source', '?')}")