import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import dotenv_values

config = dotenv_values(".env")

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def load_vectorstore(chroma_path: str = CHROMA_PATH):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    return vectorstore

def build_rag_chain(chroma_path: str = CHROMA_PATH):
    vectorstore = load_vectorstore(chroma_path)
    
    # 1. PRE-FETCH: Get a list of all unique documents and their full paths
    try:
        data = vectorstore.get(include=['metadatas', 'documents'])
        metadatas = data['metadatas']
        doc_contents = data['documents']
        
        unique_docs_info = {}
        for i, m in enumerate(metadatas):
            src_path = m.get('source', 'Unknown')
            src_name = os.path.basename(src_path)
            if src_name not in unique_docs_info:
                unique_docs_info[src_name] = {
                    "path": src_path,
                    "header": doc_contents[i][:800]
                }
        
        all_docs = sorted(list(unique_docs_info.keys()))
    except Exception:
        all_docs = []
        unique_docs_info = {}

    doc_list_str = ", ".join(all_docs) if all_docs else "No documents found."

    # 2. CUSTOM RETRIEVER: Search each document specifically
    def balanced_retriever_func(query):
        final_docs = []
        if not unique_docs_info:
            return []
            
        k_per_doc = max(2, 12 // len(unique_docs_info))
        
        for name, info in unique_docs_info.items():
            try:
                docs = vectorstore.similarity_search(
                    query, 
                    k=k_per_doc, 
                    filter={"source": info["path"]}
                )
                final_docs.extend(docs)
            except Exception as e:
                print(f"[ERROR] Retrieval failed for {name}: {e}")
        
        return final_docs

    prompt = PromptTemplate.from_template(f"""You are a specialized medical research assistant. 
The system has access to the following documents: {doc_list_str}

Below are headers and excerpts from these documents. 
Your task is to answer the question accurately, distinguishing between information from different sources.

### FORMATTING GUIDELINES:
1. **Structure:** Use clear markdown formatting. Use bolding for key terms and bullet points for lists of findings.
2. **Mathematics:** Use **LaTeX** for all mathematical equations, formulas, and statistical notation (e.g., use `$p < 0.05$` or `$$E=mc^2$$`).
3. **Citations:** Use numerical citations in the format [1], [2], etc., to refer to the specific excerpts provided in the "RELEVANT EXCERPTS" section below.
4. **Attribution:** In addition to [1], mention the document name when introducing a new study or findings.
4. **Synthesis:** Contrast findings between documents where relevant.
5. **Missing Info:** If context for one paper is insufficient for the question, be honest and state what information is missing for THAT specific paper.

--- DOCUMENT HEADERS (FIRST CHUNKS) ---
{{doc_headers}}

--- RELEVANT EXCERPTS (BALANCED PER DOCUMENT) ---
{{context}}
--- END OF CONTEXT ---

Question: {{question}}

Answer:""")

    llm = ChatGoogleGenerativeAI(
        model="gemma-3-27b-it",
        google_api_key=config["GEMINI_API_KEY"],
        temperature=0.1,
    )

    def format_headers_func(dummy=None):
        header_parts = []
        for src, info in unique_docs_info.items():
            header_parts.append(f"DOCUMENT: {src}\nSUMMARY/TITLE INFO: {info['header']}...\n")
        return "\n".join(header_parts)

    def format_docs_func(docs):
        if not docs: return "No relevant context found for the specific query."
        docs_by_source = {}
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            if source not in docs_by_source: docs_by_source[source] = []
            docs_by_source[source].append(doc)

        formatted_parts = []
        global_excerpt_count = 1
        for source, source_docs in docs_by_source.items():
            formatted_parts.append(f"\nDOCUMENT: {source}")
            for doc in source_docs:
                page = doc.metadata.get("page", "?")
                formatted_parts.append(f"[Excerpt {global_excerpt_count}, Page {page}]:\n{doc.page_content}")
                global_excerpt_count += 1
            formatted_parts.append("-" * 20)
        return "\n".join(formatted_parts)

    # Use a chain that invokes our custom balanced_retriever
    # Wrap functions with RunnableLambda for LCEL compatibility
    chain = (
        {
            "context": RunnableLambda(balanced_retriever_func) | RunnableLambda(format_docs_func), 
            "question": RunnablePassthrough(),
            "doc_headers": RunnableLambda(format_headers_func)
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, balanced_retriever_func


if __name__ == "__main__":
    print("[INFO] Building RAG chain...")
    chain, retriever = build_rag_chain()

    query = "Based on the Home EEG data provided in Figure 1, which specific sleep metric demonstrated a statistically significant reduction in Alzheimer’s Disease (AD) patients compared to controls, and why would relying on the Pittsburgh Sleep Quality Index (PSQI) or actigraphy (WASO) be insufficient for predicting this specific physiological change?"
    print(f"\n[INFO] Query: {query}")

    answer = chain.invoke(query)
    print(f"\n[INFO] Answer:\n{answer}")

    # Show sources separately
    print("\n[INFO] Sources:")
    docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever(query)
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] Page {doc.metadata.get('page', '?')} — {doc.metadata.get('source', '?')}")
