import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="Paper Assistant", layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# --- Sidebar: Upload & Settings ---
with st.sidebar:
    st.title("Settings & Upload")
    st.header("Upload paper PDFs")
    pdfs = st.file_uploader(
        "Upload reference documents", 
        accept_multiple_files=True, 
        type="pdf",
        help="Upload one or more PDFs to create a searchable context."
    )

    if pdfs:
        if st.button("Ingest Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/ingest-session",
                        files=[
                            ("files", (pdf.name, pdf.getvalue(), "application/pdf"))
                            for pdf in pdfs
                        ]
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.pdf_uploaded = True
                        st.success(f"Ingested {len(pdfs)} PDFs ({data['chunks']} chunks)")
                        st.info(f"Session ID: {st.session_state.session_id}")
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("⚠️ **Disclaimer:** This tool is for informational purposes only.")

# --- Main Chat Interface ---
st.title("Paper Assistant")

# Custom CSS for better formatting
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffa421;
        margin-bottom: 10px;
    }
    .source-header {
        font-weight: bold;
        color: #ffa421;
        margin-bottom: 5px;
    }
    .source-content {
        font-size: 0.9rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Supporting Evidence"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">[{i+1}] {source['source']} — Page {source['page']}</div>
                        <div class="source-content">"{source.get('content', 'No content available')}"</div>
                    </div>
                    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    display_message(message)

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Check if a PDF has been uploaded
    if not st.session_state.pdf_uploaded:
        st.warning("Please upload and ingest a PDF in the sidebar first to start the session.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sources..."):
                payload = {
                    "question": prompt,
                    "use_session": True,
                    "session_id": st.session_state.session_id
                }
                try:
                    response = requests.post("http://127.0.0.1:8000/query", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        # Display the new message immediately
                        st.markdown(answer)
                        with st.expander("View Supporting Evidence"):
                            for i, source in enumerate(sources):
                                st.markdown(f"""
                                <div class="source-box">
                                    <div class="source-header">[{i+1}] {source['source']} — Page {source['page']}</div>
                                    <div class="source-content">"{source.get('content', 'No content available')}"</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        })
                    else:
                        try:
                            error_detail = response.json().get('detail', 'Query failed')
                            st.error(f"Backend Error: {error_detail}")
                        except:
                            st.error(f"Backend Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
