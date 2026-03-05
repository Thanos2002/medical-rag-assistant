import streamlit as st
import requests

st.title("Paper Assistant")

st.divider()  # visual separator

# --- Section 1: Upload PDF (optional) ---
st.header("Upload a PDF (optional)")
pdf = st.file_uploader("Upload the paper you have a certain question about it and then just ask!", max_upload_size=10, accept_multiple_files=False, type="pdf")

if pdf is not None:
    if st.button("Ingest PDF"):
        response = requests.post(
            "http://127.0.0.1:8000/ingest",
            files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}
        )
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Error: {response.json()['detail']}")

st.divider()  # visual separator

st.header("Ask a Question")
user_input = st.text_area(
    "Your question:", 
    placeholder="Enter your question here...",
    help="You can drag the bottom-right corner to resize this box."
)

if user_input.strip():
    with st.spinner("Thinking..."): 
        response = requests.post(
            "http://127.0.0.1:8000/query",
            json={"question": user_input}
        )
    if response.status_code == 200:
        data = response.json()
        st.success("Done!")
        st.subheader("Answer")
        st.write(data["answer"])
        st.subheader("Sources")
        for source in data["sources"]:
            st.write(f"📄 Page {source['page']} — {source['source']}")
    else:
        st.error(f"Error: {response.json()['detail']}")

