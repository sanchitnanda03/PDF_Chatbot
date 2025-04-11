import streamlit as st
import os
from pdf_utils import extract_text_from_pdf, chunk_text
from rag_pipeline import create_faiss_index, get_top_k_chunks, generate_response_with_llama3

st.set_page_config(page_title="PDF QA Chatbot", layout="wide")
st.title("📄 PDF Question Answering Chatbot (LLaMA3-powered)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF..."):
        full_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(full_text)
        index, chunk_list, vectors = create_faiss_index(chunks)

    st.success("PDF indexed. Ask your questions!")

    hf_token = st.text_input("Enter your Hugging Face API token:", type="password")
    user_query = st.text_input("Ask a question based on the PDF:")
    if user_query and hf_token:
        with st.spinner("Generating answer..."):
            top_chunks = get_top_k_chunks(user_query, index, chunk_list, vectors)
            context = "\n".join(top_chunks)
            response = generate_response_with_llama3(context, user_query, hf_token)
            st.markdown("### 💬 Answer:")
            st.write(response)
