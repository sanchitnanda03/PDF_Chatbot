# 📄 PDF Question Answering Chatbot (RAG-based)

A Streamlit-powered chatbot that allows users to upload PDF documents and ask questions in natural language. It uses **Retrieval-Augmented Generation (RAG)** with **FAISS vector search** and **LLMs like LLaMA3** to return accurate, contextual answers based on document content.

---

## ✨ Features

- 📥 Upload any PDF document
- 🔎 Extracts and splits content into semantically searchable chunks
- 🧠 Uses Hugging Face LLMs (e.g., LLaMA3) for answer generation
- ⚡ Fast, accurate, and grounded answers
- 🖥️ Clean, user-friendly interface with Streamlit

---

## 🧠 How It Works

1. **PDF Upload:** User uploads a `.pdf` file
2. **Text Extraction:** Text is extracted and split into meaningful chunks
3. **Embedding:** Each chunk is embedded into a vector using `sentence-transformers`
4. **Vector Indexing:** FAISS stores embeddings for efficient similarity search
5. **User Question:** Converted to a vector and matched with most relevant chunks
6. **LLM Answering:** Hugging Face-hosted LLM generates the answer using those chunks
7. **Display:** Answer is shown in Streamlit UI

---

## 🧭 Project Flowchart

![PDF QnA Flowchart](https://github.com/sanchitnanda03/PDF_Chatbot/blob/main/assets/flowchart.png)

---

## 🛠️ Installation & Usage

```bash
# 1. Clone this repository
git clone https://github.com/yourusername/pdf-qna-chatbot.git
cd pdf-qna-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
