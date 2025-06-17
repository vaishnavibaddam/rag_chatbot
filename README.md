# rag_chatbot
# 📘 PDF Question Answering App (with Hugging Face & Streamlit)

This app allows users to upload a PDF and ask questions about its content. It uses Hugging Face models for embeddings and question answering, and FAISS for fast document retrieval. Built with Python and Streamlit.

---

## 🚀 Features
- Upload a PDF
- Extract and chunk text
- Embed using `MiniLM`
- Search with FAISS
- Answer questions using `Flan-T5`
- Clean, simple Streamlit interface

---

## 🛠 Tech Stack
- **Frontend**: Streamlit  
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`  
- **LLM**: `google/flan-t5-base`  
- **Vector Store**: FAISS  
- **PDF Parsing**: PyPDF2  

---

## 📥 Installation

```bash
git clone https://github.com/your-username/pdf-qa-app.git
cd pdf-qa-app
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
