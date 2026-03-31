# **📄 RAG Document Q\&A Assistant**

**An intelligent Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents.**

**🌟 Try the Live App Here:** [rag-portfolio-project-by-paresh-kapoor.streamlit.app](https://rag-portfolio-project-by-paresh-kapoor.streamlit.app/)

## **🧠 How It Works (The RAG Pipeline)**

When you upload a document and ask a question, the system does not just send the whole PDF to the AI. Instead, it uses a highly efficient RAG architecture:

1. **PDF Upload & Parsing:** Extracts raw text from the document.  
2. **Chunking:** Splits the text into manageable overlapping chunks.  
3. **Embeddings:** Converts text chunks into mathematical vectors running entirely locally.  
4. **Vector Database:** Stores the vectors for rapid mathematical semantic search.  
5. **Retrieval & Generation:** Matches your question to the most relevant chunks, then sends only that context to the LLM to generate a hallucination-free, accurate answer.

## **🛠️ Tech Stack & Dependencies**

This project is built using modern, highly efficient open-source tools:

* **Frontend:** Streamlit (streamlit)  
* **Document Processing:** PyPDF2 (PyPDF2)  
* **RAG Framework & Orchestration:** LangChain (langchain, langchain-community, langchain-text-splitters)  
* **LLM (Text Generation):** Meta Llama 3 via Groq API (langchain-groq for ultra-fast inference)  
* **Embeddings:** HuggingFace locally (langchain-huggingface, sentence-transformers)  
* **Vector Store:** Facebook AI Similarity Search (faiss-cpu)  
* **Environment Management:** (python-dotenv)

## **🚀 How to Run Locally**

### **1\. Clone the repository**

git clone [https://github.com/Paresh-Kapoor/RAG\_Portfolio\_Project.git](https://github.com/Paresh-Kapoor/RAG\_Portfolio\_Project.git)  


### **2\. Set up a virtual environment (Optional but recommended)**

conda create \-n rag\_env python=3.10 \-y  
conda activate rag\_env

### **3\. Install dependencies**

pip install \-r requirements.txt

### **4\. API Key Setup**

This app requires a **Groq API Key**. You can get one for free at [console.groq.com](https://console.groq.com/).

You can either:

* Paste it directly into the Streamlit UI sidebar when the app runs.  
* OR create a .env file in the root directory and add: GROQ\_API\_KEY="your\_api\_key\_here"

### **5\. Run the Application**

streamlit run app.py  
