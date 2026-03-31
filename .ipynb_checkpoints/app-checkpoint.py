import streamlit as st
import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup the Streamlit Page Configuration
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="centered")

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Welcome to the RAG Document Assistant.")
    
    # Input box for API key (masks the characters for security)
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    st.markdown("### Instructions:")
    st.write("1. Enter API Key above.")
    st.write("2. Upload a PDF file.")
    st.write("3. Ask questions about your document.")

# 3. Main App Header
st.title("RAG Document Q&A Assistant")
st.markdown("**Built by Paresh Kapoor | Python · LangChain · Groq API · FAISS**")
st.divider()

# 4. Define the Caching Function for PDF Processing
@st.cache_resource
def process_pdf(file):
    # Extract Text
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    extracted_text = "".join([page.extract_text() for page in pdf_reader.pages])
    
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(extracted_text)
    num_chunks = len(chunks)
    
    # Embeddings and FAISS Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store, num_pages, num_chunks

# 5. File Uploader UI
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

# 6. Main Logic Flow (Triggers only when a file is uploaded)
if uploaded_file is not None:
    try:
        # Show a loading spinner while processing the PDF
        with st.spinner("Processing document and building vector database..."):
            vector_store, num_pages, num_chunks = process_pdf(uploaded_file)
        
        # Display Success Message
        st.success(f"✅ Document processed: {num_pages} pages, {num_chunks} chunks created.")
        
        # 7. Q&A UI
        question = st.text_input("Ask anything about your document:")
        
        if st.button("Submit"):
            # Error Handling: Check for API Key and Question
            if not os.environ.get("GROQ_API_KEY"):
                st.error("⚠️ Please enter your Groq API Key in the sidebar first!")
            elif not question.strip():
                st.warning("⚠️ Please type a question before submitting.")
            else:
                # Show spinner while the AI thinks
                with st.spinner("Analyzing document and writing answer..."):
                    
                    # Initialize LLM and Retriever
                    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Increased from 3 to 5 for better summary context
                    
                    # Define LCEL Prompt
                    template = """Answer the question based ONLY on the following context. 
                    If you don't know the answer based on the context, just say "I don't know". Do not make up facts.
                    
                    Context:
                    {context}
                    
                    Question: {question}
                    
                    Answer:"""
                    prompt = PromptTemplate.from_template(template)
                    
                    # Helper function to format docs
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    # Build LCEL Chain
                    qa_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Generate Answer and Retrieve Sources
                    answer = qa_chain.invoke(question)
                    source_docs = retriever.invoke(question)
                    
                    # Display the final answer in a green box
                    st.success(answer)
                    
                    # Display the sources in an expandable section
                    with st.expander("View Source Sections Used"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Source Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()

    # Generic Error Handling to prevent app crashes
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")