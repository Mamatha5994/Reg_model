import streamlit as st
import os
import time
import tempfile

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Set page configuration
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")

# App title and logo
st.image("PragyanAI_Transparent.png")  # Make sure the filename is correct and image exists
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Load Groq API key from secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize session state
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - PDF Upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    # Use a temporary file to store PDF contents
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file.getbuffer())
                        loader = PyPDFLoader(tmp_file.name)
                        docs.extend(loader.load())

                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)

                # Create embeddings and FAISS vector store
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector = FAISS.from_documents(final_documents, embeddings)

                st.success("Documents processed successfully!")
        else:
            st.warning("Please upload at least one document.")

# Main Chat Interface
st.header("Chat with your Documents")

# Load LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant. Answer the question based only on the following context.
    Be precise and concise.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.vector is not None:
        with st.chat_message("user"):
            st.markdown(prompt_input)
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt_input})
            response_time = time.process_time() - start

            answer = response.get("answer", "Sorry, I couldn't find an answer.")

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.info(f"Response time: {response_time:.2f} seconds")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please process your documents before asking questions.")
