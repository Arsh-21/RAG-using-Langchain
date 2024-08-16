import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Function to extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to initialize Chroma and embeddings
def initialize_vector_store():
    # Replace with paths to your PDFs
    pdf_files = pdf_files = [" SETS & FUNCTIONS (VOL 1).pdf", "CALCULUS (VOL 2).pdf", "GRAPH THEORY (VOL 3).pdf"]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = []

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        chunks = text_splitter.split_text(text)
        # Convert each chunk into a Document object
        documents.extend([Document(page_content=chunk, metadata={"source": pdf_file}) for chunk in chunks])

    # Initialize the embeddings and Chroma vector store
    openai_api_key = os.getenv("OPEN_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store

# Initialize the vector store
vector_store = initialize_vector_store()

# Create a RetrievalQA chain with OpenAI
qa_chain = RetrievalQA(llm=OpenAI(), retriever=vector_store.as_retriever())

def main():
    st.title("PDF-based Chatbot with LangChain")

    user_input = st.text_input("You: ", "Hello, how can I help you?")

    if user_input:
        response = qa_chain.run(user_input)
        st.text_area("Bot:", value=response, height=200)

if __name__ == "__main__":
    main()
