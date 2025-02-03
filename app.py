import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


def load_and_split_pdfs(pdf_files, chunk_size=1000, chunk_overlap=100):
    """
    Loads text from a list of PDF files,
    splits it into smaller chunks,
    and returns a list of Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        # Each page is a Document
        pdf_pages = loader.load_and_split()
        # Split each page into even smaller chunks
        for page in pdf_pages:
            for chunk in text_splitter.split_documents([page]):
                docs.append(chunk)
    
    return docs


def build_vector_store(documents, persist_directory="chroma_db"):
    """
    Converts the documents into embeddings using OpenAIEmbeddings
    and stores them in a Chroma vector store.
    
    If you'd like to persist the database (so you don't need to rebuild
    each time), keep 'persist_directory' set to a folder name (e.g. 'chroma_db').
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create Chroma from documents
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name="math_textbooks",
        persist_directory=persist_directory
    )
    
    # Optional: if you want to persist the data for future sessions
    vector_store.persist()

    return vector_store


def create_retrievalqa_chain(vector_store):
    """
    Creates a RetrievalQA chain that uses the vector store retriever and
    an OpenAI LLM to answer queries with retrieved context.
    """
    llm = OpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0  # more deterministic answers
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


@st.cache_resource
def get_vector_store():
    """
    Loads & splits PDF documents, then builds (or loads) the Chroma vector store.
    We cache this so it doesn't re-run on every interaction.
    """
    pdf_files = [
        "data/math_textbook1.pdf",
        "data/math_textbook2.pdf"
        # Add more textbooks as needed
    ]
    docs = load_and_split_pdfs(pdf_files)

    # Build (or load) Chroma vector store
    # If the folder 'chroma_db' exists with embeddings, Chroma will load it automatically.
    vector_store = Chroma(
        collection_name="math_textbooks",
        embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_directory="chroma_db"
    )
    
    # Check if we need to populate the vector store.
    # (If it's empty, we build from scratch. Otherwise, we assume it’s already stored.)
    if not vector_store._client.list_collections():
        vector_store = build_vector_store(docs, persist_directory="chroma_db")

    return vector_store


def main():
    st.title("RAG Chatbot for Math Textbooks (ChromaDB Edition)")
    st.markdown("""
    **Ask your mathematics questions** and this chatbot will retrieve the most relevant
    info from your textbooks using Chroma for vector storage, then answer with GPT.
    """)

    # Get or build vector store
    vector_store = get_vector_store()
    qa_chain = create_retrievalqa_chain(vector_store)

    # Maintain chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User input
    with st.form(key="user_input_form", clear_on_submit=True):
        user_query = st.text_input("Your question:")
        submit_button = st.form_submit_button("Ask")

    if submit_button and user_query.strip():
        with st.spinner("Retrieving & generating answer..."):
            response = qa_chain({"query": user_query})
            answer = response["result"]
            sources = response.get("source_documents", [])

        # Save to chat history
        st.session_state["chat_history"].append((user_query, answer))

        # Display answer
        st.markdown(f"**Answer:** {answer}")

        # Optionally display sources
        if sources:
            with st.expander("Sources"):
                for idx, doc in enumerate(sources, start=1):
                    st.markdown(f"**Source {idx}**: {doc.metadata.get('source')}")
                    snippet = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    st.write(snippet)

    # Show full chat history
    if st.session_state["chat_history"]:
        st.write("---")
        st.write("### Chat History")
        for i, (question, ans) in enumerate(st.session_state["chat_history"], start=1):
            st.write(f"**Q{i}:** {question}")
            st.write(f"**A{i}:** {ans}")


if __name__ == "__main__":
    main()
