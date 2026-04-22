import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Read PDF files and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create vector store using FAISS
def get_vectore_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",     
    google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Set up QA chain with prompt
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the context, just say "answer is not available in the context". 
        Don't make up any answers.

        Context: {context}
        Question: {question}

        Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user questions
# Handle user questions
def user_input(user_question):
    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("Please upload PDF files and click 'Submit & Process' first.")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Multiple PDFs using Gemini")

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
        st.stop()

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click 'Submit & Process'",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectore_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()