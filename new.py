import os
from time import sleep

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

import cloudinary
import cloudinary.uploader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import InMemoryVectorStore

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# -------------------------------------------------
# Gemini LLM
# -------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

# -------------------------------------------------
# Session State
# -------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "cloudinary_url" not in st.session_state:
    st.session_state.cloudinary_url = None

# -------------------------------------------------
# PDF Processing
# -------------------------------------------------


def document_process(uploaded_file):

    pdf = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    )

    extracted_docs = []

    for page_num, page in enumerate(pdf):

        text = page.get_text("text")

        if text.strip():

            extracted_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                    },
                )
            )

    pdf.close()

    if not extracted_docs:

        st.error(
            "No text found in PDF. "
            "If this is a scanned PDF, OCR is required."
        )

        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(extracted_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


# -------------------------------------------------
# UI
# -------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A Chatbot",
    layout="centered",
)

st.title("📄 PDF Q&A RAG Chatbot")

# -------------------------------------------------
# Upload Section
# -------------------------------------------------

if not st.session_state.document_uploaded:

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if uploaded_file:

        with st.spinner("Uploading to Cloudinary..."):

            upload_result = cloudinary.uploader.upload(
                uploaded_file.getvalue(),
                resource_type="raw",
                folder="pdf_uploads",
            )

        st.session_state.cloudinary_url = upload_result["secure_url"]

        with st.spinner("Processing document..."):

            document_process(uploaded_file)

        st.success("Document uploaded successfully!")

        sleep(1)

        st.rerun()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

if st.sidebar.button(
    "Reset Session",
    use_container_width=True,
):

    st.session_state.vector_db = None
    st.session_state.messages = []
    st.session_state.document_uploaded = False
    st.session_state.cloudinary_url = None

    st.rerun()

# -------------------------------------------------
# Chat Interface
# -------------------------------------------------

if (
    st.session_state.document_uploaded
    and st.session_state.vector_db is not None
):

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input(
        "Ask any questions"
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        documents = st.session_state.vector_db.similarity_search(
            query=query,
            k=4,
        )

        context = ""

        for doc in documents:

            context += (
                f"Page {doc.metadata.get('page')}:\n"
                + doc.page_content
                + "\n\n"
            )

        prompt = f"""
        You are a helpful PDF assistant. Answer ONLY from the supplied context.

        If the answer is not available in the context,
        say: "I couldn't find that information in the document."

        Context:{context}

        Question:{query}

        Answer:
        """

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)

        with st.chat_message("assistant"):
            st.markdown(response.content)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )