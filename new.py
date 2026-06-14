import streamlit as st
import os
from time import sleep
from dotenv import load_dotenv
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore

load_dotenv()
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Setup Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# Streamlit session state initialization
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

def document_process(path):
    # 1. Open the PDF directly using PyMuPDF
    doc_pdf = fitz.open(
        stream=file.getvalue(),
        filetype="pdf"
    )
    extracted_docs = []
    
    # 2. Extract plain text directly from each page instead of generating images
    for i, page in enumerate(doc_pdf):
        with st.spinner(f"Extracting text from page {i+1}..."):
            # .get_text("text") extracts structural blocks cleanly from text-based PDFs
            page_text = page.get_text("text")
            
        if page_text.strip():  # Only append pages that contain actual text content
            extracted_docs.append(Document(page_content=page_text, metadata={"page": i+1}))

    doc_pdf.close()

    if not extracted_docs:
        st.error("No extractable text found. Make sure this isn't a scanned image PDF.")
        st.stop()

    ## 3. Split the extracted native text into manageable context blocks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    docs = splitter.split_documents(extracted_docs)
    print(f"Generated {len(docs)} document chunks.")
    print("done")

    ## 4. Embedding and Vector Indexing
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

### Main Application Layout
st.set_page_config(page_title="Document QA Bot", layout="centered")
st.subheader("Document RAG Chatbot - Native Text Extraction")

# Initialize Cloudinary URL
if "cloudinary_url" not in st.session_state:
    st.session_state.cloudinary_url = None

if not st.session_state.document_uploaded:

    file = st.file_uploader(
        "Select Your PDF File",
        type=["pdf"]
    )

    if file:

        with st.spinner("Uploading and Processing PDF..."):

            temp_path = "uploaded_document.pdf"

            # Save locally
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                temp_path,
                resource_type="raw",
                folder="pdf_uploads"
            )

            # Save URL
            st.session_state.cloudinary_url = upload_result["secure_url"]

            # Process PDF
            document_process(temp_path)

            # Delete local file (optional)
            if os.path.exists(temp_path):
                os.remove(temp_path)

        st.success("Document uploaded and indexed successfully!")

        sleep(1)

        st.rerun()

### Chat UI Interface Execution
if st.sidebar.button(
    "Reset Session & Upload New File",
    use_container_width=True
):

    st.session_state.document_uploaded = False
    st.session_state.vector_db = None
    st.session_state.messages = []
    st.session_state.cloudinary_url = None

    if os.path.exists("uploaded_document.pdf"):
        os.remove("uploaded_document.pdf")

    st.rerun()

    # Render previous interactions
    for oneMessage in st.session_state.messages:
        st.chat_message(oneMessage["role"]).markdown(oneMessage["content"])
        
    query = st.chat_input("Ask Anything about the document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        try:
            # Retrieve the 4 closest matching chunks
            documents = st.session_state.vector_db.similarity_search(query=query, k=4)
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            st.stop()
            
        context = ""        
        for doc in documents:
            context += f"[Page {doc.metadata.get('page', 'Unknown')} Chunks]:\n" + doc.page_content + "\n\n"
            
        prompt = f"""You are a precise document validation assistant. Provide an accurate, contextual answer for the user's question strictly based on the provided document layout.

Context:
{context}

Question: {query}

Instructions:
- Use only facts present in the text above. 
- If information is missing or unclear based on context, state that explicitly.
"""
            
        with st.spinner("Analyzing context..."):
            result = llm.invoke(prompt)
            
        st.session_state.messages.append({"role": "ai", "content": result.content})
        st.chat_message("ai").markdown(result.content)