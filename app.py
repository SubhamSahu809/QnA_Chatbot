from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
def document_process(path):
    ## document loading
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(len(docs))
    print(docs)

    ## splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)
    print(len(docs))
    print(docs)


    ##embedding and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True



### document upload
st.subheader("Document Q&A Chatbot - Ask Anything")
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select Your PDF File", type="pdf")
    if file:
        with open("uploaded_document.pdf","wb") as f:
            f.write(file.getvalue())
        with st.spinner("Processing..."):
            document_process("./uploaded_document.pdf")
            
            
        st.markdown("Document Uploaded Successfully.")
        sleep(1)
        st.rerun()

### chat ui
if st.session_state.document_uploaded and st.session_state.vector_db:
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]
        
        st.chat_message(role).markdown(content)
        
    query = st.chat_input("Ask Anything...")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})
        st.chat_message("users").markdown(query)
        
        try:
            documents = st.session_state.vector_db.similarity_search(
                query=query,
                k=2
            )
        except Exception as e:
            st.error(f"Error retrieving documents: {e}")
            st.stop()
            
        context = ""        
        for doc in documents:
            context = context + doc.page_content + "\n\n"
            
        print(context)
        prompt = f"""You are a helpful assistant and you provide answer for user questions based on the provided context.
            Context: {context}
            and
            Question: {query}
        """
            
        result = llm.invoke(prompt)
            
        st.session_state.messages.append({"role":"ai", "content": result.content})
        st.chat_message("ai").markdown(result.content)
        print(result.content)
