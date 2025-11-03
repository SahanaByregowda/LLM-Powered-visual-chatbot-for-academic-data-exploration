import os
import glob
# import streamlit as st # Removed direct streamlit import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document # Used for Document type hinting

# Import configurations from config.py
from config import MD_CE, HPC_FOLDER, EMBED_MODEL, PERSIST_CE, PERSIST_HPC

# Initialize Text Splitter and Embeddings globally for document processing
# These are used by both CE and HPC retrievers
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# @st.cache_resource is handled in main_app.py, so we can't use it here directly
# The functions will be wrapped by st.cache_resource in main_app.py

def ce_retriever_builder():
    """
    Loads the CE Markdown document, splits it into chunks,
    creates embeddings, and builds a Chroma vector store.
    Returns a retriever for the CE curriculum.
    """
    if not os.path.exists(MD_CE):
        # st.error(f"⚠️ CE Markdown file not found at: {MD_CE}") # Removed st.error
        return None, f"⚠️ CE Markdown file not found at: {MD_CE}" # Return error message
    try:
        with open(MD_CE, "r", encoding="utf-8") as f:
            txt = f.read()
        chunks = splitter.split_documents([Document(page_content=txt)])
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_CE)
        # st.sidebar.success(f"CE document loaded and indexed from '{MD_CE}'.") # Removed st.sidebar.success
        return vectordb.as_retriever(search_kwargs={"k": 8}), None
    except Exception as e:
        # st.error(f"Error building CE vector store: {e}") # Removed st.error
        return None, f"Error building CE vector store: {e}"


def hpc_retriever_builder():
    """
    Loads Markdown documents from the HPC folder, splits them into chunks,
    creates embeddings, and builds a Chroma vector store.
    Returns a retriever for the HPC curriculum.
    """
    docs = []
    if not os.path.exists(HPC_FOLDER):
        # st.error(f"⚠️ HPC folder not found at: {HPC_FOLDER}") # Removed st.error
        return None, f"⚠️ HPC folder not found at: {HPC_FOLDER}"
    try:
        for path in glob.glob(os.path.join(HPC_FOLDER, "*.md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read()))
        
        if not docs:
            # st.warning(f"No .md files found in HPC folder: {HPC_FOLDER}. Returning a fallback retriever.") # Removed st.warning
            # Fallback to avoid error if no docs are found
            return Chroma.from_documents([Document(page_content="No HPC documents found.")], embeddings).as_retriever(), "No .md files found in HPC folder."
        
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_HPC)
        # st.sidebar.success(f"HPC documents loaded and indexed from '{HPC_FOLDER}'.") # Removed st.sidebar.success
        return vectordb.as_retriever(search_kwargs={"k": 8}), None
    except Exception as e:
        # st.error(f"Error building HPC vector store: {e}") # Removed st.error
        return None, f"Error building HPC vector store: {e}"

# The actual initialization of retrievers will happen in main_app.py now,
# wrapped with st.cache_resource.
# retriever_ce = ce_retriever() # Removed top-level call
# retriever_hpc = hpc_retriever() # Removed top-level call
