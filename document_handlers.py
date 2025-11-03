import os
import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document 


from config import MD_CE, HPC_FOLDER, EMBED_MODEL, PERSIST_CE, PERSIST_HPC


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def ce_retriever_builder():
    """
    Loads the CE Markdown document, splits it into chunks,
    creates embeddings, and builds a Chroma vector store.
    Returns a retriever for the CE curriculum.
    """
    if not os.path.exists(MD_CE):
       
        return None, f" CE Markdown file not found at: {MD_CE}" 
    try:
        with open(MD_CE, "r", encoding="utf-8") as f:
            txt = f.read()
        chunks = splitter.split_documents([Document(page_content=txt)])
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_CE)
        
        return vectordb.as_retriever(search_kwargs={"k": 8}), None
    except Exception as e:
        
        return None, f"Error building CE vector store: {e}"


def hpc_retriever_builder():
    """
    Loads Markdown documents from the HPC folder, splits them into chunks,
    creates embeddings, and builds a Chroma vector store.
    Returns a retriever for the HPC curriculum.
    """
    docs = []
    if not os.path.exists(HPC_FOLDER):
        return None, f" HPC folder not found at: {HPC_FOLDER}"
    try:
        for path in glob.glob(os.path.join(HPC_FOLDER, "*.md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read()))
        
        if not docs:
            
            return Chroma.from_documents([Document(page_content="No HPC documents found.")], embeddings).as_retriever(), "No .md files found in HPC folder."
        
        chunks = splitter.split_documents(docs)
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_HPC)
        return vectordb.as_retriever(search_kwargs={"k": 8}), None
    except Exception as e:
       
        return None, f"Error building HPC vector store: {e}"


