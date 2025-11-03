import os

# --- General Configuration ---
# OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY")
# MINI-RAG CSV START
# Path for storing CSV Mini-RAG vector store
CSV_VECTOR_PERSIST_DIR = "./chroma_csv"

# Embedding model for CSV Mini-RAG
CSV_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of columns to retrieve for context
CSV_RETRIEVAL_K = 3
# MINI-RAG CSV END


# Embedding Model (used by document_handlers and potentially ce_chart_handlers for RAG)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Document Paths (for Flowchart and CE Chart Generation) ---
MD_CE = "/Users/sahanabyregowda/Desktop/chatbot/ce_curriculum_full.md" # For CE document RAG (Mermaid diagrams)
HPC_FOLDER = "/Users/sahanabyregowda/Desktop/chatbot/scraped_hpc_data" # For HPC document RAG (Mermaid diagrams)
# IMPORTANT: This JSON_CE path is used by ce_chart_handlers.py for loading curriculum data for charts
JSON_CE = "/Users/sahanabyregowda/Desktop/chatbot/parsed_ce_curriculum_with_departments.json"

# ChromaDB Persistence Directories (for document_handlers.py)
PERSIST_CE = "./chroma_ce_flow"
PERSIST_HPC = "./chroma_hpc_flow"

# --- CSV Data Path (for Chart Generation) ---
CSV_FILE_NAME = "/Users/sahanabyregowda/Desktop/chatbot/output_file.csv"

# --- LLM Chart Generation Settings ---
MAX_RETRIES_CHART_GEN = 2 # Max attempts to fix chart code


