import os
import glob
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import re # Import regex module for more aggressive cleaning


from prompts import STYLE_PROMPT, FALLBACK



API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("⚠️  Please set OPENAI_API_KEY in your environment.")
    st.stop()


PDF_CE = "/Users/sahanabyregowda/Desktop/chatbot/ce_curriculum_full.md"
HPC_FOLDER = "/Users/sahanabyregowda/Desktop/chatbot/scraped_hpc_data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_CE  = "./chroma_ce_flow"
PERSIST_HPC = "./chroma_hpc_flow"


import json

JSON_CE = "/path/to/enhanced_curriculum_with_nlp.json"  

def _load_ce_docs() -> list[Document]:
    """Loads structured CE content from a JSON file."""
    try:
        with open(JSON_CE, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        docs = []

       
        for entry in json_data.get("curriculum", []):
            topic = entry.get("topic", "Untitled Topic")
            description = entry.get("description", "")
            prerequisites = ", ".join(entry.get("prerequisites", []))
            learning_outcomes = "\n- " + "\n- ".join(entry.get("learning_outcomes", [])) if entry.get("learning_outcomes") else ""

            
            content = f"## {topic}\n\n{description}\n\n**Prerequisites**: {prerequisites}\n\n**Learning Outcomes**:{learning_outcomes}"
            docs.append(Document(page_content=content, metadata={"source": JSON_CE}))
        
        return docs

    except Exception as e:
        st.error(f"⚠️ Error loading CE JSON file: {e}")
        st.stop()



def _load_hpc_docs() -> list[Document]:
    """Loads content from all Markdown files in the HPC folder."""
    docs = []
    try:
        for path in glob.glob(os.path.join(HPC_FOLDER, "*.md")):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(), metadata={"source": path}))
    except Exception as e:
        st.error(f"⚠️ Error loading HPC Markdown files from {HPC_FOLDER}: {e}")
        st.stop()
    return docs

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource(show_spinner=" Building CE vector store …")
def ce_retriever():
    """Builds and returns the CE vector store retriever."""
    ce_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    ce_raw_docs = _load_ce_docs()
    
    chunks = []
    for doc in ce_raw_docs:
        chunks.extend(ce_splitter.split_documents([doc]))

    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_CE)
    return vectordb.as_retriever(search_kwargs={"k": 8})

@st.cache_resource(show_spinner="🔧 Building HPC vector store …")
def hpc_retriever():
    """Builds and returns the HPC vector store retriever."""
    hpc_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=200)
    hpc_raw_docs = _load_hpc_docs()
    
    chunks = []
    for doc in hpc_raw_docs:
        chunks.extend(hpc_splitter.split_documents([doc]))

    
    print(f"\nDEBUG: Total HPC chunks generated: {len(chunks)}\n")
    
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_HPC)
    return vectordb.as_retriever(search_kwargs={"k": 15})

retriever_ce  = ce_retriever()
retriever_hpc = hpc_retriever()


HPC_KEYWORDS = {
    "hpc", "eumaster4hpc", "mobility", "summer school", "challenge", "workshop", "internship",
    "application", "apply", "admission", "procedure", "documents", "requirements",
    "thesis", "master thesis", "career", "jobs", "consortium", "governance", "news",
    "contact", "universities", "study programme", "extracurricular", "teaching materials",
    "academic journey", "curriculum", "year one", "year two", "specialisation", "program"
}

def choose_retriever(q: str):
    """Chooses the appropriate retriever based on the query."""
    q_lower = q.lower()
    
    if "eumaster4hpc" in q_lower:
        return "HPC", retriever_hpc
    
    if any(kw in q_lower for kw in HPC_KEYWORDS):
        return "HPC", retriever_hpc
    
    return "CE", retriever_ce




st.set_page_config(page_title="🎓 CE + HPC Flowchart Generator", layout="centered")
st.title(" CE + HPC Flowchart Generator")
st.markdown("Ask about the structure – I'll return a Mermaid flowchart.")

query = st.text_input(" Your chart request", "Draw the mobility semester structure in the HPC program")
show_chunks = st.checkbox("Show retrieved chunks (debug)")

if query:
    domain, retriever = choose_retriever(query)
    st.info(f"**Identified Query Domain:** `{domain}`")
    
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(d.page_content for d in docs) + "\n\n" + FALLBACK

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    prompt = STYLE_PROMPT.format(context=context_text, query=query)

    with st.spinner(" GPT-4 is generating your chart …"):
        answer = chain.run(prompt)

    st.subheader(" Generated Chart")
    if "```mermaid" in answer:
        start_idx = answer.find("```mermaid")
        end_idx = answer.find("```", start_idx + len("```mermaid"))

        if start_idx != -1 and end_idx != -1:
            mermaid_raw = answer[start_idx + len("```mermaid"):end_idx]
            
            
            mermaid = re.sub(r'[^\x00-\x7F]+', '', mermaid_raw)
            mermaid = re.sub(r'\s+', ' ', mermaid).strip()
            mermaid_lines = [line.strip() for line in mermaid.split('\n') if line.strip()]
            mermaid = '\n'.join(mermaid_lines)
            

            print("\n--- Cleaned Mermaid Code (repr for console debugging) ---")
            print(repr(mermaid))
            print("--------------------------------------------------\n")
            
        else:
            st.warning("Mermaid block delimiters not found. Displaying raw response.")
            st.code(answer)
            mermaid = None

        if mermaid:
            st.code(mermaid, language="mermaid")
            st.components.v1.html(
                f"""
                <script src="[https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js](https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js)"></script>
                <script>
                  mermaid.initialize({{ startOnLoad: true }});
                </script>
                <div class="mermaid">{mermaid}</div>
                """,
                height=600,
                scrolling=True,
            )
           
            retriever_name = domain
            st.session_state.chat_history.append({"role": "assistant", "content": f"Generated {retriever_name} flowchart for: {query}"})
            st.session_state.chat_history.append({"role": "assistant", "content": f"```mermaid\n{mermaid}\n```"})
        else:
            st.warning("GPT-4 did not return valid Mermaid code; raw response below:")
            st.code(answer)
            retriever_name = domain
            st.session_state.chat_history.append({"role": "assistant", "content": f"Failed to generate {retriever_name} flowchart for: {query}"})
    else:
        st.warning("GPT-4 did not return Mermaid code; raw response below:")
        st.code(answer)
        retriever_name = domain
        st.session_state.chat_history.append({"role": "assistant", "content": f"Failed to generate {retriever_name} flowchart for: {query}"})

    if show_chunks:
        st.write(f"**Router selected:** `{domain}` knowledge-base")
        st.write(f"**Number of chunks retrieved:** {len(docs)}")
        for i, d in enumerate(docs, 1):
            source_info = d.metadata.get('source', 'Unknown')
            st.write(f"**Chunk {i} (from: {source_info}):**", d.page_content[:400] + " …")

