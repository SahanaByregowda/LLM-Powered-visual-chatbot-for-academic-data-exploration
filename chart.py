import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import io
import contextlib
import re

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


st.set_page_config(
    page_title="Curriculum Chart Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Curriculum Data Chart Generator")
st.markdown("Upload your curriculum JSON, ask a question, and get a chart!")

def clean_and_parse_number(value_str):
    """Parse strings like '7,5 ECTS', '??', '90 h' into float"""
    if not isinstance(value_str, str):
        return 0.0
    value_str = value_str.strip().replace(",", ".")
    if "??" in value_str:
        return 0.0
    try:
        numeric_match = re.match(r"^\d+(\.\d+)?", value_str)
        if numeric_match:
            return float(numeric_match.group(0))
        return 0.0
    except (ValueError, IndexError):
        return 0.0

def extract_workload(workload_str):
    """Parse workload strings like 'Contact hours: 60h\nIndependent study: 90h'"""
    contact_hours = 0.0
    independent_study = 0.0

    if not isinstance(workload_str, str):
        return contact_hours, independent_study

    lines = workload_str.strip().split("\n")
    for line in lines:
        line = line.lower()
        if "contact hours" in line or "attendance time" in line:
            contact_hours += clean_and_parse_number(line)
        elif "independent study" in line or "self study" in line:
            independent_study += clean_and_parse_number(line)

    return contact_hours, independent_study


def fallback_generate_chart_data(data):
    """Manual safe fallback if LLM code fails"""
    df = pd.DataFrame(data)

  
    if "Lecturers" not in df.columns and "Module Coordinator" not in df.columns:
        st.error("No 'Lecturers' or 'Module Coordinator' field found in JSON data.")
        return None

    df['Lecturers'] = df.get('Lecturers', '').fillna('').astype(str)
    df['Module Coordinator'] = df.get('Module Coordinator', '').fillna('').astype(str)

    
    df['All Professors'] = df['Lecturers'] + ', ' + df['Module Coordinator']
    df['All Professors'] = df['All Professors'].apply(lambda x: [name.strip() for name in x.split(',') if name.strip()])
    df = df.explode('All Professors')

    professor_counts = df['All Professors'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    professor_counts.plot(kind='bar', ax=ax, color='skyblue')

    ax.set_title('Number of Modules Taught by Each Professor')
    ax.set_xlabel('Professor')
    ax.set_ylabel('Number of Modules')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


@st.cache_resource(show_spinner=" Loading curriculum data and building knowledge base...")
def setup_knowledge_base(uploaded_file):
    if uploaded_file is None:
        return None, None

    try:
        curriculum_data = json.load(uploaded_file)
        documents = [Document(page_content=json.dumps(m, ensure_ascii=False)) for m in curriculum_data]

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        return curriculum_data, retriever
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None, None


@st.cache_resource(show_spinner=" Initializing LLM...")
def initialize_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, openai_api_key=api_key)
    return llm

llm_openai = initialize_llm()


system_prompt_template = """
You are a Python data analysis and visualization expert.
Generate a `generate_chart_data(data)` function that returns a matplotlib figure.
Always handle inconsistent data formats safely.
"""

chart_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", "{question}")
])


uploaded_file = st.sidebar.file_uploader(
    "Upload your curriculum JSON file",
    type="json",
    help="Upload the 'enhanced_curriculum_with_nlp.json' file."
)

curriculum_data, retriever = setup_knowledge_base(uploaded_file)

if curriculum_data is None:
    st.info("Please upload a JSON file to begin.")
    st.stop()

user_question = st.text_area(
    "What chart would you like to generate?",
    "Bar chart of number of modules taught by each professor",
    height=100
)

if st.button("Generate Chart"):
    with st.spinner("Generating chart code..."):
        # Retrieve context
        docs = retriever.get_relevant_documents(user_question)
        context_for_llm = "\n\n".join(doc.page_content for doc in docs)

        # Get LLM response
        llm_response = llm_openai.invoke(chart_generation_prompt.format_messages(
            question=user_question,
            context=context_for_llm
        ))
        python_code_raw = llm_response.content

    st.subheader("Generated Python Code (for review)")
    st.code(python_code_raw, language="python")

    # Extract code block
    match = re.search(r"```python\s+(.*?)```", python_code_raw, re.DOTALL)
    python_code = match.group(1).strip() if match else None

    # Execution environment
    exec_globals = {
        "json": json,
        "defaultdict": defaultdict,
        "re": re,
        "pd": pd,
        "plt": plt,
        "clean_and_parse_number": clean_and_parse_number,
        "extract_workload": extract_workload,
        "data": curriculum_data
    }

    try:
        if python_code:
            compiled_code = compile(python_code, "<generated_chart_script>", "exec")
            exec(compiled_code, exec_globals)

            if 'generate_chart_data' in exec_globals:
                fig = exec_globals['generate_chart_data'](curriculum_data)
            else:
                st.warning("LLM code invalid — using fallback chart.")
                fig = fallback_generate_chart_data(curriculum_data)
        else:
            st.warning("No valid code from LLM — using fallback chart.")
            fig = fallback_generate_chart_data(curriculum_data)

        if fig:
            st.subheader("Generated Chart")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.error("Chart generation failed.")
    except Exception as e:
        st.error(f"Error executing generated code: {e}")
        st.subheader("Fallback Chart")
        fig = fallback_generate_chart_data(curriculum_data)
        if fig:
            st.pyplot(fig)
