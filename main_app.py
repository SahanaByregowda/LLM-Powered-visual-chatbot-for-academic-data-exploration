import os
import re
import time
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Custom Modules
from config import API_KEY, MAX_RETRIES_CHART_GEN, JSON_CE, MD_CE, HPC_FOLDER, CSV_FILE_NAME
from document_handlers import ce_retriever_builder, hpc_retriever_builder
from csv_chart_handlers import load_csv_data, get_plot_code_from_llm, execute_generated_code, get_insights_from_llm
from ce_chart_handlers import ask_and_generate_ce_chart_script, execute_generated_ce_chart_code, get_ce_chart_insights_from_llm
from router import route_user_query
from prompts import STYLE_PROMPT, FALLBACK
from json_to_mermaid import generate_mermaid_from_prerequisites



st.set_page_config(page_title="🎓 Academic Assistant Bot", layout="wide")


if not API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

def sanitize_mermaid_code(raw: str) -> str:
    """Cleans Mermaid code to avoid syntax issues with v10+."""
    raw = re.sub(r'-->\s*\|[^|]*?\|\s*', '-->', raw)
    lines = raw.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith("classDef") and line.endswith(";"):
            line = line.rstrip(";")
        if line:
            cleaned.append(line)
    return '\n'.join(cleaned)


def render_mermaid_diagram(mermaid_code: str):
    """Renders a Mermaid diagram in the Streamlit frontend."""
    html_content = f"""
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true }});
    </script>
    <div class="mermaid">{mermaid_code}</div>
    """
    st.components.v1.html(html_content, height=600, scrolling=True)

@st.cache_resource(show_spinner="Loading CSV data...")
def cached_load_csv_data():
    return load_csv_data()

df, df_head_str, df_info_str, column_names, csv_load_error = cached_load_csv_data()
if csv_load_error:
    st.error(csv_load_error)
    st.stop()

@st.cache_resource(show_spinner=" Building CE document vector store …")
def cached_ce_retriever():
    retriever, error = ce_retriever_builder()
    if error: st.error(error); st.stop()
    return retriever

@st.cache_resource(show_spinner=" Building HPC document vector store …")
def cached_hpc_retriever():
    retriever, error = hpc_retriever_builder()
    if error: st.error(error); st.stop()
    elif error == "No .md files found in HPC folder.":
        st.sidebar.warning(f"No .md files found in HPC folder: {HPC_FOLDER}. Functionality may be limited.")
    return retriever

retriever_ce = cached_ce_retriever()
retriever_hpc = cached_hpc_retriever()


st.title("Chat Bot")
st.markdown("Ask about student data (CSV) or program structures (CE/HPC).")


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []


query = st.text_input("Your request", key="user_query_input")
show_debug_info = st.checkbox("Show debug info (retrieved chunks)", value=False)


if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    query_domain = route_user_query(query).strip()
    st.info(f"*Identified Query Domain:* {query_domain}")

    
    retrieval_time, generation_time, visualization_time = 0, 0, 0
    
   
    if query_domain == "CSV_DATA":
        st.subheader(" Generating Data Visualization (from CSV)")
        for attempt in range(MAX_RETRIES_CHART_GEN + 1):
            gen_start = time.time()
            explanation, plot_code = get_plot_code_from_llm(query, st.session_state.chat_history, None)
            generation_time += time.time() - gen_start

            if explanation and plot_code:
                st.markdown(f"*Explanation:*\n{explanation}")
                st.code(plot_code, language="python")
                
                vis_start = time.time()
                error_message_from_exec = execute_generated_code(plot_code, "generated_chart.png")
                visualization_time += time.time() - vis_start

                if not error_message_from_exec:
                    st.success("Chart saved successfully.")
                    st.image("generated_chart.png", caption="Generated Chart", use_column_width=True)

                    gen_start = time.time()
                    insights = get_insights_from_llm(user_query=query, chart_explanation=explanation, chart_type_description=explanation.splitlines()[0], conversation_history=st.session_state.chat_history)
                    generation_time += time.time() - gen_start
                    st.subheader("💡 Textual Insights"); st.write(insights)
                    break 
                
            else:
                st.error("Failed to generate chart code from LLM."); break
        
    
    elif query_domain == "DOCUMENT_CE_CHART":
        st.subheader(" Generating Data Visualization (from CE Curriculum JSON)")
        for attempt in range(MAX_RETRIES_CHART_GEN + 1):
            gen_start = time.time()
            explanation, plot_code = ask_and_generate_ce_chart_script(query, st.session_state.chat_history, None)
            generation_time += time.time() - gen_start

            if explanation and plot_code:
                st.markdown(f"*Explanation:*\n{explanation}")
                st.code(plot_code, language="python")
                
                vis_start = time.time()
                error_message_from_exec = execute_generated_ce_chart_code(plot_code, "generated_ce_chart.png")
                visualization_time += time.time() - vis_start

                if not error_message_from_exec:
                    st.success("CE Chart saved successfully.")
                    st.image("generated_ce_chart.png", caption="Generated CE Chart", use_column_width=True)
                    
                    gen_start = time.time()
                    insights = get_ce_chart_insights_from_llm(user_query=query, chart_explanation=explanation, chart_type_description=explanation.splitlines()[0], conversation_history=st.session_state.chat_history)
                    generation_time += time.time() - gen_start
                    st.subheader("💡 Textual Insights (for CE Chart)"); st.write(insights)
                    break
                else:
                    st.error("Failed to generate CE chart code from LLM."); break

   
    elif query_domain == "DOCUMENT_CE_DIAGRAM":
        diagram_mode = st.radio("Select diagram type:", [" GPT-4 Generated (from Markdown)", "⚡ JSON-based Prerequisite Chart"])
        if diagram_mode == "⚡ JSON-based Prerequisite Chart":
            gen_start = time.time()
            diagram = generate_mermaid_from_prerequisites(JSON_CE)
            generation_time = time.time() - gen_start
            
            vis_start = time.time()
            if "Error" not in diagram:
                st.code(diagram, language="mermaid")
                render_mermaid_diagram(diagram) 
            else: st.error(diagram)
            visualization_time = time.time() - vis_start
        else: 
            ret_start = time.time()
            docs = retriever_ce.get_relevant_documents(query)
            retrieval_time = time.time() - ret_start
            context_text = "\n\n".join(d.page_content for d in docs) + "\n\n" + FALLBACK
            
            gen_start = time.time()
            try:
                chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), retriever=retriever_ce)
                prompt = STYLE_PROMPT.format(context=context_text, query=query)
                answer = chain.run(prompt)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    
                    answer = """Here's a sample CE curriculum structure:

```mermaid
graph TD
    A[Year 1: Foundation] --> B[Year 2: Specialization]
    B --> C[Year 3: Advanced Topics]
    C --> D[Final Project/Thesis]
    
    A --> E[Core Subjects:<br/>- Programming Fundamentals<br/>- Mathematics<br/>- Computer Systems]
    B --> F[Specialization:<br/>- Software Engineering<br/>- Networks<br/>- Databases]
    C --> G[Advanced Topics:<br/>- AI/ML<br/>- Security<br/>- Electives]
    D --> H[Capstone:<br/>- Thesis<br/>- Industrial Project<br/>- Research Work]
    
    style A fill:#e3f2fd
    style B fill:#f1f8e9
    style C fill:#fce4ec
    style D fill:#fff8e1
```

*Note: This is a sample structure. API quota exceeded - please check your OpenAI billing.*"""
                else:
                    raise e
            generation_time = time.time() - gen_start
            
            vis_start = time.time()
            mermaid_raw = answer[answer.find("```mermaid") + len("```mermaid"):answer.rfind("```")]
            mermaid = sanitize_mermaid_code(mermaid_raw)
            st.code(mermaid, language="mermaid")
            render_mermaid_diagram(mermaid) 
            visualization_time = time.time() - vis_start

    
    elif query_domain == "DOCUMENT_HPC":
        st.subheader(" Generating Diagram (from HPC Document)")
        ret_start = time.time()
        docs = retriever_hpc.get_relevant_documents(query)
        retrieval_time = time.time() - ret_start
        context_text = "\n\n".join(d.page_content for d in docs) + "\n\n" + FALLBACK
        
        gen_start = time.time()
        try:
            chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0), retriever=retriever_hpc)
            prompt = STYLE_PROMPT.format(context=context_text, query=query)
            answer = chain.run(prompt)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                
                answer = """Here's a sample mobility semester structure for the HPC program:

```mermaid
graph TD
    A[Semester 1: Home University<br/>Foundation Courses] --> B[Semester 2: Partner University<br/>Specialized HPC Courses]
    B --> C[Semester 3: Industry/Research<br/>Practical Experience]
    C --> D[Semester 4: Home University<br/>Thesis & Capstone]
    
    A --> E[Core Modules:<br/>- Parallel Programming<br/>- Computer Architecture<br/>- Mathematics]
    B --> F[Mobility Modules:<br/>- Advanced HPC<br/>- Distributed Systems<br/>- Research Methods]
    C --> G[Practical Experience:<br/>- Internship<br/>- Research Project<br/>- Industry Collaboration]
    D --> H[Final Deliverables:<br/>- Master Thesis<br/>- Defense<br/>- Portfolio]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
```

*Note: This is a sample structure. API quota exceeded - please check your OpenAI billing.*"""
            else:
                raise e
        generation_time = time.time() - gen_start
        
        vis_start = time.time()
        mermaid_raw = answer[answer.find("```mermaid") + len("```mermaid"):answer.rfind("```")]
        mermaid = sanitize_mermaid_code(mermaid_raw)
        st.code(mermaid, language="mermaid")
        render_mermaid_diagram(mermaid)
        visualization_time = time.time() - vis_start

    
    else:
        st.warning(" I couldn't determine the domain of your query. Try rephrasing.")

    
    if query_domain not in ["Unknown"]:
        total_time = retrieval_time + generation_time + visualization_time
        current_run_timings = {
            "Retrieval": f"{retrieval_time:.2f}s",
            "Generation": f"{generation_time:.2f}s",
            "Visualization": f"{visualization_time:.2f}s",
            "Total": f"{total_time:.2f}s"
        }
        st.session_state.performance_history.append(current_run_timings)

    
    st.session_state.chat_history.append({"role": "assistant", "content": "Response generated."})


if st.session_state.performance_history:
    st.subheader(" Performance History")
    df = pd.DataFrame(st.session_state.performance_history)
    df_transposed = df.T
    df_transposed.columns = [f"Run {i+1}" for i in range(len(df_transposed.columns))]
    st.table(df_transposed)

    if st.button("Clear Performance History"):
        st.session_state.performance_history = []
        st.rerun()