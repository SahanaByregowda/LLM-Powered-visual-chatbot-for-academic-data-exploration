import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from collections import defaultdict
import io
import contextlib
import re
import openai 


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate 
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.documents import Document 
# from langchain_community.llms import Together 

from typing import Optional, List, Dict, Tuple


from config import JSON_CE, API_KEY, MAX_RETRIES_CHART_GEN, EMBED_MODEL

# Initialize OpenAI client 
client = None

def get_openai_client():
    """initialization of OpenAI client"""
    global client
    if client is None:
        api_key = API_KEY or os.getenv("OPENAI_API_KEY")
        if api_key:
            client = openai.OpenAI(api_key=api_key)
        else:
            print("Warning: OpenAI API key not found. Ensure it's set in your environment or config.py.")
    return client

# Helper Functions
def clean_and_parse_number(value_str):
    """
    Cleans and parses a string to extract a float number.
    Handles commas as decimal separators, "??", and extracts numbers before units.
    """
    if not isinstance(value_str, str):
        return 0.0
    value_str = value_str.strip().replace(",", ".")
    if "??" in value_str:
        return 0.0
    try:
        # Extract the first numerical part
        match = re.search(r'(\d+(\.\d+)?)', value_str)
        if match:
            return float(match.group(1))
        return 0.0
    except (ValueError, IndexError):
        return 0.0

def extract_workload(workload_str):
    """
    Parses a workload string to extract contact and independent study hours.
    Returns a tuple (contact_hours, independent_study).
    """
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

# Load Curriculum Data
curriculum_data = []
try:
    with open(JSON_CE, "r", encoding="utf-8") as f:
       
        loaded_data = json.load(f)
        if isinstance(loaded_data, list):
            curriculum_data = loaded_data
        else:
            
            print(f"Warning: JSON file at {JSON_CE} did not contain a top-level list. Attempting to get 'curriculum' key if it's a dict.")
            curriculum_data = loaded_data.get("curriculum", [])
            if not curriculum_data:
                print(f"Error: JSON file at {JSON_CE} is not a list and does not contain a 'curriculum' key. CE chart generation will not work.")

except FileNotFoundError:
    print(f"Error: Curriculum JSON file not found at {JSON_CE}. CE chart generation will not work.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {JSON_CE}. CE chart generation will not work. Check JSON format.")
except Exception as e:
    print(f"An unexpected error occurred loading CE curriculum data: {e}")

# LLM Setup (for chart generation from CE JSON) 
# This LLM is specifically for generating Python code for charts from CE JSON
# Using GPT-4-turbo as specified in the prompt

llm_chart_gen = None

def get_llm_chart_gen():
    """Initialization of the LLM client"""
    global llm_chart_gen
    if llm_chart_gen is None:
        llm_chart_gen = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)
    return llm_chart_gen

# Prompt Template 
# This prompt is specific to generating Python charts from the curriculum_data
SYSTEM_PROMPT_CE_CHART = """
You are a Python data analysis and visualization expert.

You are given a list of module dictionaries from a university curriculum. Your task is to generate a Python function named `generate_chart_data(data)` that processes this data using `pandas` and generates charts using `matplotlib` or `seaborn`.

Instructions:
- The input `data` is a list of dictionaries, where each dictionary represents a module.
- Handle messy or inconsistent formats safely.
- Do NOT return JSON. Only return a Python code block with a single function definition.
- The generated chart MUST be saved to a file named 'generated_chart.png'.
- Do NOT include `plt.show()` or `fig.show()` calls.
- After saving, ensure `plt.clf()` is called to clear the plot.

You **must** use the helper functions provided below for data cleaning and parsing:
```python
def clean_and_parse_number(value_str):
    # Parses strings like "7,5 ECTS", "??", "5", or "90 h"
    # Returns a float.
    if not isinstance(value_str, str): return 0.0
    value_str = value_str.strip().replace(",", ".")
    if "??" in value_str: return 0.0
    try:
        match = re.search(r'(\d+(\.\d+)?)', value_str)
        if match: return float(match.group(1))
        return 0.0
    except (ValueError, IndexError): return 0.0

def extract_workload(workload_str):
    # Parses strings like "Contact hours: 60 h\\nIndependent study: 90 h"
    # Returns a tuple (contact_hours, independent_study) in float.
    contact_hours = 0.0
    independent_study = 0.0
    if not isinstance(workload_str, str): return contact_hours, independent_study
    lines = workload_str.strip().split("\\n")
    for line in lines:
        line = line.lower()
        if "contact hours" in line or "attendance time" in line:
            contact_hours += clean_and_parse_number(line)
        elif "independent study" in line or "self study" in line:
            independent_study += clean_and_parse_number(line)
    return contact_hours, independent_study
```

Important logic for workload-related questions:
If the user asks about total workload, average workload, or workload per department:
- Use `contact, independent = extract_workload(module.get("workload", ""))`
- Calculate `total_hours = contact + independent`
- Aggregate by department using `module.get("Department", "Unknown")`

In the `generate_chart_data` function:
- Define `title = "..."` early.
- Process and aggregate the data into a DataFrame.
- Create a plot with `plt` or `sns` (e.g., bar, pie, or line chart).
- Set axis labels and title using the `title` variable.
- Save the figure to a file:
  ```python
  filename = title.lower().replace(" ", "_").replace(":", "") + ".png" # Ensure valid filename
  plt.savefig(filename)
  ```
- Clear the chart with `plt.clf()` afterward.

Do not add explanations, comments, or markdown outside the code block — only return one Python function inside a code block. Use the provided curriculum data context below for analysis.

Curriculum data context (a sample of the structure, not the full data):
```json
[
    {{ "Module Name": "Example Module 1", "ECTS": "7,5 ECTS", "Department": "Informatics", "workload": "Contact hours: 30 h\\nIndependent study: 45 h", "prerequisites": [] }},
    {{ "Module Name": "Example Module 2", "ECTS": "5 ECTS", "Department": "Mathematics", "workload": "Contact hours: 20 h\\nIndependent study: 30 h", "prerequisites": ["Example Module 1"] }}
]
```
"""

# Execute and Visualize Function
def ask_and_generate_ce_chart_script(
    question: str,
    conversation_history: List[Dict],
    previous_error_message: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sends a prompt to the LLM to get Python code for plotting CE curriculum data.
    Includes conversational history and previous error messages for self-correction.
    Returns explanation and code.
    """
    client = get_openai_client()
    if client is None:
        return "OpenAI client not initialized.", None
    if not curriculum_data:
        return "CE curriculum data not loaded. Cannot generate charts.", None

    # Prepare system message with error feedback if applicable
    current_system_prompt = SYSTEM_PROMPT_CE_CHART
    if previous_error_message:
        current_system_prompt += f"\n\n*IMPORTANT: The previous attempt to execute your generated code failed with the following error:\n\n{previous_error_message}\n\nPlease review the error and generate a corrected Python code block. Focus on fixing the exact issue.*"

    # Prepare messages for the LLM, including history
    messages = []
    messages.insert(0, {"role": "system", "content": current_system_prompt})
    # Add relevant history, limiting to last few turns to manage token usage
    for msg in conversation_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=messages,
            temperature=0.1, 
            max_tokens=1500 
        )
        content = response.choices[0].message.content

        # Extract explanation and code blocks
        explanation_match = re.match(r"^(.*?)\s*```python", content, re.DOTALL)
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)

        explanation = explanation_match.group(1).strip() if explanation_match else "No explicit explanation provided (code expected directly)."
        code = code_match.group(1).strip() if code_match else None

        if code:
            return explanation, code
        else:
            return "No Python code block found in LLM response.", None

    except openai.APIError as e:
        if "429" in str(e) or "quota" in str(e).lower():
            # Return fallback code for quota exceeded
            fallback_code = """
def generate_chart_data(data):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    title = "Sample CE Curriculum Distribution"
    
    # Sample data for demonstration
    departments = ['Computer Science', 'Mathematics', 'Physics', 'Engineering']
    module_counts = [15, 8, 6, 12]
    
    df = pd.DataFrame({'Department': departments, 'Module Count': module_counts})
    
    plt.figure(figsize=(10, 6))
    plt.bar(df['Department'], df['Module Count'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(title)
    plt.xlabel('Department')
    plt.ylabel('Number of Modules')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = title.lower().replace(" ", "_").replace(":", "") + ".png"
    plt.savefig(filename)
    plt.clf()
"""
            return "API quota exceeded - showing sample chart", fallback_code
        return f"OpenAI API Error during CE chart code generation: {e}", None
    except Exception as e:
        return f"An unexpected error occurred during LLM call for CE chart code: {e}", None

# Code Execution Function 
def execute_generated_ce_chart_code(code: str, output_filename: str = "generated_ce_chart.png") -> Optional[str]:
    """
    Executes the generated Python code for CE curriculum charts in a controlled environment.
    Uses the globally loaded 'curriculum_data' from this module.
    Returns None on success, or an error message string on failure.
    """
    if not curriculum_data:
        return "CE curriculum data is not loaded. Cannot execute chart code."

    exec_globals = {
        "json": json,
        "defaultdict": defaultdict,
        "re": re,
        "pd": pd,
        "plt": plt,
        "clean_and_parse_number": clean_and_parse_number,
        "extract_workload": extract_workload,
        "data": curriculum_data, 
        "output_filename": output_filename,
        '__builtins__': { 
            'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
            'dict': dict, 'list': list, 'tuple': tuple, 'set': set,
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'min': min, 'max': max, 'sum': sum, 'round': round, 'abs': abs,
            '__import__': __import__, 
        }
    }

    
    if "import seaborn as sns" in code:
        exec_globals['sns'] = sns

    # Modify code for saving and preventing display
    modified_code = code.replace("plt.savefig(filename)", f"plt.savefig(output_filename)")
    modified_code = modified_code.replace("plt.show()", "# plt.show() - disabled by wrapper")
    modified_code = modified_code.replace("plt.clf()", "# plt.clf() - handled by wrapper if needed") # Let wrapper manage clf

    # Ensure plt.close() and plt.clf() are called for matplotlib/seaborn plots
    if "plt." in modified_code or "sns." in modified_code:
        if "plt.close()" not in modified_code:
            modified_code += "\nplt.close()"
        if "plt.clf()" not in modified_code: # Ensure clf is called to prevent plot overlap in Streamlit
            modified_code += "\nplt.clf()"


    try:
      
        exec(modified_code, exec_globals)

        
        if 'generate_chart_data' in exec_globals:
            exec_globals['generate_chart_data'](curriculum_data) 
            return None 
        else:
            return "Generated code did not define 'generate_chart_data' function."

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        return f"{error_type}: {error_message}\nFull Traceback:\n{full_traceback}\nGenerated Code:\n{modified_code}"

# LLM Interaction Function for Textual Insights
def get_ce_chart_insights_from_llm(
    user_query: str,
    chart_explanation: str,
    chart_type_description: str,
    conversation_history: List[Dict]
) -> str:
    """
    Sends a prompt to the LLM to get textual insights about the generated CE chart.
    Includes conversational history.
    """
    client = get_openai_client()
    if client is None:
        return "OpenAI client not initialized. Cannot generate insights."
    if not curriculum_data:
        return "CE curriculum data not loaded. Cannot generate insights."

    system_message = f"""
    You are an intelligent data analyst, specializing in university curriculum data.
    The user has just generated a chart based on the CE curriculum data.
    Your task is to provide a concise textual summary and key insights derived from the chart and the underlying curriculum data.
    Relate the insights back to the user's original query and the context of the conversation.

    *IMPORTANT: When discussing distributions, counts, or totals, always provide the exact numerical values for each category if they are available or inferable from the chart's context or the provided curriculum data.* For example, if discussing module count by department, state "Department X has Y modules." or "The total workload for Department Z is A hours."

    Here is the chart context (derived from CE curriculum data):
    Original User Query: "{user_query}"
    Chart Type Explanation (from previous LLM generation): "{chart_explanation}"
    Description of what was plotted: "{chart_type_description}"
    
    A sample of the curriculum data structure is:
    ```json
    [
        {{ "Module Name": "Example Module", "ECTS": "7,5 ECTS", "Department": "Informatics", "workload": "Contact hours: 30 h\\nIndependent study: 45 h", "prerequisites": [] }}
    ]
    ```
    
    You have access to the full `curriculum_data` list of dictionaries.
    Focus on what the chart reveals. Avoid repeating information already in the chart explanation.
    Provide 2-3 key bullet points or a short paragraph.
    """

    messages = []
    messages.insert(0, {"role": "system", "content": system_message})
    for msg in conversation_history[-4:]: # Limit history to recent interactions
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": "Please provide insights for the chart that was just generated."})

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=messages,
            temperature=0.4, 
            max_tokens=300 
        )
        return response.choices[0].message.content.strip()

    except openai.APIError as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return "API quota exceeded. Key insight: The chart shows the curriculum data distribution requested, but detailed analysis is unavailable due to API limits. Please check your OpenAI billing to enable full insights."
        return f"OpenAI API Error (for CE chart insights): {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM call for CE chart insights: {e}"

