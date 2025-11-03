import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import traceback
import plotly.graph_objects as go
import seaborn as sns
import openai # For OpenAI API calls
from typing import Optional, List, Dict, Tuple

# Import configurations and API key
from config import CSV_FILE_NAME, API_KEY, MAX_RETRIES_CHART_GEN

# Initialize OpenAI client lazily to avoid import-time errors
client = None

def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global client
    if client is None:
        import os
        api_key = API_KEY or os.getenv("OPENAI_API_KEY")
        if api_key:
            client = openai.OpenAI(api_key=api_key)
        else:
            print("Warning: OpenAI API key not found in csv_chart_handlers.py. Ensure it's set in main_app.py.")
    return client

# --- CSV Data Loading and Context Preparation Function ---
def load_csv_data() -> Tuple[pd.DataFrame, str, str, str, Optional[str]]:
    """
    Loads the CSV data and prepares context strings for the LLM.
    Returns df, df_head_str, df_info_str, column_names, and an error message if any.
    """
    df = None
    df_head_str = ""
    df_info_str = ""
    column_names = ""
    error_message = None

    try:
        df = pd.read_csv(CSV_FILE_NAME)
        df_head_str = df.head(3).to_string() # First 3 rows for LLM context
        df_info_str_buffer = io.StringIO()
        df.info(buf=df_info_str_buffer)
        df_info_str = df_info_str_buffer.getvalue() # DataFrame info for LLM context
        column_names = ", ".join(df.columns.tolist()) # Column names for LLM context
    except FileNotFoundError:
        error_message = f"Error: The CSV file '{CSV_FILE_NAME}' was not found."
    except Exception as e:
        error_message = f"Error loading CSV file: {e}"
    
    return df, df_head_str, df_info_str, column_names, error_message

# Globally load data, but handle errors in main_app.py
# This global 'df' will be used by execute_generated_code
_df, _df_head_str, _df_info_str, _column_names, _csv_load_error = load_csv_data()

# These global variables are now set by load_csv_data()
df = _df
df_head_str = _df_head_str
df_info_str = _df_info_str
column_names = _column_names
csv_load_error = _csv_load_error # Expose error status


# --- LLM Interaction Function for Chart Generation ---
def get_plot_code_from_llm(
    user_query: str,
    conversation_history: List[Dict],
    previous_error_message: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sends a prompt to the LLM to get Python code for plotting,
    including context about the dataframe, with support for multiple libraries.
    Includes conversational history and previous error messages for self-correction.
    """
    client = get_openai_client()
    if client is None:
        return "OpenAI client not initialized due to missing API key.", None

    system_message = f"""
    You are an intelligent data visualization assistant.
    The user will provide a natural language query for a chart based on their data.
    The data has been loaded into a pandas DataFrame named df.
    Here is a summary of the data for context:
    DataFrame Columns: {column_names}
    DataFrame Head (first 3 rows):
    
    {df_head_str}
    
    DataFrame Info (types and non-nulls):
    
    {df_info_str}
    

    Your task is to:
    1.  *Analyze the user's query and the provided data context.*
    2.  *Determine the most suitable type of chart and the best plotting library for it.*
        You have access to the following libraries:
        * matplotlib.pyplot (aliased as plt): For general-purpose plots, basic bar/line/scatter/histograms, when simplicity or fine-grained control is needed.
        * seaborn (aliased as sns): For statistical plots, visualizing relationships between variables, distributions, and creating aesthetically pleasing plots with fewer lines of code. Often preferred for scatter plots, box plots, violin plots, heatmaps, pair plots, etc.
        * plotly.graph_objects (aliased as go): For interactive charts, web-based visualizations, and specialized plots like Sankey diagrams or choropleth maps. Use when interactivity or specific complex plot types are requested.
        *Prioritize based on user intent:* If a statistical plot is implied, use seaborn. If interactivity or a specific Plotly-supported chart (like Sankey) is implied, use plotly. Otherwise, default to matplotlib.
    3.  *Generate a complete, executable Python script* that:
        * Assumes the data is already loaded into a pandas DataFrame df.
        * Includes appropriate labels for x and y axes, and a descriptive title.
        * Handles common data aggregation or preparation if needed (e.g., groupby().sum(), value_counts(), melt()).
        * *Always saves the generated chart to a file named 'generated_chart.png'.*
        * *Does NOT include any plt.show(), fig.show(), or similar display calls.*
        * *For Matplotlib (plt):*
            * Ensure import matplotlib.pyplot as plt.
            * Include plt.tight_layout() for better label fitting.
            * Include plt.close() at the end to free memory.
            * Save using plt.savefig('generated_chart.png').
        * *For Seaborn (sns):*
            * Ensure import seaborn as sns and import matplotlib.pyplot as plt.
            * Plotting functions often return a Matplotlib Axes object (ax). Use fig, ax = plt.subplots() if you need an explicit figure.
            * Include plt.tight_layout() and plt.close().
            * Save using plt.savefig('generated_chart.png').
        * *For Plotly (go):*
            * Ensure import plotly.graph_objects as go and import pandas as pd.
            * Prepare data as needed (e.g., for Sankey, map labels to indices).
            * Create figure using fig = go.Figure(...).
            * Save using fig.write_image("generated_chart.png") (requires kaleido package).
        * *Crucially, wrap the Python code in a markdown block like so:*
            python
            # your code here
            
    4.  *Before the code block, provide a brief explanation* of why you chose that particular chart type and library.

    """
    if previous_error_message:
        system_message += f"\n\n*IMPORTANT: The previous attempt to execute your generated code failed with the following error:\n\n{previous_error_message}\n\nPlease review the error and generate a corrected Python code block. Focus on fixing the exact issue.*"
    
    # Prepare messages for the LLM, including history
    messages = []
    # Add relevant history, limiting to last few turns to manage token usage
    for msg in conversation_history[-4:]: # Limit history to recent interactions
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.insert(0, {"role": "system", "content": system_message})
    messages.append({"role": "user", "content": f"Please generate a chart for: {user_query}"})

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", # Using gpt-4o for best performance in code generation and reasoning
            messages=messages,
            temperature=0.1, # Keep low for deterministic code
            max_tokens=1500 # Allow sufficient tokens for code and explanation
        )
        content = response.choices[0].message.content
        explanation_match = re.match(r"^(.*?)\s*```python", content, re.DOTALL) # Adjusted regex
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL) # Adjusted regex

        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."
        code = code_match.group(1).strip() if code_match else None

        if code:
            return explanation, code
        else:
            return "No Python code block found in LLM response.", None # Return error message

    except openai.APIError as e:
        if "429" in str(e) or "quota" in str(e).lower():
            # Return fallback code for quota exceeded
            fallback_code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample analysis for gender distribution across cohorts
title = "Sample Gender Distribution Across Cohorts"

# Create sample data
sample_data = {
    'Cohort': ['2020', '2021', '2022', '2023'] * 2,
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female'],
    'Count': [45, 48, 52, 50, 35, 42, 48, 45]
}

df_sample = pd.DataFrame(sample_data)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_sample, x='Cohort', y='Count', hue='Gender', palette=['lightblue', 'lightpink'])
plt.title(title)
plt.xlabel('Cohort')
plt.ylabel('Number of Students')
plt.legend(title='Gender')
plt.tight_layout()
plt.savefig("generated_chart.png")
plt.clf()
"""
            return "API quota exceeded - showing sample chart", fallback_code
        return f"OpenAI API Error during chart code generation: {e}", None
    except Exception as e:
        return f"An unexpected error occurred during LLM call for chart code: {e}", None

# --- Code Execution Function ---
def execute_generated_code(code: str, output_filename: str = "generated_chart.png") -> Optional[str]:
    """
    Executes the generated Python code in a controlled environment.
    Uses the globally loaded 'df' (DataFrame) from this module.
    Returns None on success, or an error message string on failure.
    """
    if df is None:
        return "DataFrame 'df' is not loaded. Cannot execute chart code."

    exec_globals = {
        'pd': pd,
        'plt': plt,
        'io': io,
        'df': df,  # Use the globally loaded df
        'output_filename': output_filename,
        '__builtins__': {
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'dict': dict,
            'list': list,
            'tuple': tuple,
            'set': set,
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'abs': abs,
            '__import__': __import__,
            'enumerate': enumerate, # Added for Sankey diagram
        }
    }

    # Conditionally add seaborn and plotly to exec_globals if used in the code
    if "import seaborn as sns" in code:
        exec_globals['sns'] = sns
    if "import plotly.graph_objects as go" in code:
        exec_globals['go'] = go
        # Plotly requires kaleido for write_image, ensure it's imported for exec
        try:
            import kaleido
            exec_globals['kaleido'] = kaleido
        except ImportError:
            return "Plotly requires the 'kaleido' package to save images. Please install it (`pip install kaleido`)."


    # Modify code for saving and preventing display
    modified_code = code.replace("plt.savefig('generated_chart.png')", f"plt.savefig(output_filename)")
    modified_code = modified_code.replace("fig.write_image('generated_chart.png')", f"fig.write_image(output_filename)")
    modified_code = modified_code.replace("plt.show()", "# plt.show() - disabled by wrapper")
    modified_code = modified_code.replace("fig.show()", "# fig.show() - disabled by wrapper")

    # Ensure plt.close() is called for matplotlib/seaborn plots
    if "plt.close()" not in modified_code and ("plt." in modified_code or "sns." in modified_code):
        modified_code += "\nplt.close()"

    try:
        exec(modified_code, exec_globals)
        # st.success(f"Chart saved successfully as '{output_filename}'") # Removed st.success
        return None # Indicate success by returning None
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        # st.error(f"Error executing generated code: {error_type}: {error_message}") # Removed st.error
        # st.code(f"--- Traceback ---\n{full_traceback}\n--- Generated Code ---\n{modified_code}", language="python") # Removed st.code
        return f"{error_type}: {error_message}\nFull Traceback:\n{full_traceback}" # Return error message


# --- LLM Interaction Function for Textual Insights ---
def get_insights_from_llm(
    user_query: str,
    chart_explanation: str,
    chart_type_description: str,
    conversation_history: List[Dict]
) -> str:
    """
    Sends a prompt to the LLM to get textual insights about the generated chart.
    Includes conversational history.
    """
    client = get_openai_client()
    if client is None:
        return "OpenAI client not initialized due to missing API key. Cannot generate insights."

    system_message = f"""
    You are an intelligent data analyst.
    The user has just generated a chart based on their data.
    Your task is to provide a concise textual summary and key insights derived from the chart and the underlying data.
    Relate the insights back to the user's original query and the context of the conversation.

    Here is the chart context:
    Original User Query: "{user_query}"
    Chart Type Explanation (from previous LLM generation): "{chart_explanation}"
    Description of what was plotted: "{chart_type_description}"
    DataFrame Columns: {column_names}
    DataFrame Info:
    
    {df_info_str}
    ```

    Focus on what the chart reveals. Avoid repeating information already in the chart explanation.
    Provide 2-3 key bullet points or a short paragraph.
    """

    messages = []
    for msg in conversation_history[-4:]: # Limit history to recent interactions
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.insert(0, {"role": "system", "content": system_message})
    messages.append({"role": "user", "content": "Please provide insights for the chart that was just generated."})


    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo", # Using gpt-4o for best performance
            messages=messages,
            temperature=0.4, # Slightly higher temperature for more insightful text
            max_tokens=300 # Sufficient tokens for a summary
        )
        return response.choices[0].message.content.strip()

    except openai.APIError as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return "API quota exceeded. Key insight: The chart shows the distribution pattern requested, but detailed analysis is unavailable due to API limits. Please check your OpenAI billing to enable full insights."
        return f"OpenAI API Error (for insights): {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLM call for insights: {e}"
