# router.py

# --- Keywords for Routing ---

# Prioritize general chart-specific keywords for generic CSV data analysis
CSV_CHART_KEYWORDS = {
    "chart", "plot", "graph", "visualize", "show me", "distribution", "count",
    "number of", "percentage", "students", "nationality", "gender", "cohort", "compare",
    "demographics", "csv"
}

# Keywords highly specific to HPC program structure and non-chart requests (mostly diagrams/text)
HPC_KEYWORDS = {
    "hpc", "eumaster4hpc", "mobility", "summer school", "challenge", "workshop", "internship",
    "application", "apply", "admission", "procedure", "documents", "requirements",
    "thesis", "master thesis", "career", "jobs", "consortium", "governance", "news",
    "contact", "universities", "study programme", "extracurricular", "teaching materials",
    "academic journey", "year one", "year two", "specialisation", "program",
    "flowchart", "diagram", "structure", "process", "pathway", "steps", "visualize flow"
}

# Keywords specific to CE curriculum content that usually imply a DIAGRAM (e.g., prerequisites)
CE_DIAGRAM_KEYWORDS = {
    "prerequisites", "module flow", "curriculum structure diagram", "ce diagram",
    "computational engineering diagram", "show modules flow", "draw ce prerequisites"
}

# Keywords specific to CE curriculum content that usually imply a CHART (e.g., ECTS, workload)
CE_CHART_KEYWORDS = {
    "ce chart", "computational engineering chart", "ce plot", "ects distribution",
    "workload distribution", "modules by department", "module count by department",
    "average ects", "total workload", "show ce data", "ce statistics"
}

def route_user_query(q: str) -> str:
    """
    Routes the user query to the appropriate data domain.
    Returns "CSV_DATA", "DOCUMENT_HPC", "DOCUMENT_CE_DIAGRAM", "DOCUMENT_CE_CHART", or "UNKNOWN".
    The order of checks is crucial for correct routing.
    """
    q_lower = q.lower()

    # 1. Check for CE Chart specific queries (e.g., "CE ECTS distribution chart")
    if any(kw in q_lower for kw in CE_CHART_KEYWORDS):
        return "DOCUMENT_CE_CHART"

    # 2. Check for CE Diagram specific queries (e.g., "prerequisites diagram for CE")
    if any(kw in q_lower for kw in CE_DIAGRAM_KEYWORDS):
        return "DOCUMENT_CE_DIAGRAM"

    # 3. Check for general CSV/Chart related queries.
    # This should come after specific CE chart/diagram requests to avoid overlap.
    if any(kw in q_lower for kw in CSV_CHART_KEYWORDS):
        # Add a heuristic to prevent misclassification if a CE/HPC query also uses a general chart term
        # This check is less strict now that CE_CHART_KEYWORDS is handled first
        if not (any(hpc_kw in q_lower for hpc_kw in HPC_KEYWORDS)):
            return "CSV_DATA"

    # 4. Check for HPC document specific queries (diagrams/text)
    if any(kw in q_lower for kw in HPC_KEYWORDS):
        return "DOCUMENT_HPC"
    
    return "UNKNOWN" # Fallback if no specific domain is identified

