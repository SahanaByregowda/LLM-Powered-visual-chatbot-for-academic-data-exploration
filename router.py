# router.py

#  Keywords for Routing 


CSV_CHART_KEYWORDS = {
    "chart", "plot", "graph", "visualize", "show me", "distribution", "count",
    "number of", "percentage", "students", "nationality", "gender", "cohort", "compare",
    "demographics", "csv"
}


HPC_KEYWORDS = {
    "hpc", "eumaster4hpc", "mobility", "summer school", "challenge", "workshop", "internship",
    "application", "apply", "admission", "procedure", "documents", "requirements",
    "thesis", "master thesis", "career", "jobs", "consortium", "governance", "news",
    "contact", "universities", "study programme", "extracurricular", "teaching materials",
    "academic journey", "year one", "year two", "specialisation", "program",
    "flowchart", "diagram", "structure", "process", "pathway", "steps", "visualize flow"
}


CE_DIAGRAM_KEYWORDS = {
    "prerequisites", "module flow", "curriculum structure diagram", "ce diagram",
    "computational engineering diagram", "show modules flow", "draw ce prerequisites"
}


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

    
    if any(kw in q_lower for kw in CE_CHART_KEYWORDS):
        return "DOCUMENT_CE_CHART"

   
    if any(kw in q_lower for kw in CE_DIAGRAM_KEYWORDS):
        return "DOCUMENT_CE_DIAGRAM"


    if any(kw in q_lower for kw in CSV_CHART_KEYWORDS):
        
        if not (any(hpc_kw in q_lower for hpc_kw in HPC_KEYWORDS)):
            return "CSV_DATA"

 
    if any(kw in q_lower for kw in HPC_KEYWORDS):
        return "DOCUMENT_HPC"

    return "UNKNOWN" 
