import json

def generate_mermaid_from_prerequisites(json_path: str) -> str:
    """
    Loads curriculum JSON and generates a Mermaid flowchart based on prerequisites.
    Each prerequisite is a directed edge.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading JSON file: {e}"

    edges = []
    all_topics = set()
    for entry in data.get("curriculum", []):
        topic = entry.get("topic", "Untitled")
        all_topics.add(topic)
        for prereq in entry.get("prerequisites", []):
            edges.append(f'"{prereq}" --> "{topic}"')

    if not edges:
        return "flowchart TD\n  A[No prerequisites found in data]"

    # Optionally, add nodes with no prerequisites so they still show
    topics_with_prereqs = {edge.split(" --> ")[1].strip('"') for edge in edges}
    root_topics = all_topics - topics_with_prereqs
    for root in root_topics:
        edges.append(f'"{root}"')  # standalone node

    diagram = "flowchart TD\n  " + "\n  ".join(edges)
    return diagram
