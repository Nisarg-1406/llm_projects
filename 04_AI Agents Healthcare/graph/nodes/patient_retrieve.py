from typing import Any, Dict

from graph.state import GraphState
from ingestion import patient_retriever

def patient_retrieve(state: GraphState) -> Dict[str, Any]:
    # print("---RETRIEVE---")
    question = state["question"]

    documents = patient_retriever.invoke(question)
    return {"documents": documents, "question": question}