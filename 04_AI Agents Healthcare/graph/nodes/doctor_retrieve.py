from typing import Any, Dict

from graph.state import GraphState
from ingestion import doctor_retriever


def doctor_retrieve(state: GraphState) -> Dict[str, Any]:
    # print("---RETRIEVE---")
    question = state["question"]

    documents = doctor_retriever.invoke(question)
    return {"documents": documents, "question": question}
