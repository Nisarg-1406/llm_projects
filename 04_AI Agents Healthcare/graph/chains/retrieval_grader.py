import os
import re
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ibm import WatsonxLLM

llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct", 
    url=os.environ["WATSONX_URL"],
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={
        "decoding_method": "greedy",
        "temperature": 0,
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "stop_sequences": ["Human:", "Observation"],
    },
)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# def grade_documents_structured(query: str, document: str) -> GradeDocuments:
#     prompt = f"""
#     You are a grader assessing relevance of a retrieved document to a user question. 
#     If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

#     Retrieved document: 
#     {document}

#     User question: {query}

#     Output a JSON object with the following structure:
#     {{
#         "binary_score": "yes/no"  # 'yes' if the document is relevant, 'no' otherwise
#     }}
#     """

#     response = llm.invoke(prompt)
#     print("RESPONSE: ", response)

#     try:
#         json_match = re.search(r"\{.*\}", response, re.DOTALL)
#         if not json_match:
#             raise ValueError("Error: No valid JSON found in LLM response!")

#         json_response = json_match.group(0).strip()

#         data = json.loads(json_response)
#         return GradeDocuments(**data)
#     except Exception as e:
#         print(f"Error parsing LLM response: {e}")
#         return GradeDocuments(binary_score="no")

def grade_documents_structured(query: str, document: str) -> GradeDocuments:
    prompt = f"""
    You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score '__yes__' or '__no__' to indicate whether the document is relevant to the question.
    Output strictly only one word: '__yes__' or '__no__'. Do not add any explanation or additional text.

    Retrieved document: 
    {document}

    User question: {query}
    """

    response = llm.invoke(prompt)
    response_lower = response.strip().lower()
    # print("response_lower: ", response_lower)
    
    if "__yes__" in response_lower:
        return GradeDocuments(binary_score="yes")
    elif "__no__" in response_lower:
        return GradeDocuments(binary_score="no")
    
    # print("Error parsing LLM response: Could not extract binary score.")
    return GradeDocuments(binary_score="no")

# Custom retrieval grader
def retrieval_grader(query: str, document: str) -> GradeDocuments:
    return grade_documents_structured(query, document)

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""

#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )

# structured_llm_grader = llm.with_structured_output(GradeDocuments)

# system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
#     If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
# grade_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
#     ]
# )
# retrieval_grader = grade_prompt | structured_llm_grader