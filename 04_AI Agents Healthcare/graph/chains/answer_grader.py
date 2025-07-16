import os
import re
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ibm import WatsonxLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableSequence
# from langchain_openai import ChatOpenAI

# class GradeAnswer(BaseModel):
#     binary_score: bool = Field(
#         description="Answer addresses the question, 'yes' or 'no'"
#     )
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# structured_llm_grader = llm.with_structured_output(GradeAnswer)

# system = """You are a grader assessing whether an answer addresses / resolves a question \n 
#      Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
# answer_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
#     ]
# )

# answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct", 
    url=os.environ["WATSONX_URL"],
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={
        "decoding_method": "greedy",
        "temperature": 0,
        "min_new_tokens": 1,
        "max_new_tokens": 100,
        "stop_sequences": ["Human:", "Observation"],
    },
)

class GradeAnswer(BaseModel):
    """Binary score for whether the answer addresses the question."""
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")

# def grade_answer_structured(question: str, generation: str) -> GradeAnswer:
#     prompt = f"""
#     You are a grader assessing whether an answer addresses / resolves a question.
#     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.

#     User question: 
#     {question}

#     LLM generation: {generation}

#     Output a JSON object with the following structure:
#     {{
#         "binary_score": "yes/no"  # 'yes' if the answer resolves the question, 'no' otherwise
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
#         return GradeAnswer(**data)
#     except Exception as e:
#         print(f"Error parsing LLM response: {e}")
#         return GradeAnswer(binary_score=False)

def grade_answer_structured(question: str, generation: str) -> GradeAnswer:
    prompt = f"""
    You are a grader assessing whether an answer addresses a question.
    Give a binary score '__yes__' or '__no__'. 
    '__yes__' means that the answer resolves the question.
    Output strictly only one word: '__yes__' or '__no__'. Do not add any explanation or additional text.

    User question: 
    {question}

    LLM generation: {generation}
    """

    response = llm.invoke(prompt)
    response_lower = response.strip().lower()
    # print("response_lower: ", response_lower)
    
    if "__yes__" in response_lower:
        return GradeAnswer(binary_score="yes")
    elif "__no__" in response_lower:
        return GradeAnswer(binary_score="no")
    
    # print("Error parsing LLM response: Could not extract binary score.")
    return GradeAnswer(binary_score="no")

def answer_grader(question: str, generation: str) -> GradeAnswer:
    return grade_answer_structured(question, generation)