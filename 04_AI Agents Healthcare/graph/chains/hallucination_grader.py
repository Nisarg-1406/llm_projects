import os
import re
import json
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_core.runnables import RunnableSequence
# from langchain_openai import ChatOpenAI
from langchain_ibm import WatsonxLLM

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# class GradeHallucinations(BaseModel):
#     """Binary score for hallucination present in generation answer."""

#     binary_score: bool = Field(
#         description="Answer is grounded in the facts, 'yes' or 'no'"
#     )


# structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
#      Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
# hallucination_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system),
#         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#     ]
# )

# hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

# Initialize Watsonx LLM
llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct", 
    url=os.environ["WATSONX_URL"],
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={
        "decoding_method": "greedy",
        "temperature": 0,
        "min_new_tokens": 5,
        "max_new_tokens": 100,
        "stop_sequences": ["Human:", "Observation"],
    },
)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

def grade_hallucinations_structured(documents: str, generation: str) -> GradeHallucinations:
    prompt = f"""
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

    Set of facts: 
    {documents}

    LLM generation: {generation}

    Output a JSON object with the following structure:
    {{
        "binary_score": "yes/no"  # 'yes' if the answer is grounded in the facts, 'no' otherwise
    }}
    """
    response = llm.invoke(prompt)
    # print("RESPONSE: ", response)

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("Error: No valid JSON found in LLM response!")

        json_response = json_match.group(0).strip()

        data = json.loads(json_response)
        return GradeHallucinations(**data)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return GradeHallucinations(binary_score=False)

def hallucination_grader(documents: str, generation: str) -> GradeHallucinations:
    return grade_hallucinations_structured(documents, generation)