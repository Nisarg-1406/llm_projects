import os
from typing import List, TypedDict
from dotenv import load_dotenv
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE
from graph.nodes import generate, grade_documents, doctor_retrieve
from langchain_core.runnables import RunnableSequence
from langgraph.graph import END, StateGraph
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

diabetes_loader = TextLoader("/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/diabetes.txt")
low_hemoglobin_loader = TextLoader("/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/low_hemoglobin.txt")

diabetes_docs = diabetes_loader.load()
low_hemoglobin_docs = low_hemoglobin_loader.load()

docs_list = diabetes_docs + low_hemoglobin_docs

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embedding_function,
#     persist_directory="/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/chroma_db_1",
# )
# Retrieve all stored documents
# docs = vectorstore.get(include=['documents', 'metadatas'])
# print(docs)
# # Print the stored documents
# for doc, metadata in zip(docs['documents'], docs['metadatas']):
#     print(f"Document: {doc}\nMetadata: {metadata}\n{'-'*50}")

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="data/chroma_db",
    embedding_function=embedding_function,
).as_retriever()

llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct", 
    url=os.environ["WATSONX_URL"],
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 1000,
        GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
    },
)

prompt_template = PromptTemplate(
    input_variables=["documents"],
    template="""Based on the following documents, provide additional information about symptoms and their cure: {documents}. 
    Details should strictly come from the data stored in the database. ONLY give symptoms and cure for asked medical_name. 
    Do not give any other information outside this document.""",
)

additional_info_chain: RunnableSequence = prompt_template | llm

HUMAN_IN_LOOP = "human_in_loop"

class GraphState(TypedDict):
    """
    Attributes:
        question: The user's question
        generation: LLM generation
        web_search: Whether to perform a web search
        documents: List of retrieved documents
        additional_info: Additional information retrieved from Chroma DB (if requested by the user)
    """
    question: str
    generation: str
    documents: List[str]
    additional_info: str
    medical_name: str

def decide_to_generate(state):
    # print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        # print(
        #     "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        # )
        state["generation"] = "Please Try Again!!"
        return END
    else:
        # print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    # print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader(documents, generation)

    if hallucination_grade := score.binary_score:
        # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader(question, generation)
        if answer_grade := score.binary_score:
            # print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            # print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            state["generation"] = "Please Try Again!!"
            return END
    else:
        # print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def human_in_loop(state: GraphState) -> dict:
    # print("---HUMAN IN LOOP---")
    # user_input = input("Want to learn more about the symptoms and its cure? (yes/no): ")
    user_input = os.environ.get("HUMAN_IN_LOOP_OVERRIDE")
    if user_input == "yes":
        # medical_name = input("Enter the medical name (e.g., diabetes, low hemoglobin): ")
        medical_name = os.environ.get("MEDICAL_NAME_OVERRIDE")
        state["medical_name"] = medical_name
        # print(f"---FETCHING ADDITIONAL DOCUMENTS FOR {medical_name.upper()} FROM CHROMA DB---")
        
        documents = retriever.invoke(state["medical_name"])
        
        document_texts = [doc.page_content for doc in documents]
        documents_str = "\n\n".join(document_texts)
        # print("---GENERATING ADDITIONAL INFORMATION---")
        state["additional_info"] = additional_info_chain.invoke({"documents": documents_str})
    else:
        # print("---NO ADDITIONAL INFORMATION REQUESTED---")
        state["additional_info"] = None

    return state

def combine_responses(state: GraphState) -> dict:
    # print("---COMBINING RESPONSES---")
    if state.get("additional_info"):
        state["generation"] = f"{state['generation']}\nAdditional Information:\n{state['additional_info']}"
    # print("---FINAL RESPONSE READY---")
    return state

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, doctor_retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)

workflow.add_node(HUMAN_IN_LOOP, human_in_loop)
workflow.add_node("combine_responses", combine_responses)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        END: END,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": HUMAN_IN_LOOP,
        END: END,
    },
)
workflow.add_conditional_edges(
    HUMAN_IN_LOOP,
    lambda state: "combine_responses" if state.get("additional_info") else END,
    {
        "combine_responses": "combine_responses",
        END: END,
    },
)
workflow.add_edge("combine_responses", END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph_AgentDoctor.png")