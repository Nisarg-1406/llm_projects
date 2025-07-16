from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# history_loader = TextLoader("/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/medical_history.txt")
# docs_list = history_loader.load()

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=350, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)

# nutrition_loader = TextLoader("/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/nutrition_plan.txt")
# nutrition_docs = nutrition_loader.load()

# excercise_loader = TextLoader("/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/excercise_plan.txt")
# excercise_docs = excercise_loader.load()

# docs_list = nutrition_docs + excercise_docs

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=410, chunk_overlap=10
# )
# doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embedding_function,
#     persist_directory="/Users/nisargmehta/Documents/LLM/langchain/AI_Agents/ai_agents_healthcare/data/doctor_chroma",
# )

doctor_retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="data/doctor_chroma",
    embedding_function=embedding_function,
).as_retriever()

patient_retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="data/patient_chroma",
    embedding_function=embedding_function,
).as_retriever()
