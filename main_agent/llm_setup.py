import os
from langchain_ollama import OllamaLLM 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


def get_local_llm():
    try:
        
        llm = OllamaLLM(model="mistral", base_url="http://localhost:11434", temperature=0.0)
        
      
        print("Connecting to Mistral...")
        llm.invoke("Hello")
        print("Mistral Connected!")
        return llm
    except Exception as e:
        print(f"Error connecting to Ollama: {e}. Ensure Ollama is running.")
        raise

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vector_store(embeddings_model):
    CHROMA_DIR = "./chroma_db_store" 
    vectorstore = Chroma(
        collection_name="prescription_records",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_DIR
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_tavily_tool():
    if not os.getenv("TAVILY_API_KEY"):
        print("WARNING: TAVILY_API_KEY not set. Enrichment will be mocked.")
    return TavilySearchResults(max_results=3)


print("Initializing FOSS Stack...")
LLM = get_local_llm()
EMBEDDINGS = get_embeddings_model()
RETRIEVER = get_vector_store(EMBEDDINGS)
TAVILY_TOOL = get_tavily_tool() 

print("FOSS Stack Initialized Successfully.")