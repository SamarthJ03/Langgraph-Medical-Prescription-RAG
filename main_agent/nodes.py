from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Literal, Any
from .state import AgentState 
from .llm_setup import LLM, RETRIEVER 


def _extract_llm_output(raw_output: Any) -> str:
    if hasattr(raw_output, 'content'):
        return raw_output.content
    return str(raw_output)


LANGUAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a multilingual text analyzer. Classify the user's input language.
      Your response MUST be one of the following single-word labels, and nothing else: 'en' (English), 'hi' (Hindi/Devanagari script), or 'hinglish' (Mixed English and Hindi in Roman script).
      Respond ONLY with the single label."""
    ),
    ("human", "{text}")
])
language_chain = LANGUAGE_PROMPT | LLM

def input_manager(state: AgentState) -> AgentState:
    print("---NODE 0: INPUT MANAGER---")
    user_message = state["messages"][-1].content
    
    
    raw_output = language_chain.invoke({"text": user_message})
    language = _extract_llm_output(raw_output).strip().lower()
    
    
    valid_languages = ["en", "hi", "hinglish"]
    if language not in valid_languages:
        language = "en" 

    
    if language in ["hi", "hinglish"]:
        next_step: Literal["translator", "router"] = "translator"
    else:
        next_step: Literal["translator", "router"] = "router"
        
    print(f"Detected Language: {language}, Next Step: {next_step}")
    
  
    return {
        "original_query": user_message,
        "retrieval_query": user_message,
        "input_language": language,
        "final_answer": "",
        "next_step": next_step
    }



TRANSLATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert translator. Translate the user's query into concise, 
      high-quality English, optimized for technical vector search. For example, 
      'BP ki dawa' should become 'Medicine for high blood pressure'.
      Respond ONLY with the single English translation string."""
    ),
    ("human", "{query}")
])
translator_chain = TRANSLATOR_PROMPT | LLM

def query_translator(state: AgentState) -> AgentState:
    print("---NEW NODE: QUERY TRANSLATOR---")
    original_query = state["original_query"]
    
    raw_output = translator_chain.invoke({"query": original_query})
    translated_query = _extract_llm_output(raw_output).strip()
    
    print(f"Original Query: '{original_query}' -> Translated Query: '{translated_query}'")
    return {
        "retrieval_query": translated_query,
        "next_step": "router" 
    }


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query router. Analyze the user's request and decide 
      the complexity: 'complex' (multiple steps, date comparison) or 'simple' (single fact).
      Respond ONLY with the single word 'complex' or 'simple'."""
    ),
    ("human", "{query}")
])
router_chain = ROUTER_PROMPT | LLM 

def initial_router(state: AgentState) -> AgentState:
    print("---NODE 2: INITIAL ROUTER---")
  
    query = state["retrieval_query"] 
    
    raw_output = router_chain.invoke({"query": query})
    decision = _extract_llm_output(raw_output).strip().lower()
    
    if "complex" in decision:
        next_step: Literal["refine_query", "generate_answer"] = "refine_query"
    else:
        next_step: Literal["refine_query", "generate_answer"] = "generate_answer"

    print(f"Router Decision: {next_step}")
    return {"next_step": next_step}


REFINER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert query refinement assistant. Rewrite the original query 
      using the insufficient documents provided to create a NEW, more precise search query 
      optimized for vector similarity search. Respond ONLY with the new query string."""
    ),
    ("human", """Original Query: {original_query}
      --- Insufficient Documents Retrieved ---
      {retrieved_documents}
      What is the best NEW search query?""")
])
refiner_chain = REFINER_PROMPT | LLM 

def query_refiner(state: AgentState) -> AgentState:
    print("---NODE 3: QUERY REFINER---")
   
    original_query = state["original_query"] 
    retrieved_docs_text = "\n\n---\n\n".join(state["retrieved_documents"])
    
    raw_output = refiner_chain.invoke({
        "original_query": original_query, 
        "retrieved_documents": retrieved_docs_text
    })
    
    new_retrieval_query = _extract_llm_output(raw_output).strip()

    print(f"Refined Query: {new_retrieval_query}")
    return {
        "retrieval_query": new_retrieval_query,
        "retrieved_documents": []
    }


def retrieve_prescriptions(state: AgentState) -> AgentState:
    print("---NODE 4: RETRIEVING PRESCRIPTIONS---")

    query = state["retrieval_query"]
    docs = RETRIEVER.invoke(query)
    retrieved_content = [doc.page_content for doc in docs]
    print(f"Retrieved {len(retrieved_content)} documents.")
    return {"retrieved_documents": retrieved_content}


GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert document grader. Your sole task is to determine 
      if the retrieved evidence is fully sufficient to answer the ORIGINAL query. 
      You MUST respond ONLY with one of the following two labels:
      1. 'generate_answer' (if evidence is sufficient)
      2. 'refine_query' (if evidence is insufficient, irrelevant, or incomplete)
      
      DO NOT include any explanation, punctuation, or extra words.
      Respond ONLY with the single label."""
    ),
    ("human", """Original Query: {original_query}
      Retrieved Documents (evidence): {documents_text}
      Decision:""")
])
grader_chain = GRADER_PROMPT | LLM

def grade_documents(state: AgentState) -> AgentState:
    print("---NODE 5: GRADING DOCUMENTS---")
    documents_text = "\n\n---\n\n".join(state["retrieved_documents"])
    grading_input = {
        "original_query": state["original_query"],
        "documents_text": documents_text
    }
    
    raw_output = grader_chain.invoke(grading_input)
    response = _extract_llm_output(raw_output).strip().lower()

    if "generate_answer" in response:
        next_step: Literal["refine_query", "generate_answer"] = "generate_answer"
    else:
        next_step: Literal["refine_query", "generate_answer"] = "refine_query"

    print(f"Grader Decision: {next_step}")
    return {"next_step": next_step}


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a patient medical history assistant. Answer the user's question 
      based ONLY on the provided context. State if the answer is not in the context.
      The user's original query was in the language: {input_language}. 
      Translate your final, synthesized answer back into that language (e.g., Hindi/Hinglish).
      
      Context (Prescription Records):
      {retrieved_documents}
      """
    ),
    ("human", "{original_query}")
])
synthesizer_chain = SYNTHESIS_PROMPT | LLM

def final_synthesizer(state: AgentState) -> AgentState:
    print("---NODE 6: FINAL SYNTHESIZER---")
    raw_output = synthesizer_chain.invoke({
        "original_query": state["original_query"],
        "retrieved_documents": "\n\n---\n\n".join(state["retrieved_documents"]),
        "input_language": state["input_language"]
    })
    
    final_answer = _extract_llm_output(raw_output)
    
    print("Synthesis Complete.")
    return {"final_answer": final_answer}


def route_next_step(state: AgentState) -> str:

    return state["next_step"]