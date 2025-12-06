from typing import Literal
from langgraph.graph import StateGraph, END
from .state import AgentState 
from .nodes import (
    input_manager,
    query_translator, 
    initial_router,
    retrieve_prescriptions,
    grade_documents,
    query_refiner,
    final_synthesizer,
    route_next_step
)


workflow = StateGraph(AgentState)


workflow.add_node("input_manager", input_manager)
workflow.add_node("query_translator", query_translator) 
workflow.add_node("initial_router", initial_router)
workflow.add_node("retrieve_prescriptions", retrieve_prescriptions)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("query_refiner", query_refiner)
workflow.add_node("final_synthesizer", final_synthesizer)


workflow.set_entry_point("input_manager")


workflow.add_conditional_edges(
    "input_manager", 
    route_next_step, 
    {
        "translator": "query_translator",
        "router": "initial_router"
    }
)


workflow.add_edge("query_translator", "initial_router")


workflow.add_conditional_edges(
    "initial_router", 
    route_next_step, 
    {
        "refine_query": "query_refiner",    
        "generate_answer": "retrieve_prescriptions" 
    }
)

workflow.add_edge("query_refiner", "retrieve_prescriptions") 

workflow.add_conditional_edges(
    "grade_documents", 
    route_next_step, 
    {
        "refine_query": "query_refiner",     
        "generate_answer": "final_synthesizer" 
    }
)


workflow.add_edge("retrieve_prescriptions", "grade_documents")
workflow.add_edge("final_synthesizer", END)


AGENT_APP = workflow.compile()
print("LangGraph Agent Compiled Successfully with Translation Support.")