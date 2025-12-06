import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from main_agent.graph_builder import AGENT_APP
from data_ingestion.ingestion_nodes import extract_text_from_file_object, structure_prescription, format_and_index

LLM_MODEL_NAME = "Ollama/Mistral" 



if "session_id" not in st.session_state:
    st.session_state.session_id = "default_user_session_" + str(uuid.uuid4())
    st.session_state.messages = [AIMessage(content="Hello! I can help you retrieve information from your medical records. What is your query?")] 

def handle_ingestion(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a file first.")
        return

    doc_id = str(uuid.uuid4())
    st.markdown(f"### 📥 Ingestion Pipeline Started (ID: `{doc_id}`)")

    try:
        with st.spinner("Step 1: Running Document Scan (OCR Mock)..."):
       
            raw_text = extract_text_from_file_object(uploaded_file)
            st.success("✅ Document Scanned/Text Extracted.")
            with st.expander("Show Raw OCR Text"):
                st.code(raw_text)

        with st.spinner("Step 2: Structuring Data (LLM + Pydantic)..."):
           
            prescription_record = structure_prescription(raw_text, doc_id)
            st.success("✅ Data Structured and Validated.")
            with st.expander("Show Structured Pydantic Output"):
                st.json(prescription_record.model_dump(), expanded=False)

        with st.spinner("Step 3: Enriching and Indexing (ChromaDB)..."):
            
            format_and_index(prescription_record)
            st.success(f"✅ Document Indexed in ChromaDB!")
            st.toast("Document Indexed!", icon="🎉")

    except Exception as e:
        st.error(f"❌ Ingestion Failed: {e}")
        st.toast("Ingestion Failed!", icon="🚨")



def run_agent(query: str):
    
    config = {
        "configurable": {"thread_id": st.session_state.session_id},
        "recursion_limit": 5 
    }
    
    initial_messages = st.session_state.messages + [HumanMessage(content=query)]
    initial_state = {
        "messages": initial_messages,
        "retrieved_documents": [],
        "next_step": "router" 
    }

    final_output_message = "Agent finished, but no final answer was captured."

    
    st.info("Agent is thinking... Trace below:")
    trace_placeholder = st.empty()
    trace_markdown = ""
    

    for event in AGENT_APP.stream(initial_state, config=config):
        
       
        for node_name, output in event.items():
            
            
            if node_name != "END":
                 trace_markdown += f"**➡️ Running Node:** `{node_name}`\n"
                 trace_placeholder.markdown(trace_markdown)
            
            
            if isinstance(output, dict) and output.get("final_answer"):
                final_output_message = output["final_answer"]
                trace_markdown += "✅ **Answer Synthesized**\n"
                trace_placeholder.markdown(trace_markdown)
                return final_output_message

    
    return final_output_message


st.title("👨‍⚕️ FOSS Medical Agent Demo")
st.caption(f"LangGraph RAG Agent with FOSS Stack ({LLM_MODEL_NAME}, ChromaDB)")

st.divider()


with st.sidebar:
    st.header("Upload Prescriptions")
    st.info("Upload mock files to populate the knowledge base (ChromaDB).")
    
    uploaded_file = st.file_uploader(
        "Upload a mock Prescription Image/PDF", 
        type=['png', 'jpg', 'jpeg', 'pdf'],
        key="uploader"
    )

    if st.button("Index Document", disabled=uploaded_file is None):
        handle_ingestion(uploaded_file)

st.header("Ask a Question")


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if query := st.chat_input("Ask about your prescription history..."):
 
    with st.chat_message("user"):
        st.write(query)
    

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        

        with st.spinner("Processing query..."):
            final_response = run_agent(query)
        
        message_placeholder.markdown(final_response)
        
  
    st.session_state.messages.append(HumanMessage(content=query))
    st.session_state.messages.append(AIMessage(content=final_response))

    st.rerun()