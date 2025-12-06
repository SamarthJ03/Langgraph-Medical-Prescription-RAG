import pytesseract
from PIL import Image
from typing import List
import uuid

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from main_agent.schemas import PrescriptionRecord, MedicineEnrichment, MedicationDetails
from main_agent.llm_setup import LLM, EMBEDDINGS, RETRIEVER, TAVILY_TOOL, get_vector_store

def extract_text_from_file_object(file_object) -> str:
    print("---INGESTION: Running OCR Mock---")
    try:
        if file_object.name == "BP_mock.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2024-05-10
            Doctor: Dr. Sunil Patel
            Meds: Lisinopril 20mg - Take 1 tablet daily. Qty: 30
            Diagnosis: Essential Hypertension
            """
        elif file_object.name == "Diabetes_mock.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2023-01-15
            Doctor: Dr. Priya Sharma
            Meds: Metformin 500mg - Twice a day with food. Qty: 90
            Diagnosis: Type II Diabetes Mellitus
            """
        elif file_object.name == "HighBP_20mg.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2024-05-10
            Doctor: Dr. Sunil Patel
            Meds: 
            1. Lisinopril 20mg - Take 1 tablet daily. Qty: 30
            2. Amlodipine 5mg - Take 1 tablet at night. Qty: 60
            Diagnosis: Essential Hypertension
            """
        elif file_object.name == "Diabetes.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2023-01-15
            Doctor: Dr. Priya Sharma
            Meds: 
            1. Metformin 500mg - Twice a day with food. Qty: 90
            Diagnosis: Type II Diabetes Mellitus
            """
        elif file_object.name == "Cholesterol.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2024-10-01
            Doctor: Dr. V.K. Singh
            Meds: 
            1. Atorvastatin 40mg - Take 1 tablet every night. Qty: 30
            2. Ezetimibe 10mg - Take 1 tablet in the morning. Qty: 30
            Diagnosis: Hyperlipidemia (High Cholesterol)
            """
        elif file_object.name == "Pediatric.jpg":
            raw_text = """
            Patient: A.B. Sharma (Child)
            Date: 2023-06-25
            Doctor: Dr. Anjali Rao
            Meds: 
            1. Amoxicillin 250mg/5ml suspension - Take 5ml three times a day for 7 days. Qty: 1
            Diagnosis: Acute Otitis Media (Ear Infection)
            """
        elif file_object.name == "LowBP_10mg.jpg":
            raw_text = """
            Patient: A.B. Sharma
            Date: 2023-02-01
            Doctor: Dr. Sunil Patel
            Meds: 
            1. Lisinopril 10mg - Take 1 tablet daily. Qty: 30
            2. Aspirin 81mg - Chewable, daily. Qty: 90
            Diagnosis: Mild Hypertension
            """
        else:
             raw_text = "Doctor: Dr. Test Date: 2025-01-01 Meds: Aspirin 100mg Qty: 10"

        return raw_text.strip()
    except Exception as e:
        return f"OCR Error: Failed to process file. {e}"



formatter_parser = PydanticOutputParser(pydantic_object=PrescriptionRecord)

FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert medical data extractor. Your task is to extract all prescription details 
     from the raw text and return a JSON object that matches the provided schema.
     
     {format_instructions}
     
     IMPORTANT: Return ONLY the JSON. Do not add any explanation, markdown formatting, or code blocks.
     Infer the disease_condition if it is explicitly stated under Diagnosis."""
    ),
    ("human", "Raw Prescription Text: {raw_text}\nDocument ID: {doc_id}")
])

formatter_chain = FORMATTER_PROMPT | LLM | formatter_parser

def structure_prescription(raw_text: str, doc_id: str) -> PrescriptionRecord:
    print("---INGESTION: Structuring Data---")
    record = formatter_chain.invoke({
        "raw_text": raw_text, 
        "doc_id": doc_id,
        "format_instructions": formatter_parser.get_format_instructions()
    })
    return record

enrichment_parser = PydanticOutputParser(pydantic_object=MedicineEnrichment)

ENRICHMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an expert medical information synthesizer. Analyze the web search results 
     and extract the required information into the Pydantic schema.
     
     {format_instructions}
     
     IMPORTANT: Return ONLY the JSON. Do not add any explanation or markdown.
     Identify the top 5 primary uses for the drug."""
    ),
    ("human", 
     """Medication Name: {medicine_name}
     --- Web Search Results ---
     {search_results}"""
    )
])

enrichment_chain = ENRICHMENT_PROMPT | LLM | enrichment_parser

MEDICINE_CACHE = {} 

def enrich_medicine(med_name: str) -> MedicineEnrichment:
    if med_name in MEDICINE_CACHE:
        return MEDICINE_CACHE[med_name]
        
    print(f"Searching web for: {med_name}")
    search_query = f"primary uses and drug class of {med_name} in medicine"
    search_results = TAVILY_TOOL.invoke(search_query)
    
    response = enrichment_chain.invoke({
        "medicine_name": med_name,
        "search_results": search_results,
        "format_instructions": enrichment_parser.get_format_instructions()
    })
    
    MEDICINE_CACHE[med_name] = response
    return response



def format_and_index(record: PrescriptionRecord):
    print("---INGESTION: Indexing Data---")
    
    all_uses = set()
    for med in record.medications:
        enriched_data = enrich_medicine(med.medication_name) 
        all_uses.update(enriched_data.primary_uses)
    
    med_list = "\n".join([f"{i+1}. {m.medication_name} ({m.dosage}, {m.quantity_dispensed}) - Instructions: {m.instructions}" 
                          for i, m in enumerate(record.medications)])
    
    content = f"""
    --- Prescription Record ---
    Doctor: {record.doctor_name}
    Date: {record.prescription_date}
    Primary Condition: {record.disease_condition or 'Unknown/Inferred'}
    Medications Prescribed:
    {med_list}
    
    Related Conditions (from enrichment): {", ".join(list(all_uses))}
    ---
    """
    
    metadata = {
        "doctor": record.doctor_name,
        "date": record.prescription_date.isoformat(),
        "document_id": record.document_id,
        "conditions": ", ".join(list(all_uses))
    }
    
    document = Document(page_content=content.strip(), metadata=metadata)
    
    vectorstore = get_vector_store(EMBEDDINGS)
    vectorstore.add_documents([document])
    
    print(f"Indexed document {record.document_id} successfully.")