import uuid
from data_ingestion.ingestion_nodes import extract_text_from_file_object, structure_prescription, format_and_index

def mock_upload_file(file_name: str, raw_text_override: str):
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def read(self):
            return self.content.encode('utf-8')

    mock_file = MockUploadedFile(file_name, raw_text_override)
    doc_id = str(uuid.uuid4())
    print(f"\n--- Starting Ingestion for Document ID: {doc_id} ---")
    
    raw_text = extract_text_from_file_object(mock_file)
    print("OCR Output Mocked/Retrieved.")
    
    prescription_record = structure_prescription(raw_text, doc_id)
    print(f"Structured Record for Dr. {prescription_record.doctor_name}")
    
    format_and_index(prescription_record)
    
    print(f"--- Ingestion Complete for {doc_id} ---")

if __name__ == "__main__":
    mock_raw_text_1 = """
    Patient: A.B. Sharma
    Date: 2024-05-10
    Doctor: Dr. Sunil Patel
    Meds: Lisinopril 20mg - Take 1 tablet daily. Qty: 30
    Diagnosis: Essential Hypertension
    """
    
    mock_raw_text_2 = """
    Patient: A.B. Sharma
    Date: 2023-01-15
    Doctor: Dr. Priya Sharma
    Meds: Metformin 500mg - Twice a day with food. Qty: 90
    Diagnosis: Type II Diabetes Mellitus
    """
    
    mock_upload_file("BP_mock.jpg", mock_raw_text_1)
    mock_upload_file("Diabetes_mock.jpg", mock_raw_text_2)
    
    print("\nChromaDB is now populated and ready for the LangGraph agent!")