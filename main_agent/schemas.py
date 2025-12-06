from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class MedicationDetails(BaseModel):
    medication_name: str = Field(..., description="The brand or generic name of the drug.")
    dosage: str = Field(..., description="The strength and unit (e.g., '50 mg').")
    quantity_dispensed: str = Field(..., description="The total quantity given (e.g., '30 tablets').")
    instructions: Optional[str] = Field(None, description="The instructions for consumption (e.g., 'Take once daily').")

class PrescriptionRecord(BaseModel):
    prescription_date: date = Field(..., description="The date the prescription was issued (YYYY-MM-DD format).")
    doctor_name: str = Field(..., description="The full name of the prescribing physician.")
    disease_condition: Optional[str] = Field(None, description="The specific condition or reason for the prescription (Inferred by LLM).")
    medications: List[MedicationDetails] = Field(..., description="A list of all individual medication details prescribed.")
    document_id: str = Field(..., description="A unique identifier assigned to the original uploaded document.")

class MedicineEnrichment(BaseModel):
    medication_name: str = Field(..., description="The name of the drug searched.")
    drug_class: Optional[str] = Field(None, description="The pharmacological class, e.g., 'Antihypertensive'.")
    primary_uses: List[str] = Field(..., description="A list of the top 5 diseases or conditions this medicine is used to treat.")
    summary: str = Field(..., description="A one-sentence summary of the drug's mechanism or purpose.")