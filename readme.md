# 🏆 Multilingual Medical RAG Agent  
### *Bridging Language Gaps in Healthcare Data with LangGraph*

Millions of people especially in multilingual regions like India struggle to understand their own medical records. These documents are often unstructured (images, scans, handwritten notes), and user queries frequently come in **Hindi**, **Hinglish**, or other native languages that standard enterprise RAG systems fail to handle.

This project demonstrates a **stateful, multilingual, self-correcting RAG agent** built entirely on **FOSS tools**, powered by **LangGraph**.

---

## 🚀 Core Problem

- Medical data is locked inside unstructured PDFs/images.  
- Patients commonly ask questions in mixed languages.  
- Typical RAG solutions:  
  - Fail on Hinglish/Hindi  
  - Retrieve irrelevant documents  
  - Have no self-correction  
  - Depend on closed-source models  

---

## ✨ Our Solution: The LangGraph RAG Cycle

We built a **7-node LangGraph agent** with multilingual translation, document grading, refined retrieval, and synthesis — all running locally.

### Key Innovations

#### 🔤 1. Multilingual Query Handling
- Accepts **English**, **Hindi**, **Hinglish**  
- Automatically detects non-English  
- Translates → retrieves in English → translates final answer back  

#### 🔁 2. Iterative RAG Loop (Self-Grading)

Refiner → Retriever → Grader → (repeat if needed) → Synthesizer


If retrieved documents are low-quality, the **Grader node** triggers automatic refinement until relevance improves.

#### 🧩 3. Strict State Management  
Powered by `TypedDict` + `Annotated[List]`  
Predictable state updates across:

- Input Manager  
- Translator  
- Router  
- Refiner  
- Retriever  
- Grader  
- Synthesizer  

#### 🆓 4. Fully Open-Source Stack  
Runs completely locally → zero licensing cost.

---

## 🧠 LangGraph Execution Flow

User Input
→ Input Manager
→ Translator (if needed)
→ Router
→ Refiner
→ Retriever
→ Grader (repeat Refiner→Retriever→Grader if needed)
→ Synthesizer
→ Final Answer (translated if needed)



---

## 🔧 Tech Stack

| Component | Technology | Role |
|----------|------------|------|
| Orchestration | LangGraph | Stateful agent logic & RAG loop |
| LLM | Ollama + Mistral | Translation, reasoning, synthesis |
| Vector DB | ChromaDB | Local embedding store |
| UI | Streamlit | Chat + ingestion |
| OCR (Demo) | pytesseract, Pillow | Mock OCR pipeline |

---

## ⚠️ Demo Limitations

### 1. OCR Disclaimer  
Documents are **NOT truly OCR’d** — instead they are processed via **mock extraction** for demonstration only.

### 2. Small Local LLM Constraints  
Lightweight FOSS models may produce imperfect:

- Hindi/Hinglish translations  
- Medical interpretations  

---

## 🚀 Quick Start (1-Minute Setup)

### 1. Requirements  
- Python 3.9+  
- **Ollama installed & running**  

---

### 2. Install Dependencies

```bash
git clone [YOUR-REPO-URL]
cd [PROJECT-FOLDER]
pip install -r requirements.txt
```
3. Populate the Knowledge Base
Runs mock OCR + embedding pipeline and fills ChromaDB.

```bash

python run_ingestion.py
```
To add more mock prescription files:

Drop files like new_record.pdf or rx1.jpg into the ingestion folder

Pipeline processes them (mock OCR → embeddings)

4. Launch the App
```bash

streamlit run streamlit_app.py
```
🧪 How to Use the Demo
1. Upload a Document
Sidebar → Upload Document
Triggers mock ingestion.

2. Ask an English Query
Example:
“What medicine did I take for blood pressure?”

3. Ask a Multilingual Query
Example:
“Mujhe sardi ke liye kya dawa di gayi thi?”
(What medicine was I given for a cold?)

The agent will automatically:
Translate

Retrieve

Run the iterative RAG loop

Synthesize

Translate back

🔮 Roadmap / Future Enhancements
🚧 1. Real OCR Integration
pytesseract

Indic OCR models

Handwriting recognition

🚧 2. Specialized Multilingual LLM
Fine-tuned FOSS models tailored for:

Hindi/Hinglish linguistic nuance

Medical terminology

🚧 3. Voice Interface
Speech-to-text input

Text-to-speech healthcare explanations

🚧 4. Production Security
Encrypted vector DB

Secure API mode

Edge deployment support

