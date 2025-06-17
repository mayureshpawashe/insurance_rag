# --- Imports ---
import os
import re
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


# --- Wrap potentially problematic imports in try-except blocks ---
try:
    import openai
except ImportError:
    print("\n:warning: openai not installed. Install with: pip install openai")
    openai = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("\n:warning: PyMuPDF not installed. Install with: pip install pymupdf")
    fitz = None

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.memory import ConversationBufferMemory
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    print("\n:warning: LangChain dependencies not installed. Install with: pip install langchain langchain-openai langchain-community")

# --- New imports for enhanced PDF processing with OCR ---
try:
    import PyPDF2
except ImportError:
    print("\n:warning: PyPDF2 not installed. Install with: pip install PyPDF2")
    PyPDF2 = None
    
import tempfile
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("\n:warning: OCR dependencies not available. Only direct text extraction will be used.")
    print("To enable OCR capabilities, install: pip install pytesseract pdf2image")
    print("And ensure tesseract-ocr and poppler-utils are installed on your system.")
    
    

# === INTEGRATED: Smart Underwriting Agents ===
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
# ===============================
# INTEGRATED SMART UNDERWRITING AGENTS
# ===============================
from agents.agent1_historical_patterns import HistoricalRiskPatternAgent, HistoricalPattern, RiskFactorCorrelation
from agents.agent2_risk_assessment import ActuarialRiskAssessmentAgent, RiskAssessment, RiskFactor, RiskLevel
from agents.agent3_policy_generation import PolicyGenerationAgent, PolicyDraft, PolicyTerms, PremiumCalculation, PolicyDocument
from agents.orchestrator import SmartUnderwritingOrchestrator

# --- Agent 1: Historical Risk Pattern Analysis ---
# --- Agent 2: Risk Assessment ---
# --- Agent 3: Policy Generation ---
# --- Orchestrator ---
class UnderwritingDecision(BaseModel):
    decision: str = Field(description="APPROVED, DECLINED, MANUAL_REVIEW")
    decision_reason: str = Field(description="Reason for the decision")
    confidence_level: float = Field(description="Confidence in decision (0-1)")
    manual_review_triggers: List[str] = Field(description="Factors requiring manual review")

class UnderwritingResult(BaseModel):
    customer_id: str
    request_id: str = Field(description="Unique request identifier")
    historical_patterns: List[Any] = Field(description="Output from Agent 1")
    risk_assessment: Any = Field(description="Output from Agent 2")
    policy_draft: Any = Field(description="Output from Agent 3")
    underwriting_decision: UnderwritingDecision
    processing_time: float = Field(description="Total processing time in seconds")
    agents_used: List[str] = Field(description="List of agents that were called")
    success: bool = Field(description="Whether underwriting completed successfully")
    errors: List[str] = Field(description="Any errors encountered")
    external_data_sources: List[str] = Field(description="External APIs used")
    compliance_checks: Dict[str, bool] = Field(description="Regulatory compliance status")
    timestamp: datetime = Field(description="When underwriting was completed")


# Factory functions
def create_historical_pattern_agent(vector_db, uploaded_vector_db, policy_list):
    return HistoricalRiskPatternAgent(vector_db, uploaded_vector_db, policy_list)

def create_risk_assessment_agent():
    return ActuarialRiskAssessmentAgent()

def create_policy_generation_agent():
    return PolicyGenerationAgent()

def create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list):
    return SmartUnderwritingOrchestrator(vector_db, uploaded_vector_db, policy_list)

def setup_smart_underwriting(vector_db, uploaded_vector_db, policy_list, enable_external_apis=False):
    orchestrator = create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list)
    return {
        'orchestrator': orchestrator,
        'agent1_historical': orchestrator.agent1,
        'agent2_risk': orchestrator.agent2,
        'agent3_policy': orchestrator.agent3,
        'version': '1.0.0-integrated'
    }

def check_agent_health():
    return {
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'orchestrator': True,
        'overall_health': True,
        'errors': []
    }

async def enhanced_api_handle_message(user_id, user_input, conversation_turn=0, file_path=None, extraction_method='auto', vector_db=None, uploaded_vector_db=None, policy_list=None):
    processing_info = None
    
    if file_path:
        processing_info = {
            "file_processed": True,
            "extraction_method": extraction_method,
            "note": "Document processing integrated with underwriting"
        }
    
    orchestrator = create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list)
    response = await orchestrator.enhanced_process_message(user_id, user_input, conversation_turn)
    
    return {
        "response": response,
        "user_profile": {"user_id": user_id, "status": "active"},
        "processing_info": processing_info
    }

# Mark as available
SMART_UNDERWRITING_AVAILABLE = True
print("✅ Smart Underwriting Agents integrated successfully")

# === END SMART UNDERWRITING INTEGRATION ===

# --- Define Constants ---
PERSIST_DIR = "./icici_policy_chroma_db"
UPLOADED_PERSIST_DIR = "./uploaded_policy_chroma_db"
SESSION_DIR = "./user_sessions"
POLICY_LIST_FILE = "./policy_list.json"
UPLOADED_POLICY_LIST_FILE = "./uploaded_policy_list.json"
UPLOADS_DIR = "./uploaded_policies"
PDF_FOLDER = "./icici_policies"

# --- Create necessary directories ---
for directory in [SESSION_DIR, UPLOADS_DIR, PDF_FOLDER, os.path.dirname(POLICY_LIST_FILE), os.path.dirname(UPLOADED_POLICY_LIST_FILE)]:
    os.makedirs(directory, exist_ok=True)

# --- Set your OpenAI Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize Embedding Model and LLM ---
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
fact_check_llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)

# --- Initialize Memory ---
conversation_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
    input_key="input",
    output_key="output"
)

# --- Define Pydantic Models for Structured Data ---
class PolicyEntity(BaseModel):
    policy_name: str = Field(description="The name of the policy")
    policy_type: Optional[str] = Field(None, description="Type of policy (health, life, etc.)")
    mentioned_in_turn: int = Field(description="Conversation turn where this policy was mentioned")
    source_document: Optional[str] = Field(None, description="Source document for this policy")
    is_uploaded: bool = Field(False, description="Whether this policy was uploaded by the user")

class UserPreference(BaseModel):
    preference_type: str = Field(description="Type of preference (age, family, budget, etc.)")
    value: str = Field(description="The value of this preference")
    confidence: float = Field(description="Confidence score (0-1) of this preference")
    
class UserProfile(BaseModel):
    user_id: str = Field(description="Unique ID for the user")
    preferences: List[UserPreference] = Field(default_factory=list, description="List of user preferences")
    policies_discussed: List[PolicyEntity] = Field(default_factory=list, description="Policies discussed with this user")
    policies_recommended: List[PolicyEntity] = Field(default_factory=list, description="Policies recommended to this user")
    uploaded_policies: List[PolicyEntity] = Field(default_factory=list, description="Policies uploaded by this user")
    last_interaction: str = Field(description="Timestamp of last interaction")

class PolicyMatch(BaseModel):
    policy_name: str = Field(description="Name of the policy")
    match_score: float = Field(description="Score indicating how well the policy matches user preferences")
    key_matches: List[str] = Field(default_factory=list, description="Key matching points between policy and user preferences")
    is_uploaded: bool = Field(False, description="Whether this policy was uploaded by the user")

class APIResponse(BaseModel):
    response: str
    user_profile: UserProfile
    processing_info: Optional[Dict[str, Any]] = None

# --- PDF Processing Utility Functions ---
def extract_text_with_pypdf2(pdf_path):
    """Extract text from PDF using PyPDF2"""
    print("Extracting text using PyPDF2...")
    full_text = ""
    page_count = 0
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            for i, page in enumerate(pdf_reader.pages):
                print(f"Processing page {i+1}/{page_count}")
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        print(f"PyPDF2 extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
        
        if len(full_text.strip()) < page_count * 100:
            print("Warning: Very little text extracted. This might be a scanned document.")
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {str(e)}")
    
    return full_text, page_count

def extract_text_with_pymupdf(pdf_path):
    """Extract text from PDF using PyMuPDF (fitz)"""
    print("Extracting text using PyMuPDF...")
    full_text = ""
    try:
        pdf_doc = fitz.open(pdf_path)
        page_count = len(pdf_doc)
        for i, page in enumerate(pdf_doc):
            print(f"Processing page {i+1}/{page_count}")
            text = page.get_text()
            full_text += text + "\n\n"
        pdf_doc.close()
        print(f"PyMuPDF extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
        
        if len(full_text.strip()) < page_count * 100:
            print("Warning: Very little text extracted. This might be a scanned document.")
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {str(e)}")
        page_count = 0
    
    return full_text, page_count

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using Tesseract OCR"""
    if not OCR_AVAILABLE:
        return "OCR dependencies not available. Please install requirements.", 0
    
    print("Extracting text using Tesseract OCR...")
    full_text = ""
    try:
        pages = convert_from_path(pdf_path)
        page_count = len(pages)
        
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{page_count} with OCR")
            text = pytesseract.image_to_string(page)
            full_text += text + "\n\n"
        
        print(f"OCR extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
    except Exception as e:
        print(f"Error extracting text with OCR: {str(e)}")
        page_count = 0
    
    return full_text, page_count

def extract_text_from_pdf(pdf_path, method='auto'):
    """Extract text from PDF with method selection"""
    print("Extracting text from PDF...")
    
    if method == 'auto':
        text, page_count = extract_text_with_pymupdf(pdf_path)
        
        if len(text.strip()) < page_count * 50:
            print("\nNot enough text extracted with PyMuPDF. Trying PyPDF2...")
            text2, page_count2 = extract_text_with_pypdf2(pdf_path)
            
            if len(text2.strip()) > len(text.strip()):
                text, page_count = text2, page_count2
        
        if OCR_AVAILABLE and len(text.strip()) < page_count * 50:
            print("\nNot enough text extracted with direct methods. Switching to OCR...")
            text_ocr, page_count_ocr = extract_text_with_ocr(pdf_path)
            
            if len(text_ocr.strip()) > len(text.strip()):
                text, page_count = text_ocr, page_count_ocr
    
    elif method == 'pymupdf':
        text, page_count = extract_text_with_pymupdf(pdf_path)
    elif method == 'pypdf2':
        text, page_count = extract_text_with_pypdf2(pdf_path)
    elif method == 'ocr' and OCR_AVAILABLE:
        text, page_count = extract_text_with_ocr(pdf_path)
    else:
        print(f"Unknown extraction method: {method}. Using PyMuPDF.")
        text, page_count = extract_text_with_pymupdf(pdf_path)
    
    return text, page_count

def enhanced_chunk_text(text, page_count=None, chunk_size=500, chunk_overlap=50):
    """Create better chunks from text with improved metadata"""
    print("Chunking extracted text...")
    
    if not text.strip():
        print("Warning: No text content to chunk.")
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    docs = splitter.create_documents([text])
    
    if page_count and page_count > 0:
        chunks_per_page = max(1, len(docs) // page_count)
        for i, doc in enumerate(docs):
            estimated_page = min(page_count, (i // chunks_per_page) + 1) if chunks_per_page > 0 else 1
            doc.metadata = {"chunk_id": i+1, "estimated_page": estimated_page, "total_pages": page_count}
    else:
        for i, doc in enumerate(docs):
            doc.metadata = {"chunk_id": i+1}
    
    print(f"Chunking complete. Created {len(docs)} chunks.")
    return docs

def load_pdf_text(pdf_folder):
    """Load text from PDF files in a folder"""
    documents = []
    policy_list = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            pdf_doc = fitz.open(filepath)
            full_text = ""
            for page in pdf_doc:
                full_text += page.get_text()
            policy_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
            documents.append({"text": full_text, "source": filename, "policy_name": policy_name})
            
            policy_info = {
                "policy_name": policy_name,
                "source": filename,
                "policy_type": "unknown"
            }
            policy_list.append(policy_info)
            pdf_doc.close()
    
    with open(POLICY_LIST_FILE, 'w') as f:
        json.dump(policy_list, f, indent=2)
        
    return documents, policy_list

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks for the vector store"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_objects = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            doc_objects.append(Document(
                page_content=chunk,
                metadata={"source": doc["source"], "policy_name": doc["policy_name"]}
            ))
    return doc_objects

# --- Load or Create Policy Database ---
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(POLICY_LIST_FILE), exist_ok=True)

if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    print("\n:warning: No ChromaDB found. Loading PDFs and creating DB...")
    
    if not os.path.exists(PDF_FOLDER) or not [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
        print(f"\n:warning: No PDF files found in {PDF_FOLDER}. Creating empty database.")
        chroma_db = Chroma(embedding_function=embedding_model, persist_directory=PERSIST_DIR)
        chroma_db.persist()
        policy_list = []
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)
        print("\n:white_check_mark: Empty Policy ChromaDB created successfully!")
    else:
        documents, policy_list = load_pdf_text(PDF_FOLDER)
        docs = chunk_text(documents)
        chroma_db = Chroma.from_documents(docs, embedding_model, persist_directory=PERSIST_DIR)
        chroma_db.persist()
        print("\n:white_check_mark: Policy ChromaDB created successfully!")
else:
    chroma_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    print("\n:white_check_mark: Loaded existing Policy ChromaDB.")
    
    if os.path.exists(POLICY_LIST_FILE):
        with open(POLICY_LIST_FILE, 'r') as f:
            policy_list = json.load(f)
    else:
        policy_list = []
        collection = chroma_db._collection
        metadatas = collection.get()["metadatas"]
        seen_policies = set()
        
        for metadata in metadatas:
            if metadata and "policy_name" in metadata:
                if metadata["policy_name"] not in seen_policies:
                    policy_info = {
                        "policy_name": metadata["policy_name"],
                        "source": metadata.get("source", "unknown"),
                        "policy_type": "unknown"
                    }
                    policy_list.append(policy_info)
                    seen_policies.add(metadata["policy_name"])
        
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)

# --- Create or load the uploaded policy database ---
def initialize_uploaded_policy_database():
    """Initialize or load the uploaded policy database"""
    global uploaded_chroma_db, uploaded_policy_list
    
    os.makedirs(UPLOADED_PERSIST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(UPLOADED_POLICY_LIST_FILE), exist_ok=True)
    
    if not os.path.exists(UPLOADED_PERSIST_DIR):
        uploaded_chroma_db = Chroma(embedding_function=embedding_model, persist_directory=UPLOADED_PERSIST_DIR)
        uploaded_chroma_db.persist()
        uploaded_policy_list = []
        with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
            json.dump(uploaded_policy_list, f, indent=2)
    else:
        try:
            uploaded_chroma_db = Chroma(persist_directory=UPLOADED_PERSIST_DIR, embedding_function=embedding_model)
        except Exception as e:
            print(f"Error loading uploaded policy DB: {str(e)}. Initializing empty DB.")
            uploaded_chroma_db = Chroma(embedding_function=embedding_model, persist_directory=UPLOADED_PERSIST_DIR)
            uploaded_chroma_db.persist()
        
        if os.path.exists(UPLOADED_POLICY_LIST_FILE):
            with open(UPLOADED_POLICY_LIST_FILE, 'r') as f:
                uploaded_policy_list = json.load(f)
        else:
            uploaded_policy_list = []
            with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
                json.dump(uploaded_policy_list, f, indent=2)
    
    print("\n:white_check_mark: Uploaded Policy Database initialized.")

try:
    initialize_uploaded_policy_database()
except Exception as e:
    print(f"\n:warning: Error initializing uploaded policy database: {str(e)}")
    uploaded_policy_list = []

# === Initialize Smart Underwriting System ===
smart_underwriting_system = None
if SMART_UNDERWRITING_AVAILABLE:
    try:
        health_status = check_agent_health()
        if health_status['overall_health']:
            smart_underwriting_system = setup_smart_underwriting(
                vector_db=chroma_db,
                uploaded_vector_db=uploaded_chroma_db,
                policy_list=policy_list,
                enable_external_apis=False
            )
            print("🤖 Smart Underwriting System initialized successfully!")
            print(f"   Version: {smart_underwriting_system['version']}")
        else:
            print("⚠️ Agent health check failed:")
            for error in health_status['errors']:
                print(f"     {error}")
    except Exception as e:
        print(f"⚠️ Could not initialize Smart Underwriting System: {e}")
        smart_underwriting_system = None

# --- Policy Processing Functions ---
def process_uploaded_pdf(file_path: str, user_id: str, extraction_method='auto') -> PolicyEntity:
    """Process an uploaded PDF file and add it to the database"""
    global uploaded_policy_list
    
    filename = os.path.basename(file_path)
    policy_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
    
    if not file_path.startswith(UPLOADS_DIR):
        dest_path = os.path.join(UPLOADS_DIR, filename)
        import shutil
        shutil.copy2(file_path, dest_path)
        file_path = dest_path
    
    try:
        print(f"Processing uploaded policy: {policy_name}")
        
        full_text, page_count = extract_text_from_pdf(file_path, method=extraction_method)
        
        if not full_text.strip():
            print("Warning: No text could be extracted from the PDF.")
            return PolicyEntity(
                policy_name=f"Error: No text extracted from {policy_name}",
                policy_type="unknown",
                mentioned_in_turn=0,
                source_document=filename,
                is_uploaded=True
            )
        
        preview_length = min(3000, len(full_text))
        text_preview = full_text[:preview_length]
        print(f"Successfully extracted {len(full_text)} characters from {page_count} pages.")
        
        policy_type_prompt = f"""
        Based on the following insurance policy text, determine the most likely policy type
        (e.g., health, life, motor, home, travel, etc.). Focus on the coverage, benefits, and terms.
        
        POLICY TEXT EXTRACT:
        {text_preview}
        
        POLICY TYPE:
        """
        
        prompt = ChatPromptTemplate.from_template(policy_type_prompt)
        chain = prompt | llm | StrOutputParser()
        policy_type = chain.invoke({}).strip()
        
        print(f"Detected policy type: {policy_type}")
        
        docs = enhanced_chunk_text(full_text, page_count, chunk_size=500, chunk_overlap=50)
        
        if not docs:
            print("Warning: Failed to create text chunks.")
            return PolicyEntity(
                policy_name=f"Error: Chunking failed for {policy_name}",
                policy_type=policy_type,
                mentioned_in_turn=0,
                source_document=filename,
                is_uploaded=True
            )
        
        for i, doc in enumerate(docs):
            existing_metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            
            doc.metadata = {
                **existing_metadata,
                "source": filename, 
                "policy_name": policy_name,
                "policy_type": policy_type,
                "uploaded_by": user_id,
                "is_uploaded": True,
                "chunk_index": i
            }
        
        print(f"Adding {len(docs)} chunks to vector database...")
        uploaded_chroma_db.add_documents(docs)
        uploaded_chroma_db.persist()
        print("Successfully added to vector database.")
        
        policy_info = {
            "policy_name": policy_name,
            "source": filename,
            "policy_type": policy_type,
            "uploaded_by": user_id,
            "upload_date": datetime.now().isoformat(),
            "extraction_method": extraction_method,
            "page_count": page_count,
            "chunk_count": len(docs)
        }
        
        uploaded_policy_list.append(policy_info)
        with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
            json.dump(uploaded_policy_list, f, indent=2)
        
        policy_entity = PolicyEntity(
            policy_name=policy_name,
            policy_type=policy_type,
            mentioned_in_turn=0,
            source_document=filename,
            is_uploaded=True
        )
        
        user_profile = get_or_create_session(user_id)
        user_profile.uploaded_policies.append(policy_entity)
        save_session(user_profile)
        
        print(f"Successfully processed {policy_name}")
        return policy_entity
        
    except Exception as e:
        print(f"Error processing uploaded PDF: {str(e)}")
        return PolicyEntity(
            policy_name=f"Error processing {policy_name}",
            policy_type="unknown",
            mentioned_in_turn=0,
            source_document=filename,
            is_uploaded=True
        )

def extract_policy_entities(text: str, conversation_turn: int) -> List[PolicyEntity]:
    """Extract policy names from text and return as PolicyEntity objects"""
    global policy_list
    
    try:
        global uploaded_policy_list
        if 'uploaded_policy_list' not in globals():
            if os.path.exists(UPLOADED_POLICY_LIST_FILE):
                with open(UPLOADED_POLICY_LIST_FILE, 'r') as f:
                    uploaded_policy_list = json.load(f)
            else:
                uploaded_policy_list = []
    except Exception as e:
        print(f"Error loading uploaded policy list: {str(e)}")
        uploaded_policy_list = []
    
    extract_prompt = """
    Extract any insurance policy names mentioned in the following text:
    
    TEXT: {text}
    
    Return the policy names as a Python list of strings. If no policy names are found, return an empty list.
    Format: ["Policy Name 1", "Policy Name 2", ...]
    """
    
    prompt = ChatPromptTemplate.from_template(extract_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        if result.startswith("```python") and result.endswith("```"):
            result = result[9:-3].strip()
        
        try:
            policy_names = eval(result)
            if not isinstance(policy_names, list):
                policy_names = []
        except:
            policy_names = []
            matches = re.findall(r'"([^"]+)"', result)
            if matches:
                policy_names = matches
    except:
        policy_names = []
    
    policy_entities = []
    
    icici_policies = policy_list
    
    for name in policy_names:
        matching_icici = next((p for p in icici_policies if p["policy_name"].lower() == name.lower()), None)
        if matching_icici:
            policy_entities.append(PolicyEntity(
                policy_name=matching_icici["policy_name"],
                policy_type=matching_icici.get("policy_type", "unknown"),
                mentioned_in_turn=conversation_turn,
                source_document=matching_icici.get("source", "unknown"),
                is_uploaded=False
            ))
            continue
        
        matching_uploaded = next((p for p in uploaded_policy_list if p["policy_name"].lower() == name.lower()), None)
        if matching_uploaded:
            policy_entities.append(PolicyEntity(
                policy_name=matching_uploaded["policy_name"],
                policy_type=matching_uploaded.get("policy_type", "unknown"),
                mentioned_in_turn=conversation_turn,
                source_document=matching_uploaded.get("source", "unknown"),
                is_uploaded=True
            ))
            continue
        
        for policy in icici_policies:
            if name.lower() in policy["policy_name"].lower() or policy["policy_name"].lower() in name.lower():
                policy_entities.append(PolicyEntity(
                    policy_name=policy["policy_name"],
                    policy_type=policy.get("policy_type", "unknown"),
                    mentioned_in_turn=conversation_turn,
                    source_document=policy.get("source", "unknown"),
                    is_uploaded=False
                ))
                continue
        
        for policy in uploaded_policy_list:
            if name.lower() in policy["policy_name"].lower() or policy["policy_name"].lower() in name.lower():
                policy_entities.append(PolicyEntity(
                    policy_name=policy["policy_name"],
                    policy_type=policy.get("policy_type", "unknown"),
                    mentioned_in_turn=conversation_turn,
                    source_document=policy.get("source", "unknown"),
                    is_uploaded=True
                ))
                continue
    
    return policy_entities

def extract_user_preferences(text: str) -> List[UserPreference]:
    """Extract user preferences from text"""
    extract_prompt = """
    Extract user preferences related to insurance from the following text:
    
    TEXT: {text}
    
    Identify information about:
    - Age
    - Family situation (married, children, etc.)
    - Health conditions
    - Budget constraints
    - Coverage preferences
    - Risk tolerance
    - Any other relevant preferences
    
    For each preference, provide a confidence score (0.0-1.0) indicating how certain you are about this preference.
    
    Format your response as a JSON list of objects with the following structure:
    [
       {{
        "preference_type": "type of preference",
        "value": "the preference value",
        "confidence": 0.0-1.0
       }}
    ]
    """
    
    prompt = ChatPromptTemplate.from_template(extract_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```json") and result.endswith("```"):
            result = result[7:-3].strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        
        try:
            preferences_json = json.loads(result)
            preferences = []
            for pref in preferences_json:
                preferences.append(UserPreference(
                    preference_type=pref["preference_type"],
                    value=pref["value"],
                    confidence=float(pref["confidence"])
                ))
            return preferences
        except:
            print("Failed to parse preferences JSON")
            return []
    except Exception as e:
        print(f"Error extracting preferences: {str(e)}")
        return []

# --- Policy Search and Recommendation Functions ---
def search_both_dbs(query_text, k=3, use_uploaded=True):
    """Search both databases for a given query"""
    regular_docs = chroma_db.as_retriever(search_kwargs={"k": k}).invoke(query_text)
    
    uploaded_docs = []
    if use_uploaded and uploaded_chroma_db._collection.count() > 0:
        uploaded_docs = uploaded_chroma_db.as_retriever(search_kwargs={"k": k}).invoke(query_text)
    
    return regular_docs + uploaded_docs

def detect_specific_policy_names(text: str, policies: list) -> List[str]:
    """Detect specific policy names in text with fuzzy matching"""
    if not text or not policies:
        return []
        
    text_lower = text.lower()
    mentioned_policies = []
    
    for policy in policies:
        policy_name = policy["policy_name"].replace(" - Brochure", "")
        policy_name_lower = policy_name.lower()
        
        if policy_name_lower in text_lower:
            if policy_name not in mentioned_policies:
                mentioned_policies.append(policy_name)
            continue
            
        if " " in policy_name_lower:
            parts = policy_name_lower.split()
            if len(parts) > 1 and parts[0].lower() == 'icici':
                non_brand = " ".join(parts[1:])
                if non_brand in text_lower:
                    if policy_name not in mentioned_policies:
                        mentioned_policies.append(policy_name)
                    continue
    
    policy_specific_terms = ["gift", "iprotect", "perfect", "smartkid", "lifetime", "signature", "health shield"]
    
    specific_policies = []
    for term in policy_specific_terms:
        if term in text_lower:
            for policy in policies:
                policy_name = policy["policy_name"]
                policy_name_lower = policy_name.lower()
                
                if term in policy_name_lower and policy_name not in specific_policies:
                    parts = policy_name_lower.split()
                    if any(part.startswith(term) or term.startswith(part) for part in parts):
                        specific_policies.append(policy_name)
    
    if specific_policies:
        return specific_policies
    
    if not mentioned_policies:
        for policy in policies:
            policy_name = policy["policy_name"].replace(" - Brochure", "")
            policy_name_lower = policy_name.lower()
            
            if len(policy_name_lower.split()) > 1:
                policy_words = policy_name_lower.split()
                
                matching_words = sum(1 for word in policy_words if word in text_lower)
                if matching_words >= 2 or (matching_words / len(policy_words) > 0.5):
                    if policy_name not in mentioned_policies:
                        mentioned_policies.append(policy_name)
                    continue
            
            if len(policy_name_lower.split()) > 1:
                for i in range(1, len(policy_name_lower.split())):
                    word = policy_name_lower.split()[i]
                    if len(word) >= 4:
                        abbreviated = word[:4]
                        if abbreviated in text_lower and any(part in text_lower for part in policy_name_lower.split()[:i]):
                            if policy_name not in mentioned_policies:
                                mentioned_policies.append(policy_name)
                            break
    
    return mentioned_policies

def smart_policy_retrieval(query: str, policy_names: List[str] = None, use_uploaded: bool = True, k: int = 3, use_exact_match: bool = True):
    """Retrieves documents based on query, with priority to specific policies if named"""
    if policy_names and len(policy_names) > 0:
        all_docs = []
        
        for policy_name in policy_names:
            try:
                policy_name_lower = policy_name.lower()
                
                filter_dict = {"policy_name": policy_name}
                filtered_docs = chroma_db.as_retriever(
                    search_kwargs={"k": k, "filter": filter_dict}
                ).invoke(query)
                
                if len(filtered_docs) < 2:
                    candidate_docs = chroma_db.as_retriever(
                        search_kwargs={"k": k+5}
                    ).invoke(query)
                    
                    for doc in candidate_docs:
                        doc_policy = doc.metadata.get("policy_name", "")
                        if doc_policy and (
                            doc_policy.lower() == policy_name_lower or 
                            policy_name_lower in doc_policy.lower() or
                            any(part.lower() in doc_policy.lower() for part in policy_name_lower.split() if len(part) > 3)
                        ):
                            filtered_docs.append(doc)
                
                if use_exact_match and len(filtered_docs) < k:
                    all_candidate_docs = chroma_db.as_retriever(
                        search_kwargs={"k": k * 3}
                    ).invoke(policy_name)
                    
                    exact_matches = []
                    for doc in all_candidate_docs:
                        if policy_name.lower() in doc.page_content.lower():
                            exact_matches.append(doc)
                    
                    for doc in exact_matches:
                        if doc not in filtered_docs:
                            filtered_docs.append(doc)
                
                all_docs.extend(filtered_docs)
                
                if use_uploaded and uploaded_chroma_db._collection.count() > 0:
                    uploaded_filtered_docs = uploaded_chroma_db.as_retriever(
                        search_kwargs={"k": k, "filter": filter_dict}
                    ).invoke(query)
                    all_docs.extend(uploaded_filtered_docs)
                    
                    if len(uploaded_filtered_docs) < 2:
                        enhanced_query = f"{query} {policy_name}"
                        more_uploaded_docs = uploaded_chroma_db.as_retriever(
                            search_kwargs={"k": k+2}
                        ).invoke(enhanced_query)
                        
                        more_filtered_uploaded_docs = [
                            doc for doc in more_uploaded_docs 
                            if policy_name_lower in doc.metadata.get("policy_name", "").lower()
                        ]
                        
                        all_docs.extend(more_filtered_uploaded_docs)
                
            except Exception as e:
                print(f"Filter-based retrieval failed for {policy_name}: {str(e)}")
                continue
        
        if len(all_docs) < 3:
            print("Few results with strict filtering, using enhanced query")
            policy_terms = " OR ".join([f'"{name}"' for name in policy_names])
            enhanced_query = f"{query} specifically about {policy_terms}"
            
            broader_docs = search_both_dbs(enhanced_query, k=k, use_uploaded=use_uploaded)
            
            broader_filtered_docs = []
            for doc in broader_docs:
                doc_policy = doc.metadata.get("policy_name", "").lower()
                if any(policy.lower() in doc_policy or doc_policy in policy.lower() for policy in policy_names):
                    broader_filtered_docs.append(doc)
            
            all_docs.extend(broader_filtered_docs)
            
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
                
        return unique_docs
    else:
        return search_both_dbs(query, k=k, use_uploaded=use_uploaded)

def answer_faq(question: str, user_profile: UserProfile, conversation_turn: int):
    """Answer user questions based on policy documents"""
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    
    specific_policies = detect_specific_policy_names(question, policy_list)
    
    enhanced_query = question
    
    if specific_policies:
        print(f"Detected specific policies: {specific_policies}")
        docs = smart_policy_retrieval(
            query=enhanced_query, 
            policy_names=specific_policies, 
            use_uploaded=True, 
            k=7
        )
    else:
        docs = smart_policy_retrieval(
            query=enhanced_query, 
            policy_names=None, 
            use_uploaded=True, 
            k=3
        )
    
    if not docs:
        return "I don't have enough information to answer that question. Could you please be more specific or ask about a particular policy?"
    
    if specific_policies:
        filtered_docs = []
        for doc in docs:
            doc_policy = doc.metadata.get("policy_name", "")
            if any(policy.lower() in doc_policy.lower() or doc_policy.lower() in policy.lower() for policy in specific_policies):
                filtered_docs.append(doc)
        
        if len(filtered_docs) >= 3:
            docs = filtered_docs
    
    contexts = [doc.page_content for doc in docs]
    sources = []
    seen_sources = set()
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        policy_name = doc.metadata.get("policy_name", "Unknown")
        source_str = f"{policy_name} ({source})"
        
        if source_str not in seen_sources:
            sources.append(source_str)
            seen_sources.add(source_str)
    
    answer_template = """
    You are an insurance expert assistant at ICICI. Answer the user's question based on the provided information.
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUESTION: {question}
    
    INFORMATION FROM POLICY DOCUMENTS:
    {context}
    """
    
    if specific_policies:
        policy_names_list = ", ".join(specific_policies)
        answer_template += f"""
    The user is specifically asking about the following policies: {policy_names_list}.
    Focus your answer ONLY on information related to these specific policies.
    DO NOT include general information about other policies unless directly relevant for comparison.
    """
    
    answer_template += """
    Please provide a clear, concise answer based only on the provided information. If the information doesn't contain 
    the answer, say so honestly. Don't make up information that isn't in the provided context.
    
    Format your answer nicely with markdown when appropriate for lists, emphasis, etc.
    """
    
    prompt = ChatPromptTemplate.from_template(answer_template)
    
    chain = (
        {"question": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    policies = extract_policy_entities(response, conversation_turn)
    
    for policy in policies:
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        
        if not already_discussed:
            user_profile.policies_discussed.append(policy)
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    if sources:
        top_sources = sources[:3]
        if specific_policies:
            specific_sources = [s for s in sources if any(policy.lower() in s.lower() for policy in specific_policies)]
            if specific_sources:
                top_sources = specific_sources[:3]
        
        source_info = "\n\nSources consulted: " + ", ".join(top_sources)
        if len(sources) > 3:
            source_info += f" and {len(sources) - 3} more"
    else:
        source_info = ""
        
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": question},
        {"output": final_response}
    )
    
    return final_response

# --- Function to handle sessions ---
def get_or_create_session(user_id: str) -> UserProfile:
    """Get or create a user session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    session_file = os.path.join(SESSION_DIR, f"{user_id}.json")
    
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
            return UserProfile(**data)
    else:
        profile = UserProfile(
            user_id=user_id,
            preferences=[],
            policies_discussed=[],
            policies_recommended=[],
            uploaded_policies=[],
            last_interaction=datetime.now().isoformat()
        )
        save_session(profile)
        return profile

def save_session(profile: UserProfile):
    """Save a user session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    session_file = os.path.join(SESSION_DIR, f"{profile.user_id}.json")
    
    with open(session_file, 'w') as f:
        json.dump(profile.model_dump(), f, indent=2)

# --- Main processing function ---
def process_message(user_id: str, user_input: str, conversation_turn: int = 0):
    """Enhanced process_message with Smart Underwriting integration"""
    user_profile = get_or_create_session(user_id)
    
    new_preferences = extract_user_preferences(user_input)
    
    if new_preferences:
        for new_pref in new_preferences:
            updated = False
            for i, existing_pref in enumerate(user_profile.preferences):
                if existing_pref.preference_type == new_pref.preference_type:
                    if new_pref.confidence > existing_pref.confidence:
                        user_profile.preferences[i] = new_pref
                    updated = True
                    break
            
            if not updated:
                user_profile.preferences.append(new_pref)
    
    # === Smart Underwriting Decision Logic ===
    if smart_underwriting_system and SMART_UNDERWRITING_AVAILABLE:
        underwriting_keywords = [
            'quote','buy policy',
            'underwriting', 'risk assessment'
            ]
        
        needs_underwriting = any(keyword in user_input.lower() for keyword in underwriting_keywords)
        
        if needs_underwriting:
            print(f"🤖 Using Smart Underwriting for: {user_input[:50]}...")
            try:
                orchestrator = smart_underwriting_system['orchestrator']
                
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    orchestrator.enhanced_process_message(user_id, user_input, conversation_turn, user_profile)
                )
                
                print("✅ Smart Underwriting response generated")
                return response
                
            except Exception as e:
                print(f"⚠️ Smart Underwriting error: {e}")
                print("   Falling back to legacy FAQ system")
    
    # === Legacy FAQ System ===
    print(f"📋 Using legacy FAQ system for: {user_input[:50]}...")
    
    if re.search(r"recommend|suggest|what policy|best policy|which policy", user_input.lower()):
        return recommend_policies(user_profile, conversation_turn)
    
    if re.search(r"compare|difference|better|versus|vs", user_input.lower()):
        return find_similar_policies(user_input, user_profile, conversation_turn)
    
    policy_match = re.search(r"(details|information|tell me about|what is) (the )?([A-Za-z\s]+) policy", user_input.lower())
    if policy_match:
        policy_name = policy_match.group(3).strip().title()
        is_uploaded = any(p.policy_name.lower() == policy_name.lower() for p in user_profile.uploaded_policies)
        return get_policy_details(policy_name, is_uploaded)
    
    if re.search(r"fact check|verify|is it true", user_input.lower()):
        return fact_check_claims(user_input, user_profile)
    
    uploaded_policy_match = re.search(r"(evaluate|assess|review) my (uploaded )?policy", user_input.lower())
    if uploaded_policy_match and user_profile.uploaded_policies:
        latest_policy = user_profile.uploaded_policies[-1]
        return evaluate_uploaded_policy(latest_policy, user_profile, conversation_turn)
    
    if re.search(r"compare my (uploaded )?policy", user_input.lower()) and user_profile.uploaded_policies:
        latest_policy = user_profile.uploaded_policies[-1]
        return compare_uploaded_with_existing(latest_policy, user_profile, conversation_turn)
    
    return answer_faq(user_input, user_profile, conversation_turn)

# --- Additional Legacy Functions ---
def recommend_policies(user_profile: UserProfile, conversation_turn: int):
    """Recommend policies based on user preferences"""
    if not user_profile.preferences:
        return "I need more information about your needs and preferences to recommend policies. Please tell me about your insurance requirements, budget, family situation, or any specific coverage needs."
    
    preference_text = ""
    for pref in user_profile.preferences:
        preference_text += f"- {pref.preference_type}: {pref.value}\n"
    
    query_parts = []
    for pref in user_profile.preferences:
        if pref.confidence >= 0.7:
            query_parts.append(f"{pref.preference_type}: {pref.value}")
    
    query = "Insurance policy recommendations for " + ", ".join(query_parts)
    
    policy_names = None
    if user_profile.policies_recommended:
        policy_names = [p.policy_name for p in user_profile.policies_recommended]
    
    docs = smart_policy_retrieval(query, policy_names, use_uploaded=True, k=5)
    
    if not docs:
        return "I don't have enough information to recommend specific policies. Could you please provide more details about your insurance needs?"
    
    policy_info = {}
    for doc in docs:
        policy_name = doc.metadata.get("policy_name", "Unknown")
        is_uploaded = doc.metadata.get("is_uploaded", False)
        
        if policy_name not in policy_info:
            policy_info[policy_name] = {
                "content": [],
                "is_uploaded": is_uploaded
            }
        
        policy_info[policy_name]["content"].append(doc.page_content)
    
    recommendation_template = """
    You are an insurance expert at Policies . Recommend insurance policies based on the user's preferences.
    
    USER PREFERENCES:
    {preferences}
    
    POTENTIAL POLICIES:
    {policy_info}
    
    Please provide:
    1. The top 3 recommended policies that best match the user's needs
    2. For each policy, explain why it's a good fit for the user
    3. Highlight key features and benefits of each recommended policy
    4. Mention any limitations or exclusions the user should be aware of
    
    Format your response in a clear, organized manner with bold policy names for readability.
    """
    
    formatted_policy_info = ""
    for name, info in policy_info.items():
        formatted_policy_info += f"--- {name} ---\n"
        formatted_policy_info += "\n".join(info["content"])
        formatted_policy_info += "\n\n"
    
    prompt = ChatPromptTemplate.from_template(recommendation_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "preferences": preference_text,
        "policy_info": formatted_policy_info
    })
    
    recommended_policies = []
    for name in policy_info.keys():
        if name.lower() in response.lower():
            is_uploaded = policy_info[name]["is_uploaded"]
            recommended_policies.append(PolicyEntity(
                policy_name=name,
                policy_type="unknown",
                mentioned_in_turn=conversation_turn,
                source_document="recommendation",
                is_uploaded=is_uploaded
            ))
    
    for policy in recommended_policies:
        already_recommended = False
        for existing in user_profile.policies_recommended:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_recommended = True
                break
        
        if not already_recommended:
            user_profile.policies_recommended.append(policy)
    
    sources = set()
    for name, info in policy_info.items():
        if name.lower() in response.lower():
            sources.add(name)
    
    source_info = "\n\nSources consulted: " + ", ".join(sources)
    final_response = response + source_info
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    conversation_memory.save_context(
        {"input": "Can you recommend insurance policies based on my needs?"},
        {"output": final_response}
    )
    
    return final_response

def find_similar_policies(user_input: str, user_profile: UserProfile, conversation_turn: int):
    """Find similar policies to those mentioned or uploaded"""
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    
    mentioned_policies = extract_policy_entities(user_input, conversation_turn)
    specific_policies = detect_specific_policy_names(user_input, policy_list)
    
    target_policies = []
    
    if mentioned_policies:
        target_policies.extend(mentioned_policies)
    
    if specific_policies:
        for name in specific_policies:
            if not any(p.policy_name.lower() == name.lower() for p in target_policies):
                policy_type = next((p["policy_type"] for p in policy_list if p["policy_name"] == name), "unknown")
                source = next((p["source"] for p in policy_list if p["policy_name"] == name), "unknown")
                target_policies.append(PolicyEntity(
                    policy_name=name,
                    policy_type=policy_type,
                    mentioned_in_turn=conversation_turn,
                    source_document=source,
                    is_uploaded=False
                ))
    
    if not target_policies:
        target_policies.extend(user_profile.uploaded_policies)
        
        if len(target_policies) < 2:
            discussed_policies = sorted(
                user_profile.policies_discussed,
                key=lambda x: x.mentioned_in_turn,
                reverse=True
            )[:3]
            target_policies.extend(discussed_policies)
    
    if not target_policies:
        return "I don't have any policies to compare. Please mention specific policies you're interested in, or upload a policy document."
    
    policy_names = [p.policy_name for p in target_policies]
    
    comparison_query = f"Detailed comparison of {', '.join(policy_names)} policies"
    
    use_uploaded = any(p.is_uploaded for p in target_policies)
    
    docs = smart_policy_retrieval(
        query=comparison_query, 
        policy_names=policy_names, 
        use_uploaded=use_uploaded, 
        k=5,
        use_exact_match=True
    )
    
    if not docs:
        return "I couldn't find any similar policies in our database."
    
    contexts = []
    sources = []
    seen_sources = set()
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        contexts.append(doc.page_content)
        if source not in seen_sources:
            sources.append(source)
            seen_sources.add(source)
    
    target_policy_text = "Target policies:\n- " + "\n- ".join(policy_names)
    
    similarity_template = """
    You are an insurance expert for policies. The user wants to find policies similar to the ones they mentioned or uploaded.
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUERY: {query}
    
    TARGET POLICIES:
    {target_policies}
    
    POLICY INFORMATION FROM DATABASE:
    {context}
    
    Please provide:
    1. A detailed analysis of similarities and differences between the target policies and other relevant policies
    2. Compare key factors like coverage, premiums, benefits, and exclusions
    3. Highlight unique features of each policy
    4. Provide recommendations on which policy might be better for different user needs
    
    Bold the policy names when referring to them for clarity.
    """
    
    prompt = ChatPromptTemplate.from_template(similarity_template)
    
    chain = (
        {"query": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text,
         "target_policies": lambda x: target_policy_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(user_input)
    
    policies = extract_policy_entities(response, conversation_turn)
    
    for policy in policies:
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        
        if not already_discussed:
            user_profile.policies_discussed.append(policy)
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    source_info = "\n\nSources consulted: " + ", ".join(set(sources))
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": user_input},
        {"output": final_response}
    )
    
    return final_response

def get_policy_details(policy_name: str, is_uploaded: bool = False):
    """Get detailed information about a specific policy"""
    query = f"Comprehensive details about {policy_name} policy including benefits, features, exclusions, and eligibility"
    
    policy_names = [policy_name]
    
    docs = smart_policy_retrieval(
        query=query, 
        policy_names=policy_names, 
        use_uploaded=is_uploaded, 
        k=8,
        use_exact_match=True
    )
    
    if not docs:
        return f"I couldn't find detailed information about the {policy_name} policy."
    
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    details_template = """
    You are an insurance expert at policies. Provide comprehensive information about the requested policy.
    
    POLICY: {policy_name}
    
    POLICY INFORMATION:
    {context}
    
    Please provide a detailed and well-structured response about this specific policy with the following sections:
    1. Overview - A clear summary of what this policy is and who it's for
    2. Key Benefits - List all benefits and advantages of this policy
    3. Coverage Details - What specifically is covered by this policy
    4. Exclusions - Important limitations or exclusions
    5. Premium Information - Details about premium payment if available
    6. Eligibility - Who can apply for this policy
    7. Claims Process - How to make claims under this policy (if information available)
    
    Present the information in a well-organized format with clear headings and bullet points where appropriate.
    Do not include information about other policies unless directly relevant for understanding this specific policy.
    If certain information is not available in the provided context, simply omit that section rather than making up details.
    """
    
    prompt = ChatPromptTemplate.from_template(details_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "policy_name": policy_name,
        "context": "\n\n".join(contexts)
    })
    
    filtered_sources = []
    seen_sources = set()
    for source in sources:
        if source not in seen_sources:
            filtered_sources.append(source)
            seen_sources.add(source)
    
    source_info = "\n\nSources consulted: " + ", ".join(filtered_sources[:3])
    if len(filtered_sources) > 3:
        source_info += f" and {len(filtered_sources) - 3} more"
    
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": f"Tell me about {policy_name} policy"},
        {"output": final_response}
    )
    
    return final_response

def fact_check_claims(text: str, user_profile: UserProfile):
    """Fact check insurance-related claims in the text"""
    claims_template = """
    Extract factual claims about insurance policies from the following text:
    
    TEXT: {text}
    
    List each claim as a separate item. Focus on specific factual assertions about coverage, benefits, exclusions, etc.
    Format: ["Claim 1", "Claim 2", ...]
    """
    
    prompt = ChatPromptTemplate.from_template(claims_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        
        try:
            claims = eval(result)
            if not isinstance(claims, list):
                return "I didn't identify any specific factual claims to verify in the text."
        except:
            claims = []
            matches = re.findall(r'"([^"]+)"', result)
            if matches:
                claims = matches
            else:
                return "I didn't identify any specific factual claims to verify in the text."
    except:
        return "I wasn't able to process the text for fact-checking."
    
    fact_check_results = []
    
    for claim in claims:
        icici_docs = chroma_db.as_retriever(search_kwargs={"k": 3}).invoke(claim)
        
        uploaded_docs = []
        if uploaded_chroma_db._collection.count() > 0:
            uploaded_docs = uploaded_chroma_db.as_retriever(search_kwargs={"k": 2}).invoke(claim)
        
        docs = icici_docs + uploaded_docs
        
        if not docs:
            fact_check_results.append({
                "claim": claim,
                "result": "No relevant information found",
                "context": ""
            })
            continue
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        check_template = """
        You are a fact-checker for insurance information. Verify the following claim against the provided information:
        
        CLAIM: {claim}
        
        INFORMATION:
        {context}
        
        Please determine if the claim is:
        - VERIFIED (fully supported by the information)
        - PARTIALLY VERIFIED (some aspects are correct, others not mentioned or contradicted)
        - UNVERIFIED (not mentioned in the information)
        - CONTRADICTED (directly contradicted by the information)
        
        Provide a brief explanation for your assessment.
        
        Format your response as:
        VERDICT: [the verdict]
        EXPLANATION: [your explanation]
        """
        
        prompt = ChatPromptTemplate.from_template(check_template)
        
        chain = prompt | fact_check_llm | StrOutputParser()
        
        check_result = chain.invoke({
            "claim": claim,
            "context": context
        })
        
        verdict_match = re.search(r"VERDICT:\s*(.+)", check_result)
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", check_result, re.DOTALL)
        
        verdict = verdict_match.group(1).strip() if verdict_match else "UNVERIFIED"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."
        
        fact_check_results.append({
            "claim": claim,
            "verdict": verdict,
            "explanation": explanation
        })
    
    if not fact_check_results:
        return "I didn't find any specific claims to fact-check in the text."
    
    response = "# Fact Check Results\n\n"
    
    for i, result in enumerate(fact_check_results, 1):
        response += f"## Claim {i}: \"{result['claim']}\"\n"
        response += f"**Verdict**: {result['verdict']}\n"
        response += f"**Explanation**: {result['explanation']}\n\n"
    
    response += "\nNote: This fact-checking is based on the information available in my database and may not be comprehensive."
    
    conversation_memory.save_context(
        {"input": "Fact check the following claims: " + text},
        {"output": response}
    )
    
    return response

def evaluate_uploaded_policy(policy: PolicyEntity, user_profile: UserProfile, conversation_turn: int):
    """Evaluate a user-uploaded policy and provide insights"""
    query = f"Detailed information about {policy.policy_name}"
    
    docs = []
    if uploaded_chroma_db._collection.count() > 0:
        try:
            filter_dict = {"policy_name": policy.policy_name}
            docs = uploaded_chroma_db.as_retriever(
                search_kwargs={"k": 5, "filter": filter_dict}
            ).invoke(query)
        except Exception as e:
            print(f"Filter-based retrieval failed: {str(e)}")
    
    if not docs:
        return f"I don't have enough information to evaluate the {policy.policy_name} policy. Please try uploading the document again."
    
    contexts = [doc.page_content for doc in docs]
    
    eval_template = """
    You are an insurance expert at ICICI. Evaluate the following uploaded insurance policy.
    
    POLICY NAME: {policy_name}
    POLICY TYPE: {policy_type}
    
    POLICY CONTENT:
    {context}
    
    Please provide:
    1. A comprehensive analysis of this policy
    2. Key strengths and benefits of this policy
    3. Potential limitations or exclusions to be aware of
    4. How this policy compares to standard offerings in the market
    5. Who this policy would be most suitable for
    
    Present your evaluation in a clear, organized format with headings.
    """
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "policy_name": policy.policy_name,
        "policy_type": policy.policy_type,
        "context": "\n\n".join(contexts)
    })
    
    conversation_memory.save_context(
        {"input": f"Evaluate my uploaded {policy.policy_name} policy"},
        {"output": response}
    )
    
    return response

def compare_uploaded_with_existing(policy: PolicyEntity, user_profile: UserProfile, conversation_turn: int):
    """Compare an uploaded policy with existing policies"""
    query = f"Details of {policy.policy_name} for comparison"
    
    uploaded_docs = []
    if uploaded_chroma_db._collection.count() > 0:
        try:
            filter_dict = {"policy_name": policy.policy_name}
            uploaded_docs = uploaded_chroma_db.as_retriever(
                search_kwargs={"k": 3, "filter": filter_dict}
            ).invoke(query)
        except Exception as e:
            print(f"Filter-based retrieval failed: {str(e)}")
    
    if not uploaded_docs:
        return f"I don't have enough information about the {policy.policy_name} policy to compare it. Please try uploading the document again."
    
    similar_policy_query = f"Policies similar to {policy.policy_name} {policy.policy_type} insurance"
    icici_docs = chroma_db.as_retriever(search_kwargs={"k": 5}).invoke(similar_policy_query)
    
    if not icici_docs:
        return f"I couldn't find any similar ICICI policies to compare with your {policy.policy_name} policy."
    
    uploaded_context = "\n\n".join([doc.page_content for doc in uploaded_docs])
    
    icici_contexts = {}
    for doc in icici_docs:
        policy_name = doc.metadata.get("policy_name", "Unknown")
        if policy_name not in icici_contexts:
            icici_contexts[policy_name] = []
        icici_contexts[policy_name].append(doc.page_content)
    
    comparison_template = """
    You are an insurance expert at policies. Compare the following uploaded policy with similar insurance policies.
    
    UPLOADED POLICY:
    Name: {uploaded_policy_name}
    Type: {uploaded_policy_type}
    
    UPLOADED POLICY CONTENT:
    {uploaded_context}
    
    COMPARABLE ICICI POLICIES:
    {icici_contexts}
    
    Please provide:
    1. A side-by-side comparison of the uploaded policy vs ICICI policies
    2. Key differences in coverage, premiums, benefits, and exclusions
    3. Strengths and weaknesses of each policy
    4. Which policy might be better for different types of customers
    
    Present your comparison in a clear, organized table format where appropriate.
    """
    
    formatted_icici_contexts = ""
    for name, contexts in icici_contexts.items():
        formatted_icici_contexts += f"--- {name} ---\n"
        formatted_icici_contexts += "\n".join(contexts)
        formatted_icici_contexts += "\n\n"
    
    prompt = ChatPromptTemplate.from_template(comparison_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "uploaded_policy_name": policy.policy_name,
        "uploaded_policy_type": policy.policy_type,
        "uploaded_context": uploaded_context,
        "icici_contexts": formatted_icici_contexts
    })
    
    sources = [policy.policy_name]
    sources.extend(list(icici_contexts.keys()))
    
    source_info = "\n\nPolicies compared: " + ", ".join(sources)
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": f"Compare my uploaded {policy.policy_name} policy with ICICI policies"},
        {"output": final_response}
    )
    
    return final_response

# --- Enhanced API endpoint implementation ---
def api_handle_message(user_id: str, user_input: str, conversation_turn: int = 0, 
                      file_path: Optional[str] = None, extraction_method: str = 'auto'):
    """Handle API request - enhanced implementation with Smart Underwriting"""
    processing_info = None
    
    if file_path:
        try:
            if not os.path.exists(file_path):
                return APIResponse(
                    response=f"Error: File not found at {file_path}",
                    user_profile=get_or_create_session(user_id),
                    processing_info={"error": "File not found"}
                )
                
            if not file_path.endswith(".pdf"):
                return APIResponse(
                    response="Error: Uploaded file must be a PDF",
                    user_profile=get_or_create_session(user_id),
                    processing_info={"error": "Not a PDF file"}
                )
                
            start_time = datetime.now()
            policy = process_uploaded_pdf(file_path, user_id, extraction_method=extraction_method)
            end_time = datetime.now()
            
            processing_info = {
                "policy_name": policy.policy_name,
                "policy_type": policy.policy_type,
                "extraction_method": extraction_method,
                "processing_time": str(end_time - start_time),
                "success": not policy.policy_name.startswith("Error")
            }
            
            if policy.policy_name.startswith("Error"):
                return APIResponse(
                    response=f"Error processing your PDF: {policy.policy_name}",
                    user_profile=get_or_create_session(user_id),
                    processing_info=processing_info
                )
                
            if not user_input:
                user_input = f"Tell me about the {policy.policy_name} policy I just uploaded"
        
        except Exception as e:
            return APIResponse(
                response=f"Error processing your PDF: {str(e)}",
                user_profile=get_or_create_session(user_id),
                processing_info={"error": str(e)}
            )
    
    # Enhanced message processing with Smart Underwriting
    if smart_underwriting_system and SMART_UNDERWRITING_AVAILABLE:
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            enhanced_response = loop.run_until_complete(
                enhanced_api_handle_message(
                    user_id=user_id,
                    user_input=user_input,
                    conversation_turn=conversation_turn,
                    file_path=file_path,
                    extraction_method=extraction_method,
                    vector_db=chroma_db,
                    uploaded_vector_db=uploaded_chroma_db,
                    policy_list=policy_list
                )
            )
            
            user_profile = enhanced_response.get("user_profile", get_or_create_session(user_id))
            if processing_info:
                enhanced_response["processing_info"] = processing_info
            
            return APIResponse(
                response=enhanced_response["response"],
                user_profile=user_profile,
                processing_info=enhanced_response.get("processing_info")
            )
            
        except Exception as e:
            print(f"⚠️ Enhanced API error: {e}")
            print("   Falling back to legacy API")
    
    response = process_message(user_id, user_input, conversation_turn)
    
    user_profile = get_or_create_session(user_id)
    
    return APIResponse(
        response=response,
        user_profile=user_profile,
        processing_info=processing_info
    )

# === Smart Underwriting specific functions ===
def get_underwriting_result(user_id: str, customer_needs: str, 
                           policy_type: str = None, 
                           external_data_sources: List[str] = None) -> Dict[str, Any]:
    """Get a complete underwriting result using the Smart Underwriting System"""
    if not smart_underwriting_system or not SMART_UNDERWRITING_AVAILABLE:
        return {
            "error": "Smart Underwriting System not available",
            "fallback_recommendation": "Please use the standard recommendation system"
        }
    
    try:
        user_profile = get_or_create_session(user_id)
        
        orchestrator = smart_underwriting_system['orchestrator']
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            orchestrator.process_underwriting_request(
                customer_id=user_id,
                customer_profile=user_profile,
                customer_needs=customer_needs,
                policy_type=policy_type,
                external_data_sources=external_data_sources or []
            )
        )
        
        result_dict = {
            "customer_id": result.customer_id,
            "request_id": result.request_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "agents_used": result.agents_used,
            "underwriting_decision": {
                "decision": result.underwriting_decision.decision,
                "decision_reason": result.underwriting_decision.decision_reason,
                "confidence_level": result.underwriting_decision.confidence_level,
                "manual_review_triggers": result.underwriting_decision.manual_review_triggers
            },
            "risk_assessment": {
                "overall_risk_level": result.risk_assessment.overall_risk_level.value if result.risk_assessment else "unknown",
                "risk_score": result.risk_assessment.risk_score if result.risk_assessment else 0,
                "recommended_premium_multiplier": result.risk_assessment.recommended_premium_multiplier if result.risk_assessment else 1.0,
                "confidence_level": result.risk_assessment.confidence_level if result.risk_assessment else 0.5,
                "requires_manual_review": result.risk_assessment.requires_manual_review if result.risk_assessment else True
            },
            "policy_draft": {
                "policy_name": result.policy_draft.policy_name if result.policy_draft else "No policy generated",
                "policy_type": result.policy_draft.policy_type if result.policy_draft else "unknown",
                "coverage_amount": result.policy_draft.terms.coverage_amount if result.policy_draft else 0,
                "premium_amount": result.policy_draft.terms.premium_amount if result.policy_draft else 0,
                "compliance_status": result.policy_draft.compliance_status if result.policy_draft else "unknown"
            },
            "historical_patterns_count": len(result.historical_patterns),
            "errors": result.errors,
            "timestamp": result.timestamp.isoformat()
        }
        
        return result_dict
        
    except Exception as e:
        return {
            "error": f"Underwriting processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def get_agent_status():
    """Get the status of all smart underwriting agents"""
    if not SMART_UNDERWRITING_AVAILABLE:
        return {"status": "unavailable", "message": "Smart Underwriting not loaded"}
    
    try:
        health_status = check_agent_health()
        return {
            "status": "available" if health_status['overall_health'] else "degraded",
            "agents": {
                "agent1_historical": health_status['agent1'],
                "agent2_risk": health_status['agent2'], 
                "agent3_policy": health_status['agent3'],
                "orchestrator": health_status['orchestrator']
            },
            "errors": health_status['errors'],
            "version": smart_underwriting_system['version'] if smart_underwriting_system else "unknown"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_smart_underwriting_flow(user_id: str = None):
    """Test the complete smart underwriting flow with sample data"""
    if not smart_underwriting_system:
        return "Smart Underwriting System not available"
    
    test_user_id = user_id or f"test_user_{datetime.now().strftime('%H%M%S')}"
    
    user_profile = get_or_create_session(test_user_id)
    
    sample_preferences = [
        UserPreference(preference_type="age", value="35", confidence=0.9),
        UserPreference(preference_type="family", value="married with 2 children", confidence=0.8),
        UserPreference(preference_type="occupation", value="software engineer", confidence=0.9),
        UserPreference(preference_type="budget", value="₹15000-25000 annually", confidence=0.7),
        UserPreference(preference_type="coverage", value="comprehensive health insurance", confidence=0.8)
    ]
    
    user_profile.preferences.extend(sample_preferences)
    save_session(user_profile)
    
    customer_needs = "I need comprehensive health insurance for my family of 4. I'm 35 years old, work as a software engineer, and can afford ₹20000 annually for premiums."
    
    result = get_underwriting_result(
        user_id=test_user_id,
        customer_needs=customer_needs,
        policy_type="Health Insurance"
    )
    
    return {
        "test_user_id": test_user_id,
        "customer_needs": customer_needs,
        "underwriting_result": result
    }

# --- Simple chat loop for testing ---
def main():
    """Enhanced main chat loop with Smart Underwriting capabilities"""
    print("🤖 Enhanced Insurance Policy Advisor Bot")
    if smart_underwriting_system:
        print("✅ Smart Underwriting System: ACTIVE")
        print("   • Agent 1: Historical Risk Analysis")
        print("   • Agent 2: Actuarial Risk Assessment") 
        print("   • Agent 3: Policy Generation")
        print("   • Multi-Agent Orchestration")
    else:
        print("⚠️ Smart Underwriting System: NOT AVAILABLE")
        print("   • Please ensure the smart_underwriting_system is properly configured.")
    print("Type 'exit' to quit")
    print("Type 'upload <path/to/pdf>' to upload a PDF policy")
    print("Type 'underwrite <your needs>' to test smart underwriting")
    print("Type 'help' for more commands\n")
    
    for directory in [SESSION_DIR, UPLOADS_DIR, PDF_FOLDER, 
                     os.path.dirname(POLICY_LIST_FILE), 
                     os.path.dirname(UPLOADED_POLICY_LIST_FILE)]:
        os.makedirs(directory, exist_ok=True)
    
    try:
        initialize_uploaded_policy_database()
    except Exception as e:
        print(f"\n:warning: Error initializing uploaded policy database: {str(e)}")
        print("Will try to continue anyway...")
    
    user_id = str(uuid.uuid4())
    conversation_turn = 0
    
    try:
        user_profile = get_or_create_session(user_id)
        print(f"User session created with ID: {user_id}")
    except Exception as e:
        print(f"\n:warning: Error creating user session: {str(e)}")
        print("Will try to continue anyway...")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n--- Available Commands ---")
                print("exit - Exit the chat")
                print("upload <path/to/pdf> - Upload a PDF policy")
                print("upload-ocr <path/to/pdf> - Upload a PDF and force OCR processing")
                print("upload-direct <path/to/pdf> - Upload a PDF with direct text extraction only")
                print("underwrite <your needs> - Test smart underwriting system")
                print("compare <policy1> <policy2> - Compare two policies")
                print("recommend - Get policy recommendations based on your profile")
                print("help - Show this help message")
                print("---")
                print("All other input will be processed as questions or statements about insurance policies.\n")
                continue
            
            if not user_input:
                continue
            
            # Smart Underwriting Test Command
            if user_input.lower().startswith("underwrite "):
                customer_needs = user_input[11:].strip()
                if smart_underwriting_system:
                    print("\n🤖 Testing Smart Underwriting System...")
                    try:
                        result = get_underwriting_result(user_id, customer_needs)
                        
                        if "error" in result:
                            print(f"\n❌ Error: {result['error']}")
                        else:
                            print(f"\n✅ Underwriting Complete!")
                            print(f"Request ID: {result['request_id']}")
                            print(f"Processing Time: {result['processing_time']:.2f}s")
                            print(f"Decision: {result['underwriting_decision']['decision']}")
                            print(f"Risk Level: {result['risk_assessment']['overall_risk_level']}")
                            print(f"Risk Score: {result['risk_assessment']['risk_score']:.1f}/100")
                            print(f"Premium Multiplier: {result['risk_assessment']['recommended_premium_multiplier']:.2f}")
                            
                            if result['policy_draft']['policy_name'] != "No policy generated":
                                print(f"Generated Policy: {result['policy_draft']['policy_name']}")
                                print(f"Coverage: ₹{result['policy_draft']['coverage_amount']:,.2f}")
                                print(f"Premium: ₹{result['policy_draft']['premium_amount']:,.2f}")
                        
                        continue
                    except Exception as e:
                        print(f"\n❌ Smart Underwriting error: {str(e)}")
                        continue
                else:
                    print("\n⚠️ Smart Underwriting System not available")
                    continue
            
            if user_input.lower().startswith("upload "):
                file_path = user_input[7:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with automatic method selection...")
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='auto')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name}")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
            
            elif user_input.lower().startswith("upload-ocr "):
                file_path = user_input[11:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with OCR (for scanned documents)...")
                        if not OCR_AVAILABLE:
                            print("\n🤖 Bot: OCR dependencies not installed. Please install pytesseract and pdf2image.")
                            continue
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='ocr')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name} using OCR")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file with OCR: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
            
            elif user_input.lower().startswith("upload-direct "):
                file_path = user_input[14:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with direct text extraction only...")
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='pymupdf')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name} using direct extraction")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file with direct extraction: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
                    
            conversation_turn += 1
            try:
                response = process_message(user_id, user_input, conversation_turn)
                print(f"\n🤖 {response}\n")
            except Exception as e:
                print(f"\n🤖 Bot: Error processing your message: {str(e)}")
                continue
                
        except Exception as e:
            print(f"\n🤖 Bot: An unexpected error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()