# 🏥 Insurance RAG - AI-Powered Insurance Policy Advisor

An intelligent insurance policy recommendation system that uses RAG (Retrieval-Augmented Generation) technology to provide personalized insurance advice and policy analysis.

## 🚀 Overview

Insurance RAG is a comprehensive AI-powered system that helps users:
- Get personalized insurance policy recommendations
- Analyze and compare insurance policies
- Upload and process custom insurance documents
- Receive intelligent underwriting decisions through multi-agent systems
- Get detailed policy explanations and comparisons

## ✨ Features

### Core Functionality
- **📄 Policy Analysis**: Process and analyze insurance policy documents (PDF format)
- **🤖 AI Chat Interface**: Interactive chat system for insurance queries
- **📊 Smart Recommendations**: Personalized policy recommendations based on user profile
- **🔍 Document Search**: Semantic and lexical search through policy documents
- **📤 Document Upload**: Upload and process custom insurance policies

### Advanced Features
- **🧠 Multi-Agent Underwriting System**: 
  - Agent 1: Historical Risk Pattern Analysis
  - Agent 2: Actuarial Risk Assessment  
  - Agent 3: Policy Generation
- **💡 Smart Underwriting**: Automated underwriting decisions with confidence scoring
- **📈 Risk Assessment**: Comprehensive risk analysis for customers
- **🎯 Personalized Matching**: Match customers with suitable policies based on profile

## 🛠️ Technology Stack

- **Backend**: Python, Streamlit
- **AI/ML**: 
  - LangChain for RAG implementation
  - OpenAI GPT models for language processing
  - ChromaDB for vector storage
  - Google Generative AI integration
- **Document Processing**: 
  - PyMuPDF for PDF processing
  - PyPDF2 for text extraction
  - OCR capabilities (Tesseract, pdf2image)
- **Vector Database**: Chroma for semantic search

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Google AI API key (optional)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mayureshpawashe/insurance_rag.git
   cd insurance_rag
   ```

2. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   **Option 1: Using .env file (Recommended)**
   
   Create a `.env` file in the backend directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   **Option 2: Using PowerShell (Windows)**
   ```powershell
   $env:OPENAI_API_KEY="your_openai_api_key_here"
   $env:GOOGLE_API_KEY="your_google_api_key_here"
   ```

   **Option 3: Using Command Prompt (Windows)**
   ```cmd
   set OPENAI_API_KEY=your_openai_api_key_here
   set GOOGLE_API_KEY=your_google_api_key_here
   ```

   **Option 4: Using Terminal (Linux/macOS)**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export GOOGLE_API_KEY="your_google_api_key_here"
   ```

4. **Install additional system dependencies (for OCR)**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr poppler-utils
   
   # macOS
   brew install tesseract poppler
   ```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit web interface**
   ```bash
   cd backend
   streamlit run ui.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

### Using the System

1. **Chat Interface**: Ask questions about insurance policies
   - "I am 30 years old, earning 8 lakhs per year. I want a retirement plan. What do you suggest?"
   - "Which is better between ICICI Pru LifeTime Classic and ICICI Future Perfect?"
   - "What are the benefits of the ICICI Pru Signature plan?"

2. **Upload Documents**: Upload your own insurance policy PDFs for analysis

3. **Get Recommendations**: Receive personalized policy recommendations

## 📁 Project Structure

```
insurance_rag/
├── backend/
│   ├── agents/                    # Multi-agent underwriting system
│   │   ├── __init__.py
│   │   ├── agent1_historical_patterns.py
│   │   ├── agent2_risk_assessment.py
│   │   ├── agent3_policy_generation.py
│   │   └── orchestrator.py
│   ├── icici_policies/            # Default policy documents
│   ├── uploaded_policies/         # User-uploaded documents
│   ├── main6.py                   # Core RAG implementation
│   ├── ui.py                     # Streamlit web interface
│   ├── requirements.txt           # Python dependencies
│   ├── prompts.txt               # Sample queries
│   ├── policy_list.json          # Default policy metadata
│   └── uploaded_policy_list.json # Uploaded policy metadata
└── README.md
```

## 🤖 Multi-Agent System

The system includes three specialized AI agents:

### Agent 1: Historical Risk Pattern Analysis
- Analyzes historical data for risk patterns
- Identifies similar customer profiles
- Provides risk correlation insights

### Agent 2: Actuarial Risk Assessment
- Performs comprehensive risk assessment
- Calculates risk scores and factors
- Provides underwriting recommendations

### Agent 3: Policy Generation
- Generates policy recommendations
- Customizes policy terms
- Calculates premium suggestions

## 📊 Sample Queries

The system can handle various types of insurance queries:

- Age-based recommendations
- Income-based policy matching
- Policy comparisons
- Benefit explanations
- Claim process information
- Critical illness coverage queries
- Children's education plans
- Retirement planning advice

## 🔧 Configuration

### Policy Documents
- Default policies are stored in `backend/icici_policies/`
- Upload custom policies through the web interface
- Supported formats: PDF

### Database
- Vector embeddings stored in ChromaDB
- Automatic document chunking and indexing
- Semantic search capabilities



## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the code files
- Review sample queries in `prompts.txt`

## 🎯 Future Enhancements

- Integration with more insurance providers
- Enhanced OCR capabilities
- Mobile-responsive interface
- Multi-language support
- Advanced analytics dashboard
- API endpoints for external integration

---

**Built with ❤️ using Python, LangChain, and AI**
