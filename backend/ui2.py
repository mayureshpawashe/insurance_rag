import streamlit as st
import os
import json
import uuid
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set page config first (must be the first Streamlit command)
st.set_page_config(
    page_title="🏥 Insurance Policy Advisor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import from main_new6 with error handling
try:
    from main_new6 import (
        process_message,
        process_uploaded_pdf,
        get_or_create_session,
        get_underwriting_result,
        smart_underwriting_system,
        SMART_UNDERWRITING_AVAILABLE,
        PDF_FOLDER,
        UPLOADS_DIR,
        get_agent_status,
        policy_list,
        uploaded_policy_list
    )
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except Exception as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)
    # Fallback - create dummy functions to prevent crashes
    def process_message(user_id, message, turn): return "Import error - system not available"
    def process_uploaded_pdf(path, user_id): return type('Policy', (), {'policy_name': 'Error: System not available'})()
    def get_or_create_session(user_id): return type('Profile', (), {'user_id': user_id})()
    def get_underwriting_result(user_id, needs, policy_type): return {"error": "System not available"}
    smart_underwriting_system = None
    SMART_UNDERWRITING_AVAILABLE = False
    PDF_FOLDER = "./icici_policies"
    UPLOADS_DIR = "./uploaded_policies"
    policy_list = []
    uploaded_policy_list = []

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 85%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-active {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-inactive {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'user_id': str(uuid.uuid4()),
        'conversation_turn': 0,
        'chat_history': [],
        'uploaded_files': [],
        'underwriting_results': [],
        'session_start_time': datetime.now(),
        'total_queries': 0,
        'system_status_checked': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Enhanced message display
def display_enhanced_chat_message(role: str, content: str, timestamp: datetime = None):
    """Display chat messages with enhanced formatting"""
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%H:%M:%S")
    
    if role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem;">🤖</span>
                <strong style="margin-left: 0.5rem;">AI Assistant</strong>
                <span style="margin-left: auto; font-size: 0.8rem; color: #666;">{timestamp_str}</span>
            </div>
            <div style="line-height: 1.6;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem;">👤</span>
                <strong style="margin-left: 0.5rem;">You</strong>
                <span style="margin-left: auto; font-size: 0.8rem;">{timestamp_str}</span>
            </div>
            <div style="line-height: 1.6;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "error":
        st.error(f"❌ {content}")

# System status check
def check_system_status():
    """Check and display system status"""
    if not st.session_state.system_status_checked:
        st.session_state.system_status_checked = True
        
        if not IMPORTS_SUCCESSFUL:
            st.error(f"❌ **System Error:** {IMPORT_ERROR}")
            st.info("🔧 **Troubleshooting Steps:**\n1. Check if main_new6.py exists\n2. Verify all dependencies are installed\n3. Check Python environment")
            return False
        
        return True
    return IMPORTS_SUCCESSFUL

# Enhanced sidebar
def enhanced_sidebar():
    """Enhanced sidebar with system information"""
    with st.sidebar:
        st.markdown('<div class="main-header"><h2>🏥 Insurance Advisor</h2></div>', unsafe_allow_html=True)
        
        # System Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔧 System Status")
        
        if IMPORTS_SUCCESSFUL:
            if SMART_UNDERWRITING_AVAILABLE and smart_underwriting_system:
                st.markdown('<div class="status-active">✅ <strong>All Systems Online</strong></div>', unsafe_allow_html=True)
                st.markdown("""
                **Features Available:**
                - ✅ Smart Underwriting
                - ✅ Chat Assistant  
                - ✅ Policy Upload
                - ✅ Risk Assessment
                """)
            else:
                st.markdown('<div class="status-inactive">⚠️ <strong>Limited Functionality</strong></div>', unsafe_allow_html=True)
                st.markdown("Smart Underwriting system is not available")
        else:
            st.markdown('<div class="status-inactive">❌ <strong>System Error</strong></div>', unsafe_allow_html=True)
            st.markdown("Unable to load main system components")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Session Information
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Session Info")
        
        session_duration = datetime.now() - st.session_state.session_start_time
        st.markdown(f"""
        **Session ID:** `{st.session_state.user_id[-8:]}`  
        **Duration:** {str(session_duration).split('.')[0]}  
        **Queries:** {st.session_state.total_queries}  
        **Files:** {len(st.session_state.uploaded_files)}  
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🧭 Navigation")
        
        pages = ["💬 Chat", "🎯 Smart Underwriting", "📄 Upload Policy", "📊 Dashboard"]
        page = st.radio("Select Page:", pages, label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh System", use_container_width=True):
            st.session_state.system_status_checked = False
            st.rerun()
        
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        if st.button("📊 System Info", use_container_width=True):
            st.session_state.show_system_info = True
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("**Version:** 2.0.0")
        st.markdown("**Status:** " + ("🟢 Online" if IMPORTS_SUCCESSFUL else "🔴 Offline"))
        
        return page

# Chat interface
def chat_interface():
    """Main chat interface"""
    st.markdown('<div class="main-header"><h1>💬 Chat with Insurance Advisor</h1></div>', unsafe_allow_html=True)
    
    if not IMPORTS_SUCCESSFUL:
        st.error("❌ Chat system is not available due to import errors")
        st.code(f"Error: {IMPORT_ERROR}")
        return
    
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Messages:** {len(st.session_state.chat_history)} | **Session:** {st.session_state.user_id[-8:]}")
    
    with col2:
        if st.button("💾 Save Chat"):
            st.success("💾 Chat saved locally!")
    
    with col3:
        if st.button("🧹 Clear All"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                display_enhanced_chat_message(
                    message["role"], 
                    message["content"],
                    message.get("timestamp", datetime.now())
                )
        else:
            st.info("👋 Welcome! Start a conversation by typing a message below.")
            
            # Quick start suggestions
            st.markdown("### 💡 Try asking:")
            suggestions = [
                "What health insurance policies do you offer?",
                "Compare term life vs whole life insurance",
                "I need insurance for a family of 4",
                "What are the benefits of ICICI policies?"
            ]
            
            col1, col2 = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with col1 if i % 2 == 0 else col2:
                    if st.button(f"💭 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                        # Process the suggestion
                        process_user_input(suggestion)
    
    # Chat input
    user_input = st.chat_input("💬 Ask me anything about insurance...")
    
    if user_input:
        process_user_input(user_input)

def process_user_input(user_input):
    """Process user input and generate response"""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user", 
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    st.session_state.total_queries += 1
    
    # Process message
    with st.spinner("🤖 Thinking..."):
        try:
            start_time = time.time()
            response = process_message(
                st.session_state.user_id,
                user_input,
                st.session_state.conversation_turn
            )
            processing_time = time.time() - start_time
            
            # Add response
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now(),
                "processing_time": processing_time
            })
            
            st.session_state.conversation_turn += 1
            st.rerun()
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            st.session_state.chat_history.append({
                "role": "error", 
                "content": error_msg,
                "timestamp": datetime.now()
            })
            st.rerun()

# Smart Underwriting interface
def smart_underwriting_interface():
    """Smart underwriting interface"""
    st.markdown('<div class="main-header"><h1>🎯 Smart Underwriting System</h1></div>', unsafe_allow_html=True)
    
    if not IMPORTS_SUCCESSFUL:
        st.error("❌ Smart Underwriting system is not available due to import errors")
        return
    
    if not (SMART_UNDERWRITING_AVAILABLE and smart_underwriting_system):
        st.warning("⚠️ Smart Underwriting system is currently offline")
        st.info("💡 You can still use the basic chat interface for insurance questions")
        return
    
    # System status
    st.success("✅ Smart Underwriting System is online and ready!")
    
    # Underwriting form
    with st.form("underwriting_form", clear_on_submit=False):
        st.markdown("### 📋 Insurance Application")
        
        # Personal Information
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            occupation = st.selectbox("Occupation", [
                "Software Engineer", "Teacher", "Doctor", "Engineer", 
                "Business Owner", "Government Employee", "Other"
            ])
        
        with col2:
            annual_income = st.selectbox("Annual Income", [
                "< ₹3 Lakhs", "₹3-5 Lakhs", "₹5-10 Lakhs", 
                "₹10-20 Lakhs", "> ₹20 Lakhs"
            ])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Other"])
        
        # Insurance Requirements
        st.markdown("### 🏥 Insurance Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            policy_type = st.selectbox("Policy Type", [
                "Health Insurance", "Life Insurance", "Term Insurance", "Motor Insurance"
            ])
            coverage_amount = st.selectbox("Coverage Amount", [
                "₹2-5 Lakhs", "₹5-10 Lakhs", "₹10-25 Lakhs", "₹25-50 Lakhs", "> ₹50 Lakhs"
            ])
        
        with col2:
            premium_budget = st.selectbox("Premium Budget", [
                "< ₹10,000", "₹10,000-25,000", "₹25,000-50,000", "> ₹50,000"
            ])
        
        # Detailed needs
        customer_needs = st.text_area(
            "Describe your specific insurance needs:",
            height=100,
            placeholder="e.g., Family coverage, pre-existing conditions, preferred hospitals..."
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Get Underwriting Decision", use_container_width=True)
        
        if submitted and customer_needs:
            # Enhanced customer profile
            enhanced_needs = f"""
            Personal Profile:
            - Age: {age} years
            - Occupation: {occupation}
            - Annual Income: {annual_income}
            - Marital Status: {marital_status}
            
            Insurance Requirements:
            - Policy Type: {policy_type}
            - Coverage: {coverage_amount}
            - Budget: {premium_budget}
            
            Specific Needs: {customer_needs}
            """
            
            with st.spinner("🔄 Processing underwriting request..."):
                try:
                    start_time = time.time()
                    result = get_underwriting_result(
                        st.session_state.user_id,
                        enhanced_needs,
                        policy_type
                    )
                    processing_time = time.time() - start_time
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        return
                    
                    # Store result
                    result['processing_time'] = processing_time
                    result['customer_profile'] = {
                        'age': age, 'occupation': occupation, 
                        'income': annual_income, 'marital_status': marital_status
                    }
                    st.session_state.underwriting_results.append(result)
                    
                    # Display results
                    display_underwriting_results(result)
                    
                except Exception as e:
                    st.error(f"❌ Underwriting error: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

def display_underwriting_results(result):
    """Display underwriting results"""
    st.markdown("---")
    st.markdown("## 📊 Underwriting Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        decision = result['underwriting_decision']['decision']
        if decision == "APPROVED":
            st.success(f"✅ {decision}")
        elif decision == "DECLINED":
            st.error(f"❌ {decision}")
        else:
            st.warning(f"⚠️ {decision}")
    
    with col2:
        risk_level = result['risk_assessment']['overall_risk_level']
        risk_score = result['risk_assessment']['risk_score']
        st.metric("Risk Level", risk_level.upper(), f"Score: {risk_score:.1f}/100")
    
    with col3:
        premium_multiplier = result['risk_assessment']['recommended_premium_multiplier']
        st.metric("Premium Multiplier", f"{premium_multiplier:.2f}x")
    
    with col4:
        processing_time = result.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Detailed results
    tab1, tab2, tab3 = st.tabs(["📋 Decision Details", "📊 Risk Analysis", "🏥 Policy Details"])
    
    with tab1:
        st.markdown("### 🎯 Underwriting Decision")
        decision_data = result['underwriting_decision']
        
        st.write(f"**Decision:** {decision_data['decision']}")
        st.write(f"**Reason:** {decision_data['decision_reason']}")
        st.write(f"**Confidence:** {decision_data['confidence_level']:.1%}")
        
        if decision_data.get('manual_review_triggers'):
            st.markdown("**Manual Review Triggers:**")
            for trigger in decision_data['manual_review_triggers']:
                st.write(f"• {trigger}")
    
    with tab2:
        st.markdown("### 📊 Risk Assessment")
        risk_data = result['risk_assessment']
        
        st.write(f"**Overall Risk Level:** {risk_data['overall_risk_level']}")
        st.write(f"**Risk Score:** {risk_data['risk_score']:.1f}/100")
        st.write(f"**Premium Multiplier:** {risk_data['recommended_premium_multiplier']:.2f}x")
        st.write(f"**Confidence Level:** {risk_data['confidence_level']:.1%}")
        st.write(f"**Manual Review Required:** {'Yes' if risk_data['requires_manual_review'] else 'No'}")
    
    with tab3:
        st.markdown("### 🏥 Policy Details")
        policy_data = result['policy_draft']
        
        if policy_data['policy_name'] != "No policy generated":
            st.write(f"**Policy Name:** {policy_data['policy_name']}")
            st.write(f"**Policy Type:** {policy_data['policy_type']}")
            st.write(f"**Coverage Amount:** ₹{policy_data['coverage_amount']:,.2f}")
            st.write(f"**Annual Premium:** ₹{policy_data['premium_amount']:,.2f}")
            st.write(f"**Monthly Premium:** ₹{policy_data['premium_amount']/12:,.2f}")
            st.write(f"**Compliance Status:** {policy_data['compliance_status']}")
        else:
            st.warning("⚠️ No policy was generated based on the assessment")

# File upload interface
def upload_interface():
    """File upload interface"""
    st.markdown('<div class="main-header"><h1>📄 Upload Insurance Policies</h1></div>', unsafe_allow_html=True)
    
    if not IMPORTS_SUCCESSFUL:
        st.error("❌ File upload system is not available due to import errors")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload insurance policy documents in PDF format"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📁 Selected: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        with col2:
            if st.button("🚀 Process File", use_container_width=True):
                with st.spinner("🔄 Processing file..."):
                    try:
                        # Save file temporarily
                        temp_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                        os.makedirs(UPLOADS_DIR, exist_ok=True)
                        
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the file
                        start_time = time.time()
                        policy = process_uploaded_pdf(temp_path, st.session_state.user_id)
                        processing_time = time.time() - start_time
                        
                        if not policy.policy_name.startswith("Error"):
                            st.success(f"✅ Successfully processed: {policy.policy_name}")
                            
                            # Store file info
                            st.session_state.uploaded_files.append({
                                'filename': uploaded_file.name,
                                'policy_name': policy.policy_name,
                                'policy_type': policy.policy_type,
                                'upload_time': datetime.now(),
                                'processing_time': processing_time,
                                'status': 'success'
                            })
                            
                            # Auto-analyze the policy
                            st.markdown("### 🔍 Policy Analysis")
                            analysis_query = f"Tell me about the {policy.policy_name} policy I just uploaded"
                            
                            with st.spinner("Analyzing policy..."):
                                try:
                                    response = process_message(
                                        st.session_state.user_id,
                                        analysis_query,
                                        st.session_state.conversation_turn
                                    )
                                    st.markdown("#### 📊 Analysis Results:")
                                    st.markdown(response)
                                except Exception as e:
                                    st.error(f"Analysis failed: {str(e)}")
                        
                        else:
                            st.error(f"❌ Error processing file: {policy.policy_name}")
                            
                    except Exception as e:
                        st.error(f"❌ Upload failed: {str(e)}")
    
    # Show upload history
    if st.session_state.uploaded_files:
        st.markdown("### 📁 Upload History")
        
        for i, file_info in enumerate(st.session_state.uploaded_files):
            with st.expander(f"📄 {file_info['filename']} - {file_info['status'].title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Policy Name:** {file_info.get('policy_name', 'N/A')}")
                    st.write(f"**Policy Type:** {file_info.get('policy_type', 'Unknown')}")
                    st.write(f"**Status:** {file_info['status'].title()}")
                
                with col2:
                    st.write(f"**Upload Time:** {file_info['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Processing Time:** {file_info.get('processing_time', 0):.2f}s")

# Dashboard interface
def dashboard_interface():
    """Dashboard interface"""
    st.markdown('<div class="main-header"><h1>📊 System Dashboard</h1></div>', unsafe_allow_html=True)
    
    # System overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Status", "🟢 Online" if IMPORTS_SUCCESSFUL else "🔴 Offline")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Session Duration", str(datetime.now() - st.session_state.session_start_time).split('.')[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Queries", st.session_state.total_queries)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Files Processed", len(st.session_state.uploaded_files))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed information
    tab1, tab2, tab3 = st.tabs(["🔧 System Info", "📊 Statistics", "🔍 Debug Info"])
    
    with tab1:
        st.markdown("### 🔧 System Information")
        
        if IMPORTS_SUCCESSFUL:
            st.success("✅ All system components loaded successfully")
            
            st.markdown("#### 📚 Available Policies")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Standard Policies:** {len(policy_list)}")
                if policy_list:
                    with st.expander("View Standard Policies"):
                        for policy in policy_list[:5]:  # Show first 5
                            st.write(f"• {policy.get('policy_name', 'Unknown')}")
            
            with col2:
                st.write(f"**Uploaded Policies:** {len(uploaded_policy_list)}")
                if uploaded_policy_list:
                    with st.expander("View Uploaded Policies"):
                        for policy in uploaded_policy_list[:5]:  # Show first 5
                            st.write(f"• {policy.get('policy_name', 'Unknown')}")
            
            st.markdown("#### 🤖 Smart Underwriting")
            if SMART_UNDERWRITING_AVAILABLE and smart_underwriting_system:
                st.success("✅ Smart Underwriting System is active")
                try:
                    agent_status = get_agent_status()
                    st.write(f"**Version:** {agent_status.get('version', 'Unknown')}")
                    st.write(f"**Status:** {agent_status.get('status', 'Unknown')}")
                except:
                    st.write("**Status:** Active (details unavailable)")
            else:
                st.warning("⚠️ Smart Underwriting System is not available")
        else:
            st.error("❌ System components failed to load")
            st.code(f"Import Error: {IMPORT_ERROR}")
    
    with tab2:
        st.markdown("### 📊 Usage Statistics")
        
        if st.session_state.chat_history:
            st.write(f"**Total Messages:** {len(st.session_state.chat_history)}")
            
            # Message type breakdown
            user_messages = sum(1 for msg in st.session_state.chat_history if msg['role'] == 'user')
            assistant_messages = sum(1 for msg in st.session_state.chat_history if msg['role'] == 'assistant')
            
            st.write(f"**User Messages:** {user_messages}")
            st.write(f"**Assistant Responses:** {assistant_messages}")
        
        if st.session_state.underwriting_results:
            st.write(f"**Underwriting Requests:** {len(st.session_state.underwriting_results)}")
            
            # Decision breakdown
            decisions = [r['underwriting_decision']['decision'] for r in st.session_state.underwriting_results]
            approved = decisions.count('APPROVED')
            declined = decisions.count('DECLINED')
            manual_review = decisions.count('MANUAL_REVIEW')
            
            st.write(f"**Approved:** {approved}")
            st.write(f"**Declined:** {declined}")
            st.write(f"**Manual Review:** {manual_review}")
    
    with tab3:
        st.markdown("### 🔍 Debug Information")
        
        st.markdown("#### 🐍 Python Environment")
        st.code(f"""
System Status: {'OK' if IMPORTS_SUCCESSFUL else 'ERROR'}
Import Error: {IMPORT_ERROR if IMPORT_ERROR else 'None'}
Session ID: {st.session_state.user_id}
Start Time: {st.session_state.session_start_time}
        """)
        
        st.markdown("#### 📁 Directory Status")
        directories = [PDF_FOLDER, UPLOADS_DIR]
        for directory in directories:
            exists = os.path.exists(directory)
            st.write(f"**{directory}:** {'✅ Exists' if exists else '❌ Missing'}")
            if exists:
                try:
                    file_count = len([f for f in os.listdir(directory) if f.endswith('.pdf')])
                    st.write(f"  └── PDF Files: {file_count}")
                except:
                    st.write(f"  └── Cannot read directory")
        
        if st.button("🔄 Refresh Debug Info"):
            st.rerun()

# Main application
def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Check system status
    system_ok = check_system_status()
    
    # Create directories
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(PDF_FOLDER, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create directories: {str(e)}")
    
    # Initialize user session if system is OK
    if system_ok:
        try:
            user_profile = get_or_create_session(st.session_state.user_id)
        except Exception as e:
            st.error(f"Failed to initialize user session: {str(e)}")
    
    # Enhanced sidebar and page routing
    selected_page = enhanced_sidebar()
    
    # Route to appropriate page
    if selected_page == "💬 Chat":
        chat_interface()
    elif selected_page == "🎯 Smart Underwriting":
        smart_underwriting_interface()
    elif selected_page == "📄 Upload Policy":
        upload_interface()
    elif selected_page == "📊 Dashboard":
        dashboard_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        f"🏥 Advanced Insurance Policy Advisor v2.0 | "
        f"Session: {st.session_state.user_id[-8:]} | "
        f"Status: {'🟢 Online' if IMPORTS_SUCCESSFUL else '🔴 Offline'}"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()