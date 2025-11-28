"""
Medical Diagnosis Knowledge Assistant (MDKA)
Main Streamlit Application Interface
"""

import streamlit as st
import os
from rag_engine import MedicalRAGEngine

# Page Configuration
st.set_page_config(
    page_title="Medical Diagnosis Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTextArea textarea {
        border: 2px solid #4f46e5;
    }
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        background-color: #f0fdf4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    """Initialize RAG engine with API key"""
    try:
        api_key = st.session_state.get('gemini_api_key', '')
        if not api_key:
            return False
        
        st.session_state.rag_engine = MedicalRAGEngine(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

# New helper to clear inputs + chat history
def clear_all_inputs_and_history():
    """Clear chat history and all input widgets across tabs (but keep API key and rag engine)"""
    keys_to_clear = [
        "general_query",
        "uploaded_files",
        "patient_age",
        "patient_gender",
        "patient_history",
        "patient_symptoms",
        "patient_labs",
        "lab_data",
        "clinical_context"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    # Clear chat history
    st.session_state.chat_history = []
    # Rerun to reflect cleared UI
    st.experimental_rerun()

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Diagnosis Knowledge Assistant</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input (stored in session_state)
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key",
            key="gemini_api_key"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                if initialize_rag_system():
                    st.success("✅ System initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system")
        
        st.divider()
        
        # Document upload section (persisted to session_state)
        st.header("📚 Medical Literature")
        uploaded_files = st.file_uploader(
            "Upload medical documents (PDF)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload Kumar & Clark, PubMed papers, or clinical guidelines",
            key="uploaded_files"
        )
        
        if uploaded_files and st.session_state.rag_engine:
            if st.button("📥 Process Documents"):
                with st.spinner("Processing medical literature..."):
                    for file in uploaded_files:
                        try:
                            st.session_state.rag_engine.add_document(file)
                            st.success(f"✅ Processed: {file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        st.divider()
        
        # Clear history and inputs button
        if st.button("🗑️ Clear Chat History"):
            clear_all_inputs_and_history()
    
    # Main content area
    if not st.session_state.rag_engine:
        st.info("👈 Please enter your Gemini API key and initialize the system to begin.")
        
        # Display features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📖 Literature Analysis
            - Process medical textbooks
            - Analyze research papers
            - Extract clinical guidelines
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Clinical Reasoning
            - Structured diagnostic approach
            - Differential diagnosis
            - Evidence-based recommendations
            """)
        
        with col3:
            st.markdown("""
            ### 💡 Next Steps
            - Patient history analysis
            - Lab result interpretation
            - Treatment suggestions
            """)
    
    else:
        # Chat interface
        st.markdown("### 💬 Clinical Consultation")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input tabs
        tab1, tab2, tab3 = st.tabs(["💬 General Query", "🩺 Patient Case", "🔬 Lab Analysis"])
        
        with tab1:
            query = st.text_area(
                "Ask about medical conditions, treatments, or diagnoses:",
                height=100,
                placeholder="e.g., What are the diagnostic criteria for Type 2 Diabetes according to current guidelines?",
                key="general_query"
            )
            
            if st.button("🔍 Analyze", key="general"):
                if query:
                    process_query(query, "general")
        
        with tab2:
            st.markdown("#### Patient Information")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=45, key="patient_age")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender")
            
            with col2:
                history = st.text_area("Medical History", height=80, key="patient_history")
            
            symptoms = st.text_area(
                "Chief Complaints & Symptoms",
                height=100,
                placeholder="Describe symptoms, duration, severity...",
                key="patient_symptoms"
            )
            
            labs = st.text_area(
                "Lab Results (if any)",
                height=80,
                placeholder="Enter relevant lab values...",
                key="patient_labs"
            )
            
            if st.button("🩺 Generate Clinical Reasoning", key="patient"):
                patient_query = f"""
                Patient Case Analysis:
                - Age: {age}, Gender: {gender}
                - Medical History: {history if history else 'None reported'}
                - Symptoms: {symptoms}
                - Lab Results: {labs if labs else 'Pending'}
                
                Provide structured clinical reasoning with differential diagnosis and next steps.
                """
                process_query(patient_query, "patient_case")
        
        with tab3:
            lab_data = st.text_area(
                "Lab Report Data",
                height=200,
                placeholder="""Enter lab values, e.g.:
- Hemoglobin: 10.5 g/dL
- WBC: 12,000/μL
- Glucose (fasting): 145 mg/dL
- HbA1c: 7.2%
                """,
                key="lab_data"
            )
            
            clinical_context = st.text_area(
                "Clinical Context",
                height=80,
                placeholder="Brief clinical scenario or reason for testing...",
                key="clinical_context"
            )
            
            if st.button("🔬 Interpret Labs", key="labs"):
                lab_query = f"""
                Lab Report Interpretation:
                
                Lab Values:
                {lab_data}
                
                Clinical Context:
                {clinical_context if clinical_context else 'Not specified'}
                
                Provide interpretation, clinical significance, and recommendations.
                """
                process_query(lab_query, "lab_analysis")

def process_query(query: str, query_type: str):
    """Process user query through RAG engine"""
    try:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        
        # Get response from RAG engine
        with st.spinner("🔍 Analyzing medical literature and generating response..."):
            response = st.session_state.rag_engine.query(query)
        
        # Add assistant response to chat without HTML wrapper
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to display updated chat
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()