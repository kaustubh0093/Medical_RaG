# 🏥 Medical Diagnosis Knowledge Assistant (MDKA)

An AI-powered clinical reasoning assistant that analyzes medical literature and provides structured diagnostic support using Google's Gemini 2.5 Flash model.

## 🎯 Features

- **📚 Medical Literature Analysis**: Process textbooks (Kumar & Clark), PubMed papers, and clinical guidelines
- **🩺 Clinical Reasoning**: Structured diagnostic approach with differential diagnosis
- **💬 Interactive Interface**: User-friendly Streamlit UI for medical consultations
- **🔍 RAG Architecture**: Retrieval-Augmented Generation for evidence-based responses
- **🧠 Smart Embeddings**: ChromaDB vector database for semantic medical literature search

## 🏗️ Architecture

```
├── app.py              # Streamlit interface
├── rag_engine.py       # RAG system with Gemini 2.5 Flash
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

**Tech Stack:**
- **LLM**: Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **Frontend**: Streamlit
- **Document Processing**: PyPDF

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Setup Steps

1. **Clone or download the project**
```bash
cd medical-diagnosis-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key** (Optional - can be entered in UI)
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## 💻 Usage

### Start the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the System

1. **Initialize System**
   - Enter your Gemini API key in the sidebar
   - Click "Initialize System"

2. **Upload Medical Literature** (Optional)
   - Upload PDF files (Kumar & Clark, research papers, guidelines)
   - Click "Process Documents"

3. **Query the System**
   - **General Query**: Ask about medical conditions and treatments
   - **Patient Case**: Enter patient details for clinical reasoning
   - **Lab Analysis**: Interpret laboratory results

### Example Queries

**General Medical Query:**
```
What are the diagnostic criteria for Type 2 Diabetes according to current guidelines?
```

**Patient Case:**
```
Age: 55, Male
History: Hypertension, family history of diabetes
Symptoms: Polyuria, polydipsia, fatigue for 3 months
Labs: Fasting glucose 156 mg/dL, HbA1c 7.8%
```

**Lab Interpretation:**
```
Hemoglobin: 10.2 g/dL
MCV: 72 fL
Ferritin: 8 ng/mL
Clinical Context: 45-year-old female with fatigue
```

## 📖 System Capabilities

### Clinical Reasoning Process
1. **Patient Information Summary**: Systematic data organization
2. **Clinical Analysis**: Evidence-based reasoning
3. **Differential Diagnosis**: Likelihood-ranked possibilities
4. **Diagnostic Workup**: Recommended investigations
5. **Management Plan**: Evidence-based treatment suggestions
6. **Source References**: Citations from medical literature

### Supported Documents
- Medical textbooks (Kumar & Clark's Clinical Medicine)
- PubMed research papers
- Clinical practice guidelines
- Review articles and case studies

## ⚙️ Configuration

### Gemini Model Parameters
The system uses Gemini 2.5 Flash with:
- **Temperature**: 0.3 (balanced creativity and accuracy)
- **Max Tokens**: 8192 (comprehensive responses)
- **Context Window**: 1M tokens

### RAG Configuration
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 5 most relevant chunks
- **Embeddings**: Google's embedding-001 model

## 🔒 Security & Privacy

- API keys are not stored persistently
- Medical data processed locally
- No patient information sent to external servers
- Documents stored in local ChromaDB instance

## 🐛 Troubleshooting

### Common Issues

**"Failed to initialize system"**
- Verify your Gemini API key is correct
- Check internet connection
- Ensure all dependencies are installed

**"Error processing document"**
- Ensure PDF files are not corrupted
- Check file size (large files may take time)
- Verify PDF is text-based (not scanned images)

**Slow response times**
- First query initializes the system (slower)
- Large documents take time to process
- API rate limits may apply

## 📊 Performance Notes

- **First Query**: 3-5 seconds (system initialization)
- **Subsequent Queries**: 1-2 seconds
- **Document Processing**: 10-30 seconds per PDF
- **Memory Usage**: ~500MB for vector database

## 🔄 Updates & Maintenance

### Clear Vector Database
```python
# In the application, use the sidebar option
# Or manually delete: ./chroma_medical_db/
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## ⚠️ Important Disclaimers

1. **Not a Replacement for Medical Professionals**: This tool is for educational and informational purposes only
2. **Clinical Validation Required**: All outputs should be verified by qualified healthcare professionals
3. **No Patient Care Decisions**: Do not use for direct patient care without physician oversight
4. **Evidence Quality**: Responses depend on uploaded literature quality
5. **Regulatory Compliance**: Ensure compliance with local healthcare regulations

## 🤝 Contributing

This is a demonstration project. For production use:
- Add user authentication
- Implement audit logging
- Add HIPAA compliance measures
- Include model version control
- Add comprehensive testing

## 📚 References

- [Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Kumar & Clark's Clinical Medicine](https://www.elsevier.com/books/kumar-and-clarks-clinical-medicine/)

## 📝 License

This project is for educational purposes. Ensure compliance with:
- Medical device regulations in your jurisdiction
- Data protection laws (GDPR, HIPAA, etc.)
- Professional medical practice standards

## 👨‍💻 Development

**Built with:**
- Python 3.9+
- Gemini 2.5 Flash
- LangChain framework
- ChromaDB vector database
- Streamlit web framework

**Version**: 1.0.0

---

**⚕️ For Educational and Research Purposes Only**

*Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment.*
