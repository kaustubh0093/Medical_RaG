"""
Medical RAG Engine with ChromaDB and Gemini 2.5 Flash
Handles document processing, embedding, and clinical reasoning
"""

import os
import tempfile
from typing import List, Dict, Any
from io import BytesIO

# Core dependencies
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class MedicalRAGEngine:
    """
    RAG system for medical diagnosis assistance using Gemini 2.5 Flash
    """
    
    def __init__(self, api_key: str, persist_directory: str = "./chroma_medical_db"):
        """
        Initialize Medical RAG Engine
        
        Args:
            api_key: Google Gemini API key
            persist_directory: Path to persist ChromaDB
        """
        self.api_key = api_key
        self.persist_directory = persist_directory
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Initialize LLM - Gemini 2.5 Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=8192,
            timeout=60
        )
        
        # Initialize text splitter for medical documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self._initialize_vectorstore()
        
        # Clinical reasoning prompt template
        self.prompt_template = self._create_clinical_prompt()
        
    def _initialize_vectorstore(self):
        """Initialize ChromaDB vector store"""
        try:
            if os.path.exists(self.persist_directory):
                # Load existing database
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                # Create new database
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {str(e)}")
    
    def _create_clinical_prompt(self) -> PromptTemplate:
        """
        Create structured clinical reasoning prompt template
        Following evidence-based medical practice
        """
        template = """You are an expert medical AI assistant trained in clinical reasoning and diagnosis. 
Use the provided medical literature context to answer the question with structured clinical reasoning.

Medical Literature Context:
{context}

Question: {question}

Instructions:
1. **Patient Information Summary**: Organize all relevant patient data systematically
2. **Clinical Reasoning Process**:
   - Identify key symptoms, signs, and risk factors
   - Apply clinical reasoning frameworks (analytical and intuitive)
   - Consider pathophysiology and evidence-based guidelines
3. **Differential Diagnosis**: List possible diagnoses in order of likelihood
4. **Diagnostic Workup**: Recommend appropriate investigations
5. **Clinical Recommendations**: Suggest evidence-based next steps
6. **References**: Cite specific medical literature when applicable

Provide a comprehensive, well-structured response following these steps.
Ensure all recommendations are based on current medical evidence.

Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def add_document(self, file) -> Dict[str, Any]:
        """
        Process and add medical document to vector store
        
        Args:
            file: Uploaded file object (PDF)
            
        Returns:
            Dictionary with processing status
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            # Load PDF document
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': file.name,
                    'chunk_id': i,
                    'doc_type': 'medical_literature'
                })
            
            # Add to vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Persist changes
            self.vectorstore.persist()
            
            # Cleanup
            os.unlink(tmp_path)
            
            return {
                'status': 'success',
                'filename': file.name,
                'chunks_created': len(chunks),
                'total_pages': len(documents)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'filename': file.name if file else 'unknown',
                'error': str(e)
            }
    
    def _is_medical_question(self, question: str) -> bool:
        """Check if the question is medical or health-related"""
        try:
            context_prompt = f"""Determine if this question is related to medicine, healthcare, or medical science.
Question: {question}
Return only 'true' or 'false'."""
            
            response = self.llm.invoke(context_prompt)
            return 'true' in response.content.lower()
        except Exception:
            return True  # Default to assuming it's medical if check fails
    
    def _query_without_context(self, question: str) -> str:
        """
        Fallback query method when no documents are loaded
        Uses Gemini's base medical knowledge with internal question analysis
        """
        try:
            # Check if question is medical-related
            if not self._is_medical_question(question):
                return ("⚠️ **Out of Context**: This question appears to be unrelated to medicine, healthcare, "
                       "or medical science. Please ask medical or health-related questions only.")
                
            fallback_prompt = f"""You are a medical AI assistant. Internally analyze the following medical question 
to determine its type, complexity, and required expertise level. Based on your analysis, provide an appropriate 
response without explicitly stating the analysis process.

Question: {question}

Instructions:
- For clinical questions, structure your response with relevant medical reasoning and considerations
- For general health questions, provide clear, direct answers with evidence-based guidance
- For administrative queries, give straightforward practical responses
- Adapt your response style and depth based on the question's complexity
- Include relevant medical context only when necessary

Note: Ensure responses follow evidence-based medical principles while maintaining appropriate scope.

Response:"""
            
            response = self.llm.invoke(fallback_prompt)
            return response.content + "\n\n⚠️ **Note**: No medical literature loaded. Response based on general medical knowledge only."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _extract_relevant_contexts(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """Extract relevant contexts with metadata from documents"""
        if not self.vectorstore:
            return []
            
        results = self.vectorstore.similarity_search_with_score(
            question,
            k=k
        )
        
        contexts = []
        for doc, score in results:
            contexts.append({
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown'),
                'relevance_score': float(score),
                'chunk_id': doc.metadata.get('chunk_id', 'Unknown')
            })
        
        return contexts

    def query(self, question: str, k: int = 5) -> str:
        """Query the RAG system with detailed context analysis"""
        try:
            # Check if question is medical-related
            if not self._is_medical_question(question):
                return ("⚠️ **Out of Context**: This question appears to be unrelated to medicine, healthcare, "
                       "or medical science. Please ask medical or health-related questions only.")
                       
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return self._query_without_context(question)
            
            # Get relevant contexts
            contexts = self._extract_relevant_contexts(question, k)
            if not contexts:
                return self._query_without_context(question)
            
            # Format contexts for prompt
            context_text = "\n\n".join([f"Source: {ctx['source']}\nContent: {ctx['content']}" 
                                      for ctx in contexts])
            
            # Create enhanced prompt
            enhanced_prompt = f"""Analyze the following medical question using the provided document contexts.

Question: {question}

Relevant Document Contexts:
{context_text}

Please provide:
1. Direct answers found in the documents
2. Key points from each relevant source
3. Synthesis of information
4. Additional considerations

Response format:
### 📑 Document Analysis
[List key findings from documents]

### 🔍 Detailed Answer
[Comprehensive response]

### 📚 Source Analysis
[Break down of information by source]

Response:"""
            
            # Execute query with enhanced prompt
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
                chain_type_kwargs={"prompt": PromptTemplate(template=enhanced_prompt, 
                                                         input_variables=["question"])},
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            response = result['result']
            
            # Add source summary
            source_summary = "\n\n### 📚 Document Sources:\n"
            seen_sources = set()
            for ctx in contexts:
                source = ctx['source']
                if source not in seen_sources:
                    relevance = round((1 - ctx['relevance_score']) * 100, 2)
                    source_summary += f"- {source} (Relevance: {relevance}%)\n"
                    seen_sources.add(source)
            
            return response + source_summary
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            if self.vectorstore is None:
                return {'status': 'not_initialized'}
            
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                'status': 'initialized',
                'total_chunks': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def clear_database(self):
        """Clear all documents from vector database"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self._initialize_vectorstore()
            return {'status': 'success', 'message': 'Database cleared'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}