from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from app.services.local_loader import LocalDocumentLoader
from app.config import get_settings
import os

settings = get_settings()

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.loader = LocalDocumentLoader()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize RAG pipeline"""
        print("🚀 Initializing RAG system...")
        print(f"Environment: {settings.environment}")
        print(f"Documents path: {settings.documents_path}")
        
        # Load documents
        documents = self.loader.load_documents()
        
        if not documents:
            print("⚠️  No documents found!")
            print(f"Please add documents to: {settings.documents_path}")
            raise Exception("No documents loaded")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings
        print("🔧 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        print("💾 Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=settings.chroma_db_path
        )
        print(f"✅ Vector store created at: {settings.chroma_db_path}")
        
        # Initialize LLM
        print("🤖 Initializing Groq LLM...")
        llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        # Create prompt
        prompt_template = """You are a knowledgeable AI assistant for a professional portfolio website. 
You have access to information about the portfolio owner's projects, skills, experience, and resume.

Use the following context to answer questions accurately and professionally. 
If you're not sure about something, say so rather than making up information.

Context: {context}

Chat History: {chat_history}

Question: {question}

Provide a helpful, professional response that showcases the portfolio owner's qualifications and work:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create QA chain
        print("⛓️  Creating QA chain...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True  # Enable for debugging
        )
        
        self.is_initialized = True
        print("✅ RAG system initialized successfully!")
    
    def get_chain(self):
        """Get QA chain, initialize if needed"""
        if not self.is_initialized:
            self.initialize()
        return self.qa_chain

# Global instance
rag_service = RAGService()