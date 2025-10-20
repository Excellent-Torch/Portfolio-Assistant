from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from app.services.s3_loader import S3DocumentLoader
from app.config import get_settings
import os

settings = get_settings()

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.loader = S3DocumentLoader()
    
    def initialize(self):
        """Initialize RAG pipeline"""
        print("🚀 Initializing RAG system...")
        
        # Load documents from S3
        documents = self.loader.load_documents()
        
        if not documents:
            raise Exception("No documents loaded from S3")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store
        persist_dir = "/tmp/chroma_db" if os.getenv('AWS_EXECUTION_ENV') else "./chroma_db"
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.model_name,
            temperature=0.3,
            max_tokens=1024
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
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        print("✅ RAG system initialized!")
    
    def get_chain(self):
        """Get QA chain"""
        if self.qa_chain is None:
            self.initialize()
        return self.qa_chain

# Global instance
rag_service = RAGService()