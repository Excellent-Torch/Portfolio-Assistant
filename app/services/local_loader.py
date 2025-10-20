import os
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader
)
from typing import List
from app.config import get_settings

settings = get_settings()

class LocalDocumentLoader:
    """Load documents from local filesystem"""
    
    def __init__(self, documents_path: str = None):
        self.documents_path = documents_path or settings.documents_path
        
    def load_documents(self) -> List[Document]:
        """Load all documents from local directory"""
        documents = []
        
        # Check if directory exists
        if not os.path.exists(self.documents_path):
            print(f"⚠️  Directory not found: {self.documents_path}")
            print(f"Creating directory...")
            os.makedirs(self.documents_path, exist_ok=True)
            return documents
        
        # Load text files
        print(f"📁 Loading documents from: {self.documents_path}")
        
        try:
            # Load .txt files
            txt_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"✅ Loaded {len(txt_docs)} .txt files")

            # Load .pdf files
            pdf_loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"✅ Loaded {len(pdf_docs)} .pdf files")
            
            print(f"📄 Total documents loaded: {len(documents)}")
            
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
        
        return documents