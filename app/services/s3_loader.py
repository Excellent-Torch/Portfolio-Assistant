import boto3
from langchain.schema import Document
from typing import List
from app.config import get_settings

settings = get_settings()

class S3DocumentLoader:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=settings.aws_region)
        self.bucket_name = settings.s3_bucket_name
    
    def load_documents(self) -> List[Document]:
        """Load all documents from S3 bucket"""
        documents = []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='documents/'
            )
            
            if 'Contents' not in response:
                print("No documents found in S3")
                return documents
            
            for obj in response['Contents']:
                key = obj['Key']
                
                if key.endswith('/'):
                    continue
                
                file_obj = self.s3_client.get_object(
                    Bucket=self.bucket_name, 
                    Key=key
                )
                content = file_obj['Body'].read().decode('utf-8')
                
                doc = Document(
                    page_content=content,
                    metadata={"source": key.split('/')[-1]}
                )
                documents.append(doc)
                print(f"✅ Loaded: {key}")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error loading documents: {str(e)}")
            return documents