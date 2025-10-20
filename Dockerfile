FROM public.ecr.aws/lambda/python:3.11

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/

# Create temp directory
RUN mkdir -p /tmp/chroma_db

# Set handler
CMD ["lambda_handler.handler"]