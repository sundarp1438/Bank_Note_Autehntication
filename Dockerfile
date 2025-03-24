# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements.txt setup.py ./

# Copy source code
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Install the package
RUN pip install -e .

# Copy the trained model (this will be mounted in production)
COPY model.joblib ./

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"] 
