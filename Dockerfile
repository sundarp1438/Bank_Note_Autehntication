# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY banknote_ann_api.py params.yaml models/banknote_ann_model.h5 ./

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "banknote_ann_api:app", "--host", "0.0.0.0", "--port", "8000"]
