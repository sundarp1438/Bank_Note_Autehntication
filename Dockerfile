# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the required files
COPY banknote_ann_api.py params.yaml models/banknote_ann_model.h5 requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "banknote_ann_api:app", "--host", "0.0.0.0", "--port", "8000"]
