# Use Ubuntu base image for CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python & dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app

# Expose API port
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
