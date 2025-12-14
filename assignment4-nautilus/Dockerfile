FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./

# Copy the handbook PDF
COPY cpsc-handbook-2022.pdf ./

# Create directories for outputs
RUN mkdir -p /data/models /data/results

# Set environment variables
ENV TRANSFORMERS_CACHE=/data/models
ENV HF_HOME=/data/models

# Default command (will be overridden by k8s job)
CMD ["python", "main.py"]
