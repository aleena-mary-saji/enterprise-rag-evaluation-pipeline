# Use official lightweight Python parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling certain packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and app folders
COPY config/ /app/config/
COPY src/ /app/src/
COPY app/ /app/app/
COPY tests/ /app/tests/

# Expose default ports: FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Run the Streamlit application as the default entrypoint
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
