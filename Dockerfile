# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose port Streamlit runs on
EXPOSE 8080

RUN python -m spacy download en_core_web_sm

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]

