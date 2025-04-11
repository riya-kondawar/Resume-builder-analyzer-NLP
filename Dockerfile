# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 🔥 Download spaCy model inside the image
RUN python -m spacy download en_core_web_sm

# 🔗 Link model (only needed if your code tries to load it by name)
RUN python -m spacy link en_core_web_sm en_core_web_sm

# Copy app source code
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Start the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.enableCORS=false"]
