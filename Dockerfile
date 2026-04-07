FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/

# Create templates directory
RUN mkdir -p src/templates

# Expose port
EXPOSE 7860

# Start the app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
```
