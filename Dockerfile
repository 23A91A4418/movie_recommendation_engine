FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project source files
COPY . .

# Expose port
EXPOSE 8000

# Default command (which will be overridden by docker-compose)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
