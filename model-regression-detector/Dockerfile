FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create data and reports directories
RUN mkdir -p /app/data /app/reports

# Default entrypoint: run the evaluation pipeline
ENTRYPOINT ["python", "scripts/run_eval.py"]

# Default arguments (can be overridden)
CMD []
