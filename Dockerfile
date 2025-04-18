FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for XGBoost optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

# Create data directory
RUN mkdir -p data

# Copy application code
COPY app.py .
COPY data/encoded_data.csv data/encoded_data.csv

# Copy streamlit config
COPY .streamlit .streamlit

EXPOSE 8501

# Set environment variables for better performance
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV OMP_NUM_THREADS=4
ENV NUMEXPR_MAX_THREADS=4
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
