FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory
RUN mkdir -p ~/.streamlit

# Streamlit config
RUN echo '[server]\nheadless = true\nport = 8501\nenableCORS = false\n[logger]\nlevel = info' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
