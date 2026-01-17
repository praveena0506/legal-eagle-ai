# Start from the official Airflow image
FROM apache/airflow:2.7.1

# Switch to root to install system tools
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git && \
    apt-get clean

# Switch back to airflow user
USER airflow

# STEP 1: Install PyTorch CPU version (Lightweight & Fast)
# We use a specific URL to get the small version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install everything else
RUN pip install --no-cache-dir \
    chromadb \
    pandas \
    sqlalchemy \
    psycopg2-binary \
    pysqlite3-binary \
    requests \
    beautifulsoup4 \
    fastapi \
    uvicorn \
    pydantic