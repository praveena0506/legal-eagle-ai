# 1. Use a newer Airflow version with Python 3.10
FROM apache/airflow:2.9.2-python3.10

USER root
USER airflow

# 2. Install dependencies
# We keep pysqlite3-binary because Chroma still needs the newer SQLite
RUN pip install --no-cache-dir \
    chromadb \
    pandas \
    sqlalchemy \
    psycopg2-binary \
    pysqlite3-binary