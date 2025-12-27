import os
import sys

# --- CHROMA DB HACK FOR AIRFLOW ---
# This forces the system to use the new pysqlite3 library instead of the old system one.
# This MUST be done before importing chromadb.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------

import re
import uuid
from datetime import datetime
import chromadb
from sqlalchemy import create_engine, text
import pandas as pd

# --- CONFIGURATION ---

# Check if we are running inside Docker (Airflow) or locally
IS_DOCKER = os.path.exists('/.dockerenv') or os.getenv('AIRFLOW_HOME')

# Define Hosts based on where we are
DB_HOST = "postgres" if IS_DOCKER else "localhost"
CHROMA_HOST = "chromadb" if IS_DOCKER else "localhost"

print(f"🕵️ Detected Environment: {'DOCKER/AIRFLOW' if IS_DOCKER else 'LOCAL LAPTOP'}")
print(f"🔌 Connecting to: Postgres@{DB_HOST}, Chroma@{CHROMA_HOST}")

# 1. Connect to PostgreSQL
PG_CONNECTION = f"postgresql://user:password@{DB_HOST}:5432/legal_eagle"

# 2. Connect to ChromaDB
CHROMA_CLIENT = chromadb.HttpClient(host=CHROMA_HOST, port=8000)

# 3. Path to data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "raw_data", "dummy_case.txt")


def init_postgres():
    """Create the table in Postgres if it doesn't exist."""
    engine = create_engine(PG_CONNECTION)
    with engine.begin() as conn:
        conn.execute(text("""
                          CREATE TABLE IF NOT EXISTS legal_cases
                          (
                              case_id
                              VARCHAR
                          (
                              50
                          ) PRIMARY KEY,
                              title VARCHAR
                          (
                              255
                          ),
                              date DATE,
                              verdict VARCHAR
                          (
                              50
                          ),
                              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                              );
                          """))
    print("✅ PostgreSQL table ready.")
    return engine


def extract_data(file_path):
    """EXTRACT: Read the raw file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"✅ Extracted {len(content)} characters.")
    return content


def transform_data(raw_text):
    """TRANSFORM: Clean text and parse metadata."""
    data = {}
    lines = raw_text.split('\n')

    # Basic parsing
    data['case_id'] = lines[0].split(': ')[1].strip()
    data['title'] = lines[1].split(': ')[1].strip()
    data['date'] = lines[2].split(': ')[1].strip()
    data['verdict'] = lines[3].split(': ')[1].strip()

    # Extract body text
    body_text = raw_text.split('TEXT:')[1].strip()

    # Simple chunking by paragraph
    chunks = [p for p in body_text.split('\n') if p.strip()]

    print("✅ Transformation complete.")
    return data, chunks


def load_data(engine, meta_data, text_chunks):
    """LOAD: Save to Postgres (Idempotent) and ChromaDB."""

    # --- 1. Load Metadata to Postgres (UPDATED) ---
    # We use raw SQL with 'ON CONFLICT DO NOTHING' to handle duplicates gracefully.
    insert_sql = text("""
                      INSERT INTO legal_cases (case_id, title, date, verdict)
                      VALUES (:case_id, :title, :date, :verdict) ON CONFLICT (case_id) DO NOTHING;
                      """)

    try:
        with engine.begin() as conn:
            conn.execute(insert_sql, meta_data)
        print(f"✅ Metadata synced to Postgres (Case: {meta_data['case_id']})")
    except Exception as e:
        print(f"❌ Database Error: {e}")

    # --- 2. Load Vectors to ChromaDB ---
    # Chroma handles duplicates by ID automatically (it updates them if they exist).
    try:
        collection = CHROMA_CLIENT.get_or_create_collection(name="legal_docs")

        # Generate unique IDs for chunks to prevent overwriting if you re-run
        # We append the chunk index to the case_id to make it deterministic
        ids = [f"{meta_data['case_id']}_chunk_{i}" for i in range(len(text_chunks))]

        metadatas = [{"case_id": meta_data['case_id']} for _ in text_chunks]

        collection.add(
            documents=text_chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ {len(text_chunks)} chunks saved to ChromaDB.")

    except Exception as e:
        print(f"❌ ChromaDB Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting ETL Pipeline...")

    # 1. Init DB
    db_engine = init_postgres()

    # 2. Extract
    raw_content = extract_data(RAW_DATA_PATH)

    # 3. Transform
    meta, chunks = transform_data(raw_content)

    # 4. Load
    load_data(db_engine, meta, chunks)

    print("🎉 ETL Job Finished Successfully!")