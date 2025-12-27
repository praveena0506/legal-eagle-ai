from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# 1. Add your project to the Python Path so Airflow can find your script
# Airflow runs in /opt/airflow, so we point to where we mounted the code
sys.path.append("/opt/airflow")

# 2. Import your actual ETL function
from data_pipeline.scripts.etl_process import extract_data, transform_data, load_data, init_postgres, RAW_DATA_PATH


# 3. Define the wrapper function
# This is the function Airflow calls. It runs your whole pipeline.
def run_etl_pipeline():
    print("🚀 Airflow is starting the ETL job...")

    # Run the exact same steps you ran manually
    db_engine = init_postgres()
    raw_content = extract_data(RAW_DATA_PATH)
    meta, chunks = transform_data(raw_content)
    load_data(db_engine, meta, chunks)

    print("🎉 Airflow finished the job!")


# 4. Define the DAG (The Schedule)
default_args = {
    'owner': 'legal_eagle',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        dag_id='daily_legal_cases_etl',
        default_args=default_args,
        description='Process new legal cases every day',
        start_date=datetime(2023, 1, 1),
        schedule_interval='@daily',  # Run once a day at midnight
        catchup=False,  # Don't run for past dates
        tags=['legal', 'etl']
) as dag:
    # 5. The Task
    run_etl_task = PythonOperator(
        task_id='run_full_pipeline',
        python_callable=run_etl_pipeline
    )

    run_etl_task