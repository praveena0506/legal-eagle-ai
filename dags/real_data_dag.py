from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Point Airflow to your code folder
sys.path.append("/opt/airflow")

# Import the function we just wrote in Step 2
from data_pipeline.scripts.fetch_real_data import run_scraper

default_args = {
    'owner': 'legal_eagle',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='fetch_indian_kanoon_daily',
    default_args=default_args,
    description='Scrape 5 new cases every day',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily', # Runs at midnight
    catchup=False
) as dag:

    scrape_task = PythonOperator(
        task_id='scrape_task',
        python_callable=run_scraper
    )