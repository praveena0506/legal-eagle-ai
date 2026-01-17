from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import shutil
from pathlib import Path

# Setup Paths
sys.path.append("/opt/airflow")
from data_pipeline.scripts.prepare_data import run_data_preparation
from data_pipeline.scripts.train_model import run_training_loop


# --- AUTO-LABELER FUNCTION ---
# This moves files from "unlabeled" to "accepted/rejected" based on keywords
def auto_label_files():
    print("🏷️ Starting Auto-Labeling...")
    BASE_DIR = Path("/opt/airflow/data_pipeline/raw_data")
    UNLABELED = BASE_DIR / "unlabeled_daily_dump"
    ACCEPTED = BASE_DIR / "appeal_accepted"
    REJECTED = BASE_DIR / "appeal_rejected"

    ACCEPTED.mkdir(exist_ok=True)
    REJECTED.mkdir(exist_ok=True)

    files = list(UNLABELED.glob("*.txt"))
    if not files:
        print("   No new files to label.")
        return

    count = 0
    for f in files:
        try:
            content = f.read_text(encoding="utf-8").lower()
            if "allowed" in content or "set aside" in content:
                shutil.move(str(f), str(ACCEPTED / f.name))
                count += 1
            elif "dismissed" in content:
                shutil.move(str(f), str(REJECTED / f.name))
                count += 1
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    print(f"✅ Labeled and moved {count} files.")


# --- DAG DEFINITION ---
default_args = {
    'owner': 'legal_eagle',
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        dag_id='weekly_model_update',
        default_args=default_args,
        description='Label data, Process it, and Retrain Model',
        start_date=datetime(2023, 1, 1),
        schedule_interval='@weekly',  # Runs once a week
        catchup=False,
        tags=['mlops', 'training']
) as dag:
    # Task 1: Label the data
    label_task = PythonOperator(
        task_id='auto_label_new_data',
        python_callable=auto_label_files
    )

    # Task 2: Prepare tensors
    prepare_task = PythonOperator(
        task_id='prepare_data_tensors',
        python_callable=run_data_preparation
    )

    # Task 3: Train the model
    train_task = PythonOperator(
        task_id='train_and_save_model',
        python_callable=run_training_loop
    )

    # The Pipeline Order
    label_task >> prepare_task >> train_task