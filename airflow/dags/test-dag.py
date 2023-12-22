from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def my_python_function():
    # Your Python code goes here
    print("Hello from my Python function in Airflow!")

# Define the DAG
dag = DAG(
    'my_dag',
    description='My example DAG',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
)

# Define a task using PythonOperator
task = PythonOperator(
    task_id='my_task',
    python_callable=my_python_function,
    dag=dag,
)
