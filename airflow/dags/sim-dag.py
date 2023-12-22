import sys
import psycopg2
import pyodbc
import pandas as pd

from datetime import datetime, timedelta
import pendulum
# import subprocess
# Add the path to the /app directory to the Python path
sys.path.append('/app')
# import the simulation to run within the DAG task
from sim import AEModel, Tracker, p 

#from airflow.operators.python_operator import PythonOperator
#from airflow.utils.task_group import TaskGroup
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
#from airflow.providers.microsoft.azure.transfers.azure_data_lake_gen2_to_parquet import AzureDataLakeStorageGen2ToParquetOperator
from airflow.decorators import dag, task, task_group

# function to read in postgres data
def read_data_from_postgres():
    # Modify these parameters with your database credentials
    db_params = {
        'host': 'postgres',
        'database': 'source_system',
        'user': 'postgres',
        'password': 'postgres',
        'port': '5432',
    }

    # Define your SQL query
    sql_query = "SELECT * FROM store.ae_attends;"

    # Connect to the database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    try:
        # Execute the query
        # cursor.execute(sql_query)

        # # Fetch all the results
        # result = cursor.fetchall()

        # return result        
        # Use Pandas to read SQL query result into a DataFrame
        df = pd.read_sql_query(sql_query, connection)
        
        # Print or process the DataFrame as needed
        # print(df['patient_id'])
        # print(df['provider_id'])
        # print(df['priority'])
        # print(df['time_in_system'])
        # print(df['attendance_time'])
        # print(df['departure_time'])
        # print(df['doctor_id_seen'])

        # adjust data types
        df.patient_id = df.loc[:,'patient_id'].astype('int')
        df.provider_id = df.loc[:,'provider_id'].astype('int')

        print(df.dtypes)

        # You can return the DataFrame if you want to use it in subsequent tasks
        return df
    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()

# function to insert data into Azure SQL db
# def insert_into_azure_df(df):
#     drivers = [item for item in pyodbc.drivers()]
#     driver = drivers[-1]
#     print("driver:{}".format(driver))
#     server = 'freesqldbserver-ym.database.windows.net' 
#     database = 'myFreeDB' 
#     username = 'freedb' 
#     password = 'DBya55er' 
#     #driver= '{ODBC Driver 17 for SQL Server}'
#     conn = pyodbc.connect('DRIVER=' + driver + ';SERVER=' +
#         server + ';DATABASE=' + database +
#         ';UID=' + username + ';PWD=' + password)
#         # removed after server + ;PORT=1433
#     try:
#         cursor = conn.cursor()
#         # Insert Dataframe into SQL Server:
#         for index, row in df.iterrows():
#             cursor.execute("INSERT INTO Summary.SimTest (patient_id, provider_id, arrival_mode, priority, triage_outcome, time_in_system, attendance_time, departure_time, admitted, doctor_id_seen) values(?,?,?,?,?,?,?,?,?,?)",
#                             row.patient_id, 
#                             row.provider_id, 
#                             row.arrival_mode,
#                             row.priority,
#                             row.triage_outcome,
#                             row.time_in_system,
#                             row.attendance_time,
#                             row.departure_time,
#                             row.admitted,
#                             row.doctor_id_seen)
#         conn.commit()
#         cursor.close()
#     except pyodbc.Error as e:
#         # Print the error details
#         print(f"PyODBC Error: {e}")
#         print("SQL State:", e.sqlstate)
#         print("Native Error:", e.native_error)
#         print("Error Message:", e)
#         #traceback.print_exc()

#         # You might want to raise the exception again if you want Airflow to mark the task as failed
#         raise
#     finally:
#         # Close the database connection
#         conn.close()
@dag(
    schedule_interval="0 6 * * *",
    start_date=datetime(2023, 1, 1, tzinfo=pendulum.timezone("UTC")),
    catchup=False,
    tags=["sim_schedule"],
)
def sim_taskflow_api():
    """
    ### TaskFlow API Tutorial Documentation
    This is a simple data pipeline example which demonstrates the use of
    the TaskFlow API using three simple tasks for Extract, Transform, and Load.
    Documentation that goes along with the Airflow TaskFlow API tutorial is
    located
    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)
    """
    # Define a dummy task to serve as the starting point
    start_task = EmptyOperator(task_id='start')
    # Define the number of runs
    number_of_runs = p.number_of_runs

    @task()
    def run_simulation(run):
        # Create a new instance of the AEModel for each run
        # run is provider id
        my_ae_model = AEModel(run)
        triage_mean, cubicle_mean, doc_mean, miu_mean = my_ae_model.run()
        # Reset doctors after each run
        p.doc_ids = list(range(1, p.number_docs + 1))
    
    #@task()
    #def extract_from_postgres():
        # Use the PostgresOperator to extract data from Postgres
    # sql_query = "SELECT * FROM source_system.store.ae_attends;"
    # task_id = "extract_from_postgres_task"
    # postgres_task = PostgresOperator(
    #     task_id=task_id,
    #     sql=sql_query,
    #     postgres_conn_id="postgres_default"
    #     #dag=DAG,
    # )
        #return postgres_task

    # Use a loop to create tasks for each simulation run
    # Create a TaskGroup for the simulation tasks
    @task_group(group_id="simulation_tasks")
    def loop_simulation():
        for run_number in range(1, number_of_runs + 1):
            run_simulation(run_number) 
            
    @task()
    def extract_from_postgres():
        result = read_data_from_postgres()
        # Do something with the result, e.g., write to a file, process the data, etc.
        #print(result[:10])
        return result
    
    # @task()
    # def insert_into_azure(df):
    #     insert_into_azure_df(df)
    # Create the extract task
    # extract_task = PythonOperator(
    #     task_id='extract_from_postgres_task',
    #     python_callable=extract_from_postgres,
    #     dag=sim_taskflow_api,
    # )
    # Set up dependencies
    start_task >> loop_simulation() >> extract_from_postgres()
# Instantiate the DAG
dag_instance = sim_taskflow_api()

