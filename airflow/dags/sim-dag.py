import sys
import time
import psycopg2
import pyodbc
import pandas as pd

# import os, uuid
from io import BytesIO
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from datetime import datetime, timedelta
import pendulum

# import subprocess
# Add the path to the /app directory to the Python path
sys.path.append('/app')
# import the simulation to run within the DAG task
from sim import AEModel, Tracker, p 

# Airflow processes 
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.decorators import dag, task, task_group
from airflow.models import Variable


### Initially we set up some functions that will be used when accessing/uploading data ###
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
        # return result        
        # Use Pandas to read SQL query result into a DataFrame
        df = pd.read_sql_query(sql_query, connection)
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

# insert data into storage - upload to Azure gen2 storage
def insert_into_azure_str():
    # get current data in Postgres
    df = read_data_from_postgres()
    # below gets the secret key for Azure storage access, which was added using docker secrets
    secret_path = "/run/secrets/azure_secret"
    with open(secret_path, "r") as secret_file:
        connect_str = secret_file.read().strip()
    try:
        print("Placing data as Azure Blob storage in parquet format")
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        # Can create a unique name for the container with...
        #container_name = str(uuid.uuid4())

        container_name = "data"
        file_name_parquet = "sim_data.parquet"
        #file_name_csv = "sim_data.csv"

        # Get a reference to the container
        container_client = blob_service_client.get_container_client(container_name)
        # Check if the container exists, and create it if not
        if not container_client.exists():
            container_client.create_container()
        # Get a reference to the blob
        blob_client = container_client.get_blob_client(blob=file_name_parquet)        

        print("\nUploading to Azure Storage as blob:\n\t" + file_name_parquet)

        parquet_file = BytesIO()
        df.to_parquet(parquet_file, engine='pyarrow')
        parquet_file.seek(0)  # change the stream position back to the beginning after writing
        
        # upload function
        blob_client.upload_blob(
            data=parquet_file,
            overwrite=True
        )
        print("Parquet upload complete!")

        ### csv creation and upload if needed - uncomment csv filename above ###
        # Get a reference to the CSV blob
        # blob_client_csv = container_client.get_blob_client(blob=file_name_csv)

        # # Upload CSV data
        # csv_file = BytesIO()
        # df.to_csv(csv_file, index=False)
        # csv_file.seek(0)
        # blob_client_csv.upload_blob(
        #     data=csv_file,
        #     overwrite=True
        # )
        # print("CSV upload complete!")
        # # Upload the created file
        # with open(file=upload_file_path, mode="rb") as data:
        #     blob_client.upload_blob(data)
        #     print("upload complete!")

    except Exception as ex:
        print('Exception:')
        print(ex)


# if needed - below function to insert data into Azure SQL db
# however in this case we will not use this
def insert_into_azure_sql():
    secret_path = "/run/secrets/azure_secret"
    with open(secret_path, "r") as secret_file:
        connect_str = secret_file.read().strip()
    
    # the below will try to connect to Azure SQL server upto 10 times (required due to server sleeping)
    max_retries = 10
    current_retry = 0

    while current_retry < max_retries:
        try:
            print("Getting data from Azure Blob and inserting into Azure SQL")
            print("ATTEMPT NUMBER:", current_retry)
            container_name = "data"
            file_name_parquet = "sim_data.parquet"
            # Create the BlobServiceClient object
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)

            # Get a reference to the container
            container_client = blob_service_client.get_container_client(container_name)

            # Get a reference to the blob
            blob_client = container_client.get_blob_client(blob=file_name_parquet) 

            print("\nListing blobs...")

            # List the blobs in the container
            blob_list = container_client.list_blobs()
            for blob in blob_list:
                print("\t" + blob.name)

            downloaded_blob = container_client.download_blob(file_name_parquet)
            bytes_io = BytesIO(downloaded_blob.readall())
            df = pd.read_parquet(bytes_io)
            print(df.head())

            ### insert into Azure SQL
            # drivers = [item for item in pyodbc.drivers()]
            # driver = drivers[-1]
            # print("driver:{}".format(driver))

            # get variables from Airflow
            server_name = Variable.get("server_port")
            database_name = Variable.get("database")
            username = Variable.get("username")
            password = Variable.get("password")

            # Create a connection string
            conn_str = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server_name};DATABASE={database_name};UID={username};PWD={password}"
            #print("THIS IS THE CONNECTION STRING: ", conn_str)

            # Establish the connection
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # Insert rows from the DataFrame into the SQL Server table
            for index, row in df.iterrows():
                cursor.execute(
                    "INSERT INTO Summary.SimTest (patient_id, provider_id, arrival_mode, priority, triage_outcome, time_in_system, attendance_time, departure_time, admitted, doctor_id_seen) "
                    "OUTPUT INSERTED.patient_id "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row['patient_id'], row['provider_id'], row['arrival_mode'], row['priority'], row['triage_outcome'], row['time_in_system'], row['attendance_time'], row['departure_time'], row['admitted'], row['doctor_id_seen'] 
                )
                inserted_patient_id = cursor.fetchone()[0]
                print(f"Inserted Patient ID: {inserted_patient_id}")

            # Commit the changes and close the connection
            conn.commit()
            conn.close()

            # Break out of the loop if the connection and insertion were successful
            break
        except pyodbc.Error as ex:
            if 'Login timeout expired' in str(ex):
                current_retry += 1
                print(f"Login timeout expired. Retrying {current_retry}/{max_retries}...")
                time.sleep(5)  # Add a delay before retrying
            else:
                print('Exception:')
                print(ex)
                break  # Break out of the loop for other exceptions

    if current_retry == max_retries:
        print(f"Max retries ({max_retries}) reached. Unable to establish a connection.")

# below removes data from 'backend' postgres to avoid large volume of data building up        
def clean_up_postgres():
    ### function to clean up data left in postgres and in Azure Blob to limit size of file ###
    # Modify these parameters with your database credentials
    db_params = {
        'host': 'postgres',
        'database': 'source_system',
        'user': 'postgres',
        'password': 'postgres',
        'port': '5432',
    }

    # Define your SQL query
    sql_query = "TRUNCATE TABLE store.ae_attends;"

    # Connect to the database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    try:
        # Execute the query
        print("Attempting to truncate postgres")
        cursor.execute(sql_query)
        # Make the changes to the database persistent
        connection.commit()
    
    finally:
        # Close the cursor and connection
        cursor.close()
        connection.close()
    
    
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

    # Use a loop to create tasks for each simulation run
    # Create a TaskGroup for the simulation tasks
    @task_group(group_id="simulation_tasks")
    def loop_simulation():
        for run_number in range(1, number_of_runs + 1):
            run_simulation(run_number) 
    
    @task()
    def upload_to_azure():
        insert_into_azure_str()
    # below is task to insert into an Azure SQL instance, although we don't use in this case    
    @task()
    def insert_to_sql():
        insert_into_azure_sql()
    
    @task()
    def clean_up():
        clean_up_postgres()
    
    # Set up dependencies
    start_task >> loop_simulation() >> upload_to_azure() >> clean_up() # optional step after upload to Azure > >> insert_to_sql()
# Instantiate the DAG
dag_instance = sim_taskflow_api()

