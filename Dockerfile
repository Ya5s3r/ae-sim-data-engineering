# Use the official Airflow image as the base image
FROM apache/airflow:2.7.3

# Set the working directory in the container
WORKDIR /app

# Install any Python dependencies you need
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install Apache Airflow providers
RUN pip install apache-airflow-providers-postgres apache-airflow-providers-microsoft-azure

# Initialize the Airflow database during the build
RUN airflow db init

# Copy your Python scripts into the container
COPY src/ .

USER root 
RUN apt-get update
RUN apt-get update && apt-get install -y gnupg2
RUN apt-get install -y curl apt-transport-https
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18 
#ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

# Set the entry point and command
CMD ["webserver"]

