# Use the official Airflow image as the base image
FROM apache/airflow:2.7.3

# Set the working directory in the container
WORKDIR /app

# Install any Python dependencies you need
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install Apache Airflow providers
RUN pip install apache-airflow-providers-postgres apache-airflow-providers-microsoft-azure

# Copy your Python scripts into the container
COPY src/ .

# Set the entry point and command
CMD ["webserver"]

