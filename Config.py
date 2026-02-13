# config.py
import os
from dotenv import load_dotenv

env_path = r"D:\Projects\Aero-AI_DroneDeliverySystem\Source_Files\CloudVariables.env"
load_dotenv(dotenv_path=env_path)

def validate_env():
    required_vars = [
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "AWS_REGION",
        "RDS_HOST",
        "RDS_DB",
        "RDS_USER",
        "RDS_PASSWORD",
        "RDS_PORT",
        "S3_BUCKET",
        "SES_EMAIL"
    ]
    
    missing = [var for var in required_vars if os.getenv(var) is None]
    
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

# config.py

import os
import psycopg2
from dotenv import load_dotenv

# Absolute path to your env file
load_dotenv(r"D:\Projects\Aero-AI_DroneDeliverySystem\Source_Files\CloudVariables.env")


def get_rds_connection():
    return psycopg2.connect(
        host=os.getenv("RDS_HOST"),
        user=os.getenv("RDS_USER"),
        password=os.getenv("RDS_PASSWORD"),
        database=os.getenv("RDS_DB"),
        port=os.getenv("RDS_PORT")
    )

