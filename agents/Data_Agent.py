from Config import get_rds_connection
from agents.local_llm import ask_llm
import pandas as pd


TABLE_SCHEMA = """
Table Name: drone_image_metadata

Columns:
- id (SERIAL)
- user_id (VARCHAR)
- s3_path (TEXT)
- drone_type (VARCHAR)
- drone_type_conf (FLOAT)
- health_status (VARCHAR)
- health_conf (FLOAT)
- uploaded_at (TIMESTAMP)
- processed_at (TIMESTAMP)
- image_size_bytes (BIGINT)
- image_format (VARCHAR)
- model_version (VARCHAR)
"""


import re

def generate_sql_from_llm(user_query):
    prompt = f"""
    It is an expert PostgreSQL query generator.

    Generate ONLY a SELECT SQL query.

    Rules:
    - Only SELECT
    - Return plain SQL only

    Database Schema:
    {TABLE_SCHEMA}

    User Request:
    {user_query}
    """

    raw_output = ask_llm(prompt)

    # Clean LLM Output
    cleaned = raw_output.strip()

    # Remove markdown code blocks
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    # Extract SELECT statement only
    match = re.search(r"(select .*?;)", cleaned, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    return cleaned.strip()

def validate_sql(sql_query):
    sql_lower = sql_query.strip().lower()

    if not sql_lower.startswith("select"):
        return False

    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]

    for word in forbidden:
        if word in sql_lower:
            return False

    return True



def handle_data_query(user_query):

    try:
        # Generate SQL
        sql_query = generate_sql_from_llm(user_query)

        if not validate_sql(sql_query):
            return "⚠️ Unsafe query detected. Only SELECT queries allowed."

        # Execute SQL
        conn = get_rds_connection()

        df = pd.read_sql_query(sql_query, conn)

        conn.close()

        if df.empty:
            return "No data found."

        return df

    except Exception as e:
        return f"Data Agent Error: {str(e)}"
