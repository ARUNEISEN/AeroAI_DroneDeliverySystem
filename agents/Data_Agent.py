from Config import get_rds_connection
from agents.local_llm import ask_llm
import pandas as pd
import re


# ================
# DATABASE CONFIG
# ================

TABLE_NAME = "drone_image_metadata"

VALID_COLUMNS = [
    "id",
    "user_id",
    "s3_path",
    "drone_type",
    "drone_type_conf",
    "health_status",
    "health_conf",
    "uploaded_at",
    "processed_at",
    "image_size_bytes",
    "image_format",
    "model_version"
]


# =====================
# NORMALIZE USER INPUT
# =====================

def normalize_query(text):
    text = text.lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =============================
# DISTINCT HEALTH STATUS VALUES
# =============================

def get_health_status_values():
    try:
        conn = get_rds_connection()
        df = pd.read_sql_query(
            f"SELECT DISTINCT health_status FROM {TABLE_NAME};",
            conn
        )
        conn.close()

        values = df["health_status"].dropna().tolist()
        return values

    except:
        return []


# ================================
# MAP HEALTH STATUS FROM USER TEXT
# ================================

def detect_health_filter(user_query, health_values):

    query_clean = user_query.replace(" ", "").lower()

    for value in health_values:
        value_clean = value.replace("_", "").lower()

        if value_clean in query_clean:
            return value

    return None


# ========================
# BASIC INTENT RULE ENGINE
# ========================

def rule_based_query(user_query):

    normalized = normalize_query(user_query)

    health_values = get_health_status_values()
    health_filter = detect_health_filter(normalized, health_values)

    is_count = "count" in normalized or "total" in normalized
    is_avg = "average" in normalized or "avg" in normalized

    if is_count:
        if health_filter:
            return f"""
            SELECT COUNT(*) 
            FROM {TABLE_NAME}
            WHERE health_status = '{health_filter}';
            """

        return f"SELECT COUNT(*) FROM {TABLE_NAME};"

    if is_avg:
        if "health conf" in normalized:
            return f"SELECT AVG(health_conf) FROM {TABLE_NAME};"

        if "drone type conf" in normalized:
            return f"SELECT AVG(drone_type_conf) FROM {TABLE_NAME};"

    return None

# ==================
# LLM SQL GENERATOR 
# ==================

def generate_sql_from_llm(user_query):

    normalized_query = normalize_query(user_query)

    health_values = get_health_status_values()

    prompt = f"""
You are an expert PostgreSQL query generator.

STRICT RULES:
- Generate ONLY SELECT queries.
- Use ONLY columns from this table.
- Do NOT invent columns.
- Return plain SQL ending with semicolon.
- No explanations.
- No markdown.

Table Name: {TABLE_NAME}

Columns:
{", ".join(VALID_COLUMNS)}

Valid health_status values:
{", ".join(health_values)}

User Request:
{normalized_query}
"""

    raw_output = ask_llm(prompt)

    cleaned = raw_output.strip()
    cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)

    match = re.search(r"(select .*?;)", cleaned, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()

    return cleaned.strip()


# =====================================================
# SQL VALIDATION
# =====================================================

def validate_sql(sql_query):

    sql_lower = sql_query.strip().lower()

    if not sql_lower.startswith("select"):
        return False

    forbidden_keywords = [
        "insert", "update", "delete",
        "drop", "alter", "truncate",
        "create"
    ]

    for word in forbidden_keywords:
        if word in sql_lower:
            return False

    return True


# =====================================================
# MAIN HANDLER
# =====================================================

def handle_data_query(user_query):

    try:

        # Try fast rule-based engine first
        sql_query = rule_based_query(user_query)

        # If no rule matched → fallback to LLM
        if not sql_query:
            sql_query = generate_sql_from_llm(user_query)

        # Validate
        if not validate_sql(sql_query):
            return "Unsafe query detected."

        # Execute
        conn = get_rds_connection()
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        if df.empty:
            return "No data found."

        # Convert DataFrame to JSON-safe format
        return df.to_dict(orient="records")

    except Exception as e:
        return f"Data Agent Error: {str(e)}"