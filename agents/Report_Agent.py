import pandas as pd
from psycopg2 import Error
from Config import get_rds_connection

def handle_report(user_query: str):
    """
    Generates drone reports dynamically from AWS RDS (PostgreSQL) based on user prompts.
    Supports filtering by drone type, health status, and format (txt/csv/xlsx).
    Returns:
        - String for .txt
        - List of dicts for .csv or .xlsx
    """
    try:
        # Connect to RDS
        conn = get_rds_connection()
        df = pd.read_sql("SELECT * FROM drone_image_metadata;", conn)
        conn.close()
        df = df.applymap(lambda x: x.encode("utf-8", errors="replace").decode("utf-8") if isinstance(x, str) else x)

        # Filtering by drone_type
        for dt in df['drone_type'].unique():
            if dt.lower() in user_query.lower():
                df = df[df['drone_type'] == dt]
                break

        # Filtering by health_status
        for hs in df['health_status'].unique():
            if hs.replace("_", " ").lower() in user_query.lower():
                df = df[df['health_status'] == hs]
                break

        # Determine if summary requested
        if "health status" in user_query.lower() or "count" in user_query.lower():
            drone_counts = df['drone_type'].value_counts().to_dict()
            report_list = []
            for dt, count in drone_counts.items():
                health_summary = ", ".join(
                    f"{hs}: {sum((df['drone_type']==dt) & (df['health_status']==hs))}"
                    for hs in df['health_status'].unique()
                )
                report_list.append({
                    "Drone Type": dt,
                    "Total Count": count,
                    "Health Summary": health_summary
                })
            return report_list

        # Default: return full table
        return df.to_dict(orient='records')

    except Error as e:
        return f"Report Agent Error (RDS): {str(e)}"
    except Exception as e:
        return f"Report Agent Error: {str(e)}"
