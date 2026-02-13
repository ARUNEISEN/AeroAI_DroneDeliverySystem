import boto3
import os
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

env_path = r"D:\Projects\Aero-AI_DroneDeliverySystem\Source_Files\CloudVariables.env"
load_dotenv(dotenv_path=env_path)

def send_email_with_attachment(to_email, subject, body, attachment_path=None):
    aws_region = os.getenv("AWS_REGION", "ap-south-1")
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    sender_email = os.getenv("SES_EMAIL")

    if not aws_access_key or not aws_secret_key:
        return "AWS credentials not found."

    if not sender_email:
        return "SES sender email not configured."

    try:
        ses_client = boto3.client(
            "ses",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Message-ID"] = f"<{os.urandom(16).hex()}@example.com>"

        # Email body
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attach file if exists
        if attachment_path and os.path.exists(attachment_path):
            filename = os.path.basename(attachment_path)
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="octet-stream")
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"'
                )
                msg.attach(part)

        ses_client.send_raw_email(
            Source=sender_email,
            Destinations=[to_email],
            RawMessage={"Data": msg.as_string()},
        )

        return f"Email sent successfully to {to_email}"

    except ClientError as e:
        return f"Email failed: {e.response['Error']['Message']}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
