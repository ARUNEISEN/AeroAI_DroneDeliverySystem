from fpdf import FPDF
import os
from datetime import datetime


def generate_pdf_report(data: str, filename: str = None):
    """
    Generates a formatted PDF report from text data.
    """

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drone_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Aero AI - Drone Intelligence Report", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", size=12)

    for line in data.split("\n"):
        pdf.multi_cell(0, 8, line)

    os.makedirs("reports", exist_ok=True)
    output_path = os.path.join("reports", filename)

    pdf.output(output_path)

    return output_path
