"""
Report generation tools — Markdown and PDF report building.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from fpdf import FPDF

from ..utils.state_manager import get_output_path


def generate_markdown_report(title: str, sections: list[dict]) -> dict:
    """Generate a Markdown report from structured sections.

    Each section should have 'heading' and 'content' keys.
    Optionally include 'chart_path' to reference a chart image.

    Args:
        title: Report title.
        sections: List of dicts with 'heading' (str), 'content' (str),
                  and optional 'chart_path' (str) keys.

    Returns:
        dict: Status and file path of the generated Markdown report.
    """
    try:
        lines = [
            f"# {title}",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "---",
            "",
        ]

        for section in sections:
            heading = section.get("heading", "Section")
            content = section.get("content", "")
            chart = section.get("chart_path", "")

            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")

            if chart and os.path.exists(chart):
                lines.append(f"![{heading}]({chart})")
                lines.append("")

            lines.append("---")
            lines.append("")

        md_content = "\n".join(lines)
        filename = f"report_{uuid.uuid4().hex[:6]}.md"
        path = get_output_path(filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return {
            "status": "success",
            "report_path": path,
            "format": "markdown",
            "report_preview": md_content[:1000],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_pdf_report(title: str, sections: list[dict], charts: list[str] = []) -> dict:
    """Generate a PDF report with text sections and optional chart images.

    Args:
        title: Report title.
        sections: List of dicts with 'heading' and 'content' keys.
        charts: List of chart image file paths to embed.

    Returns:
        dict: Status and file path of the generated PDF report.
    """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 40, text=title, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(
            0, 10,
            text=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )
        pdf.ln(10)

        # Sections
        for section in sections:
            heading = section.get("heading", "Section")
            content = section.get("content", "")
            chart_path = section.get("chart_path", "")

            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 12, text=heading, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, text=content)
            pdf.ln(4)

            if chart_path and os.path.exists(chart_path):
                try:
                    pdf.image(chart_path, w=170)
                    pdf.ln(5)
                except Exception:
                    pass

        # Additional standalone charts
        for chart_path in charts:
            if os.path.exists(chart_path):
                try:
                    pdf.add_page()
                    pdf.image(chart_path, x=10, y=20, w=190)
                except Exception:
                    pass

        filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        path = get_output_path(filename)
        pdf.output(path)

        return {
            "status": "success",
            "report_path": path,
            "format": "pdf",
            "pages": pdf.pages_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
