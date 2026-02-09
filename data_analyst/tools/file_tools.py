"""
File loading tools — CSV, Excel, and PDF ingestion.
"""

from __future__ import annotations

import os

import pandas as pd
from PyPDF2 import PdfReader

from ..utils.state_manager import (
    UPLOADS_DIR,
    get_dataframe,
    list_datasets,
    store_dataframe,
)


def load_csv(file_path: str, dataset_name: str = "") -> dict:
    """Load a CSV file into a dataset for analysis.

    Args:
        file_path: Path to the CSV file (filename only or absolute path).
        dataset_name: Optional friendly name for the dataset. Defaults to filename.

    Returns:
        dict: Status message with dataset preview (first 5 rows) and metadata.
    """
    try:
        # If just a filename, look in uploads/
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOADS_DIR, file_path)

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        df = pd.read_csv(file_path)
        name = dataset_name or os.path.splitext(os.path.basename(file_path))[0]
        meta = store_dataframe(name, df)

        return {
            "status": "success",
            "dataset_name": name,
            "rows": meta["rows"],
            "columns": meta["columns"],
            "dtypes": meta["dtypes"],
            "preview": df.head(5).to_string(index=False),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_excel(file_path: str, sheet_name: str = "", dataset_name: str = "") -> dict:
    """Load an Excel file into a dataset for analysis.

    Args:
        file_path: Path to the Excel file (filename only or absolute path).
        sheet_name: Name of the sheet to load. Defaults to the first sheet.
        dataset_name: Optional friendly name. Defaults to filename_sheetname.

    Returns:
        dict: Status message with dataset preview and metadata.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOADS_DIR, file_path)

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        sheet = sheet_name if sheet_name else 0
        df = pd.read_excel(file_path, sheet_name=sheet, engine="openpyxl")
        base = os.path.splitext(os.path.basename(file_path))[0]
        name = dataset_name or f"{base}_{sheet_name}" if sheet_name else base
        meta = store_dataframe(name, df)

        return {
            "status": "success",
            "dataset_name": name,
            "rows": meta["rows"],
            "columns": meta["columns"],
            "dtypes": meta["dtypes"],
            "preview": df.head(5).to_string(index=False),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def load_pdf(file_path: str) -> dict:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file (filename only or absolute path).

    Returns:
        dict: Extracted text from each page.
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(UPLOADS_DIR, file_path)

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text.strip()})

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])

        return {
            "status": "success",
            "filename": os.path.basename(file_path),
            "total_pages": len(reader.pages),
            "text_preview": full_text[:2000],
            "full_text": full_text,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_uploaded_files() -> dict:
    """List all files in the uploads directory and all loaded datasets.

    Returns:
        dict: Lists of uploaded files and loaded datasets.
    """
    try:
        files = []
        if os.path.exists(UPLOADS_DIR):
            files = [
                f for f in os.listdir(UPLOADS_DIR)
                if os.path.isfile(os.path.join(UPLOADS_DIR, f)) and f != ".gitkeep"
            ]

        datasets = list_datasets()

        return {
            "status": "success",
            "uploaded_files": files,
            "loaded_datasets": datasets,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
