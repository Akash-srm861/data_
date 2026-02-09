"""
Web scraping tools — Fetch web pages, extract tables, and call APIs.
"""

from __future__ import annotations

import json

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..utils.state_manager import store_dataframe


def scrape_webpage(url: str, selector: str = "") -> dict:
    """Scrape text content from a web page.

    Args:
        url: The URL to scrape.
        selector: Optional CSS selector to extract specific content.
                  If empty, extracts all visible text from the page body.

    Returns:
        dict: Extracted text content from the page.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if selector:
            elements = soup.select(selector)
            text_parts = [el.get_text(strip=True) for el in elements]
            content = "\n".join(text_parts)
        else:
            content = soup.get_text(separator="\n", strip=True)

        # Limit output length
        content = content[:5000]

        return {
            "status": "success",
            "url": url,
            "content_length": len(content),
            "content": content,
        }
    except Exception as e:
        return {"status": "error", "message": f"Scraping failed: {e}"}


def scrape_table(url: str, table_index: int = 0, dataset_name: str = "") -> dict:
    """Scrape an HTML table from a web page and load it as a dataset.

    Args:
        url: The URL containing the HTML table.
        table_index: Index of the table on the page (0 = first table).
        dataset_name: Optional name for the loaded dataset.

    Returns:
        dict: Status and preview of the scraped table.
    """
    try:
        tables = pd.read_html(url)

        if not tables:
            return {"status": "error", "message": "No tables found on the page."}

        if table_index >= len(tables):
            return {
                "status": "error",
                "message": f"Table index {table_index} out of range. Found {len(tables)} table(s).",
            }

        df = tables[table_index]
        name = dataset_name or f"web_table_{table_index}"
        meta = store_dataframe(name, df)

        return {
            "status": "success",
            "dataset_name": name,
            "tables_found": len(tables),
            "rows": meta["rows"],
            "columns": meta["columns"],
            "preview": df.head(5).to_string(index=False),
        }
    except Exception as e:
        return {"status": "error", "message": f"Table scraping failed: {e}"}


def fetch_api_data(url: str, headers: str = "", dataset_name: str = "") -> dict:
    """Fetch JSON data from a REST API and optionally load it as a dataset.

    Args:
        url: The API endpoint URL.
        headers: Optional JSON string of HTTP headers, e.g. '{"Authorization": "Bearer xxxx"}'.
        dataset_name: Optional name to store the response as a dataset (if data is tabular).

    Returns:
        dict: API response data.
    """
    try:
        req_headers = {"User-Agent": "DataAnalystAgent/1.0"}
        if headers:
            try:
                extra = json.loads(headers)
                req_headers.update(extra)
            except json.JSONDecodeError:
                pass

        response = requests.get(url, headers=req_headers, timeout=15)
        response.raise_for_status()

        data = response.json()

        result = {
            "status": "success",
            "url": url,
            "status_code": response.status_code,
        }

        # Try to convert to DataFrame if data is a list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            name = dataset_name or "api_data"
            meta = store_dataframe(name, df)
            result["dataset_name"] = name
            result["rows"] = meta["rows"]
            result["columns"] = meta["columns"]
            result["preview"] = df.head(5).to_string(index=False)
        else:
            # Return raw data (truncated)
            result["data"] = str(data)[:3000]

        return result
    except Exception as e:
        return {"status": "error", "message": f"API request failed: {e}"}
