"""
Shared state management helpers for inter-agent data sharing.

DataFrames are stored in session state as dict records (JSON-serializable).
Convention:
  - state["datasets"][name]       -> list of row dicts
  - state["dataset_meta"][name]   -> {"columns": [...], "rows": int, "dtypes": {...}}
  - state["db_connection"]        -> connection string
  - state["chart_paths"]          -> list of generated chart file paths
  - state["report_paths"]         -> list of generated report file paths
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "outputs")

# ── In-memory DataFrame cache (non-serializable, kept outside session state) ──
_dataframe_cache: dict[str, pd.DataFrame] = {}


def store_dataframe(name: str, df: pd.DataFrame, tool_context=None) -> dict:
    """Store a DataFrame both in the in-memory cache and in session state (as records).

    Args:
        name: A unique name/key for the dataset.
        df: The pandas DataFrame to store.
        tool_context: ADK ToolContext for accessing session state (optional).

    Returns:
        dict with dataset metadata.
    """
    _dataframe_cache[name] = df

    meta = {
        "columns": list(df.columns),
        "rows": len(df),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Store a preview (first 100 rows) in session state for agent access
    preview = df.head(100).to_dict(orient="records")

    if tool_context is not None:
        state = tool_context.state
        if "datasets" not in state:
            state["datasets"] = {}
        if "dataset_meta" not in state:
            state["dataset_meta"] = {}
        state["datasets"][name] = preview
        state["dataset_meta"][name] = meta

    return meta


def get_dataframe(name: str) -> pd.DataFrame | None:
    """Retrieve a DataFrame from the in-memory cache.

    If the exact name is not found but there is exactly one dataset loaded,
    return that one (convenience fallback).

    Args:
        name: The dataset key.

    Returns:
        The DataFrame, or None if not found.
    """
    df = _dataframe_cache.get(name)
    if df is not None:
        return df
    # Fallback: if only one dataset is loaded, use it regardless of name
    if len(_dataframe_cache) == 1:
        return next(iter(_dataframe_cache.values()))
    return None


def list_datasets() -> list[str]:
    """Return names of all cached datasets."""
    return list(_dataframe_cache.keys())


def get_dataset_columns(name: str = "") -> dict | None:
    """Return column names and types for a loaded dataset.

    If name is empty and only one dataset is loaded, use that one.
    Returns dict with 'numeric', 'categorical', 'datetime', 'all' column lists.
    """
    if name:
        df = _dataframe_cache.get(name)
    elif len(_dataframe_cache) == 1:
        df = next(iter(_dataframe_cache.values()))
    else:
        return None
    if df is None:
        return None
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime_cols,
        "all": df.columns.tolist(),
    }


def clear_cache():
    """Clear the in-memory DataFrame cache."""
    _dataframe_cache.clear()


def get_output_path(filename: str) -> str:
    """Return an absolute path inside the outputs/ directory."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    return os.path.join(OUTPUTS_DIR, filename)


def get_upload_path(filename: str) -> str:
    """Return an absolute path inside the uploads/ directory."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    return os.path.join(UPLOADS_DIR, filename)
