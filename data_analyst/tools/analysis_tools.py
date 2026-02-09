"""
Statistical analysis tools — descriptive stats, correlation, outliers, trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.state_manager import get_dataframe, list_datasets


def describe_data(dataset_name: str) -> dict:
    """Generate descriptive statistics for a loaded dataset.

    Includes count, mean, std, min, max, quartiles for numeric columns,
    and value counts for categorical columns.

    Args:
        dataset_name: Name of a previously loaded dataset.

    Returns:
        dict: Descriptive statistics and column information.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        available = list_datasets()
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found. Available: {available}",
        }

    try:
        numeric_stats = df.describe(include="number").to_dict()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        cat_stats = {}
        for col in categorical_cols:
            vc = df[col].value_counts().head(10)
            cat_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": vc.to_dict(),
                "null_count": int(df[col].isnull().sum()),
            }

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "numeric_statistics": numeric_stats,
            "categorical_summary": cat_stats,
            "null_counts": df.isnull().sum().to_dict(),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def compute_correlation(dataset_name: str, col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient between two numeric columns.

    Args:
        dataset_name: Name of the dataset.
        col1: First column name.
        col2: Second column name.

    Returns:
        dict: Correlation coefficient and p-value.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        if col1 not in df.columns or col2 not in df.columns:
            return {
                "status": "error",
                "message": f"Column(s) not found. Available: {list(df.columns)}",
            }

        clean = df[[col1, col2]].dropna()
        if len(clean) < 3:
            return {"status": "error", "message": "Not enough data points (need at least 3)."}

        corr, p_value = stats.pearsonr(clean[col1], clean[col2])

        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        direction = "positive" if corr > 0 else "negative"

        return {
            "status": "success",
            "correlation": round(float(corr), 4),
            "p_value": round(float(p_value), 6),
            "interpretation": f"{strength} {direction} correlation (r={corr:.4f}, p={p_value:.6f})",
            "sample_size": len(clean),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_outliers(dataset_name: str, column: str) -> dict:
    """Detect outliers in a numeric column using the IQR method.

    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged as outliers.

    Args:
        dataset_name: Name of the dataset.
        column: The numeric column to analyze.

    Returns:
        dict: Outlier count, boundaries, and outlier values.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found."}

        series = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return {"status": "error", "message": f"Column '{column}' is not numeric."}

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]

        return {
            "status": "success",
            "column": column,
            "total_values": len(series),
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(series) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outlier_values": sorted(outliers.tolist())[:50],  # first 50
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def trend_analysis(dataset_name: str, date_column: str, value_column: str) -> dict:
    """Analyze trend over time — linear regression on a time series.

    Args:
        dataset_name: Name of the dataset.
        date_column: Column with dates/timestamps.
        value_column: Numeric column to analyze for trend.

    Returns:
        dict: Trend direction, slope, R² value, and summary statistics per period.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        if date_column not in df.columns or value_column not in df.columns:
            return {"status": "error", "message": "Column(s) not found."}

        temp = df[[date_column, value_column]].dropna().copy()
        temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
        temp = temp.dropna().sort_values(date_column)

        if len(temp) < 3:
            return {"status": "error", "message": "Not enough data points."}

        # Numeric index for regression
        x = np.arange(len(temp))
        y = temp[value_column].values.astype(float)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

        return {
            "status": "success",
            "trend_direction": direction,
            "slope": round(float(slope), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 6),
            "date_range": {
                "start": str(temp[date_column].min()),
                "end": str(temp[date_column].max()),
            },
            "value_summary": {
                "mean": round(float(y.mean()), 4),
                "min": round(float(y.min()), 4),
                "max": round(float(y.max()), 4),
                "std": round(float(y.std()), 4),
            },
            "interpretation": f"The data shows an {direction} trend (slope={slope:.4f}, R²={r_value**2:.4f})",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def group_statistics(dataset_name: str, group_column: str, value_column: str) -> dict:
    """Compute summary statistics grouped by a categorical column.

    Args:
        dataset_name: Name of the dataset.
        group_column: Column to group by.
        value_column: Numeric column to aggregate.

    Returns:
        dict: Count, mean, sum, min, max per group.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        if group_column not in df.columns or value_column not in df.columns:
            return {"status": "error", "message": "Column(s) not found."}

        grouped = df.groupby(group_column)[value_column].agg(
            ["count", "mean", "sum", "min", "max", "std"]
        ).round(4)

        result = grouped.reset_index().to_dict(orient="records")

        return {
            "status": "success",
            "group_column": group_column,
            "value_column": value_column,
            "groups": result,
            "total_groups": len(result),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_data(
    dataset_name: str,
    sort_by: str = "",
    ascending: bool = True,
    top_n: int = 0,
    filter_column: str = "",
    filter_value: str = "",
    columns: str = "",
) -> dict:
    """Query, sort, filter, and rank data from a loaded dataset.

    Use this for questions like "top 5 highest paid", "sort by age",
    "filter where department is Engineering", "show name and salary columns".

    Args:
        dataset_name: Name of a previously loaded dataset.
        sort_by: Column name to sort by (leave empty for no sorting).
        ascending: Sort ascending (True) or descending (False). Use False for "top/highest".
        top_n: Number of rows to return (0 = all rows, 5 = top 5, etc.).
        filter_column: Column to filter on (leave empty for no filter).
        filter_value: Value to match in the filter column.
        columns: Comma-separated list of columns to include (empty = all columns).

    Returns:
        dict: Filtered/sorted data rows and summary.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        available = list_datasets()
        return {
            "status": "error",
            "message": f"Dataset '{dataset_name}' not found. Available: {available}",
        }

    try:
        result = df.copy()

        # Apply filter
        if filter_column and filter_value:
            if filter_column not in result.columns:
                return {"status": "error", "message": f"Column '{filter_column}' not found. Available: {list(result.columns)}"}
            result = result[result[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]

        # Select columns
        if columns:
            col_list = [c.strip() for c in columns.split(",")]
            valid_cols = [c for c in col_list if c in result.columns]
            if not valid_cols:
                return {"status": "error", "message": f"None of the requested columns found. Available: {list(result.columns)}"}
            result = result[valid_cols]

        # Apply sorting
        if sort_by:
            if sort_by not in result.columns:
                return {"status": "error", "message": f"Sort column '{sort_by}' not found. Available: {list(result.columns)}"}
            result = result.sort_values(by=sort_by, ascending=ascending)

        # Apply top_n
        if top_n > 0:
            result = result.head(top_n)

        return {
            "status": "success",
            "dataset_name": dataset_name,
            "rows_returned": len(result),
            "total_rows": len(df),
            "data": result.to_dict(orient="records"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def correlation_matrix(dataset_name: str) -> dict:
    """Compute the full correlation matrix for all numeric columns.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        dict: Correlation matrix as a nested dict.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return {"status": "error", "message": "No numeric columns found."}

        corr = numeric_df.corr().round(4)

        # Find strongest correlations (excluding self-correlation)
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append({
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "correlation": float(val),
                    })

        strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "status": "success",
            "matrix": corr.to_dict(),
            "strong_correlations": strong[:20],
            "columns": list(corr.columns),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
