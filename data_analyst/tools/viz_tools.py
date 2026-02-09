"""
Visualization tools — Generate charts and plots using matplotlib and plotly.
"""

from __future__ import annotations

import os
import uuid

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from ..utils.state_manager import get_dataframe, get_output_path


def _save_matplotlib(fig, title: str) -> str:
    """Save a matplotlib figure and return the file path."""
    filename = f"{title.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.png"
    path = get_output_path(filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _save_plotly(fig, title: str) -> str:
    """Save a plotly figure as PNG and return the file path."""
    filename = f"{title.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.png"
    path = get_output_path(filename)
    pio.write_image(fig, path, width=1000, height=600, scale=2)
    return path


def create_bar_chart(
    dataset_name: str,
    x_column: str,
    y_column: str,
    title: str = "Bar Chart",
    color_column: str = "",
    top_n: int = 0,
) -> dict:
    """Create a bar chart from a dataset.

    Args:
        dataset_name: Name of the loaded dataset.
        x_column: Column for the x-axis (categories).
        y_column: Column for the y-axis (values).
        title: Chart title.
        color_column: Optional column for color grouping.
        top_n: If > 0, show only the top N categories by value.

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        plot_df = df.copy()
        if top_n > 0:
            plot_df = plot_df.nlargest(top_n, y_column)

        kwargs = {"x": x_column, "y": y_column, "title": title}
        if color_column and color_column in df.columns:
            kwargs["color"] = color_column

        fig = px.bar(plot_df, **kwargs)
        fig.update_layout(template="plotly_white")
        path = _save_plotly(fig, title)

        return {"status": "success", "chart_path": path, "chart_type": "bar"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_line_chart(
    dataset_name: str,
    x_column: str,
    y_column: str,
    title: str = "Line Chart",
    color_column: str = "",
) -> dict:
    """Create a line chart from a dataset.

    Args:
        dataset_name: Name of the loaded dataset.
        x_column: Column for the x-axis.
        y_column: Column for the y-axis.
        title: Chart title.
        color_column: Optional column for color grouping (multiple lines).

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        kwargs = {"x": x_column, "y": y_column, "title": title}
        if color_column and color_column in df.columns:
            kwargs["color"] = color_column

        fig = px.line(df, **kwargs)
        fig.update_layout(template="plotly_white")
        path = _save_plotly(fig, title)

        return {"status": "success", "chart_path": path, "chart_type": "line"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_scatter_plot(
    dataset_name: str,
    x_column: str,
    y_column: str,
    title: str = "Scatter Plot",
    color_column: str = "",
    size_column: str = "",
) -> dict:
    """Create a scatter plot from a dataset.

    Args:
        dataset_name: Name of the loaded dataset.
        x_column: Column for the x-axis.
        y_column: Column for the y-axis.
        title: Chart title.
        color_column: Optional column for color grouping.
        size_column: Optional column for point sizes.

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        kwargs = {"x": x_column, "y": y_column, "title": title}
        if color_column and color_column in df.columns:
            kwargs["color"] = color_column
        if size_column and size_column in df.columns:
            kwargs["size"] = size_column

        fig = px.scatter(df, **kwargs)
        fig.update_layout(template="plotly_white")
        path = _save_plotly(fig, title)

        return {"status": "success", "chart_path": path, "chart_type": "scatter"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_histogram(
    dataset_name: str,
    column: str,
    title: str = "Histogram",
    bins: int = 30,
    color_column: str = "",
) -> dict:
    """Create a histogram for a numeric column.

    Args:
        dataset_name: Name of the loaded dataset.
        column: Numeric column to plot.
        title: Chart title.
        bins: Number of bins.
        color_column: Optional column for color grouping.

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        kwargs = {"x": column, "title": title, "nbins": bins}
        if color_column and color_column in df.columns:
            kwargs["color"] = color_column

        fig = px.histogram(df, **kwargs)
        fig.update_layout(template="plotly_white")
        path = _save_plotly(fig, title)

        return {"status": "success", "chart_path": path, "chart_type": "histogram"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_pie_chart(
    dataset_name: str,
    names_column: str,
    values_column: str,
    title: str = "Pie Chart",
    top_n: int = 10,
) -> dict:
    """Create a pie chart from a dataset.

    Args:
        dataset_name: Name of the loaded dataset.
        names_column: Column for slice labels.
        values_column: Column for slice sizes.
        title: Chart title.
        top_n: Show top N slices; rest become "Other".

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        plot_df = df.groupby(names_column)[values_column].sum().reset_index()
        plot_df = plot_df.sort_values(values_column, ascending=False)

        if len(plot_df) > top_n:
            top = plot_df.head(top_n)
            other_val = plot_df.iloc[top_n:][values_column].sum()
            other_row = {names_column: "Other", values_column: other_val}
            plot_df = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)
        else:
            plot_df = plot_df

        fig = px.pie(plot_df, names=names_column, values=values_column, title=title)
        path = _save_plotly(fig, title)

        return {"status": "success", "chart_path": path, "chart_type": "pie"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_heatmap(
    dataset_name: str,
    title: str = "Correlation Heatmap",
) -> dict:
    """Create a correlation heatmap for all numeric columns in the dataset.

    Args:
        dataset_name: Name of the loaded dataset.
        title: Chart title.

    Returns:
        dict: Status and file path of the generated chart.
    """
    df = get_dataframe(dataset_name)
    if df is None:
        return {"status": "error", "message": f"Dataset '{dataset_name}' not found."}

    try:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return {"status": "error", "message": "No numeric columns found."}

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        # Add correlation values as text
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=14)
        fig.tight_layout()

        path = _save_matplotlib(fig, title)
        return {"status": "success", "chart_path": path, "chart_type": "heatmap"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Need pandas for pie chart grouping
import pandas as pd
