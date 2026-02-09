"""
Data Analyst Agent — Single unified agent with all tools.

This is the main entry point for the Data Analyst Agentic AI system.
Instead of multi-agent delegation (which requires transfer_to_agent and
fails with non-Gemini models), we use one agent with all tools directly.
"""

from google.adk.agents import LlmAgent

# File tools
from .tools.file_tools import list_uploaded_files, load_csv, load_excel, load_pdf

# Analysis tools
from .tools.analysis_tools import (
    compute_correlation,
    correlation_matrix,
    describe_data,
    detect_outliers,
    group_statistics,
    query_data,
    trend_analysis,
)

# SQL tools
from .tools.sql_tools import connect_database, describe_table, execute_sql, list_tables

# Visualization tools
from .tools.viz_tools import (
    create_bar_chart,
    create_heatmap,
    create_histogram,
    create_line_chart,
    create_pie_chart,
    create_scatter_plot,
)

# Report tools
from .tools.report_tools import generate_markdown_report, generate_pdf_report

# Web scraping tools
from .tools.web_tools import fetch_api_data, scrape_table, scrape_webpage

root_agent = LlmAgent(
    name="DataAnalyst",
    model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    description="Data Analyst AI that uses tools to load, analyze, and visualize data.",
    instruction="""You are a Data Analyst AI. You MUST use your tools for ALL tasks.
You work with ANY kind of data — employee records, Aadhar data, industry statistics, sales, healthcare, census, financial, or anything else the user uploads.

CRITICAL RULES:
1. NEVER write code. NEVER output Python, SQL, or JavaScript. NEVER use code blocks.
2. NEVER invent, assume, or hallucinate data. ONLY report results returned by your tools.
3. NEVER describe charts, statistics, or analysis you did not actually perform with a tool call.
4. If no dataset is loaded yet, tell the user: "Please load a dataset first using the sidebar or ask me to load a file."
5. Keep responses concise. State the result, give a brief insight, suggest one next step.
6. ALWAYS call a tool. If the user asks for analysis, call the analysis tool. If they ask for a chart, call the chart tool.
7. Use the ACTUAL column names from the loaded dataset. Do NOT assume column names like Salary, Department, etc.

dataset_name = filename without extension. Example: "industries.csv" -> dataset_name is "industries".

EXAMPLES:
- User: "load my_data.csv" -> call load_csv(filename="my_data.csv")
- User: "describe the data" -> call describe_data(dataset_name="my_data")
- User: "bar chart of revenue by region" -> call create_bar_chart(dataset_name="my_data", x_column="region", y_column="revenue", title="Revenue by Region")
- User: "top 5 rows by population" -> call query_data(dataset_name="my_data", sort_by="population", ascending=false, top_n=5)
- User: "filter where state is Maharashtra" -> call query_data(dataset_name="my_data", filter_column="state", filter_value="Maharashtra")
- User: "correlation between age and income" -> call compute_correlation(dataset_name="my_data", col1="age", col2="income")
- User: "group by sector" -> call group_statistics(dataset_name="my_data", group_column="sector", value_column="revenue")

RESPONSE FORMAT:
- After a tool returns results, summarize them in 2-4 sentences.
- Use a small table if showing rows of data.
- Do NOT repeat the same information in different words.
- Do NOT say "I will now..." or "Let me...". Just call the tool and report the result.

IMPORTANT: Data is ALREADY loaded in memory after load_csv. You do NOT need to load it again. Just use the dataset_name in subsequent tool calls.""",
    tools=[
        # File tools
        load_csv,
        load_excel,
        load_pdf,
        list_uploaded_files,
        # Analysis tools
        describe_data,
        query_data,
        compute_correlation,
        correlation_matrix,
        detect_outliers,
        trend_analysis,
        group_statistics,
        # SQL tools
        connect_database,
        list_tables,
        describe_table,
        execute_sql,
        # Visualization tools
        create_bar_chart,
        create_line_chart,
        create_scatter_plot,
        create_histogram,
        create_pie_chart,
        create_heatmap,
        # Report tools
        generate_markdown_report,
        generate_pdf_report,
        # Web scraping tools
        scrape_webpage,
        scrape_table,
        fetch_api_data,
    ],
)
