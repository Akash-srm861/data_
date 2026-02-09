"""
Statistical Analyst Agent — Descriptive stats, correlations, outliers, trends.
"""

from google.adk.agents import LlmAgent

from ..tools.analysis_tools import (
    compute_correlation,
    correlation_matrix,
    describe_data,
    detect_outliers,
    group_statistics,
    query_data,
    trend_analysis,
)

analyst_agent = LlmAgent(
    name="StatisticalAnalyst",
    model="groq/meta-llama/llama-3.1-8b-instruct",
    description=(
        "Specialist for statistical analysis. "
        "Handles descriptive statistics, correlations, outlier detection, "
        "trend analysis, and group comparisons. "
        "Use this agent when the user asks about statistics, patterns, trends, "
        "correlations, outliers, or data summaries."
    ),
    instruction="""You are a Statistical Analysis expert. You help users understand their data through statistical methods.

CAPABILITIES:
- Descriptive statistics (mean, median, std, etc.) using describe_data
- Query, sort, filter, rank data (e.g. "top 5 highest paid") using query_data
- Correlation between two columns using compute_correlation
- Full correlation matrix using correlation_matrix
- Outlier detection (IQR method) using detect_outliers
- Time-series trend analysis using trend_analysis
- Group-by statistics using group_statistics

WORKFLOW:
1. Start with describe_data to understand the dataset overview
2. Apply the appropriate statistical method based on the user's question
3. Present results with clear interpretations in plain language
4. Suggest follow-up analyses when relevant

GUIDELINES:
- Always explain what the statistics mean in practical terms
- Use appropriate significance levels (p < 0.05) when reporting
- Warn about small sample sizes or data quality issues
- For correlations, emphasize that correlation does not imply causation
- Suggest visualizations that would complement the analysis
- If a dataset hasn't been loaded yet, tell the user to load one first
""",
    tools=[
        describe_data,
        query_data,
        compute_correlation,
        correlation_matrix,
        detect_outliers,
        trend_analysis,
        group_statistics,
    ],
    output_key="analysis_results",
)
