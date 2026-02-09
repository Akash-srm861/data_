"""
Streamlit Chat UI for the Data Analyst Agentic AI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys

import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables ──
load_dotenv()

# Also load from Streamlit secrets (for Streamlit Cloud deployment)
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # No secrets available (local dev)

# ── Page config ──
st.set_page_config(
    page_title="Data Analyst AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Check API key ──
groq_key = os.getenv("GROQ_API_KEY", "")
if not groq_key:
    st.error(
        "⚠️ **GROQ_API_KEY not set!** \n\n"
        "1. Get an API key from [Groq Console](https://console.groq.com/keys)\n"
        "2. Add it to the `.env` file: `GROQ_API_KEY=your-key-here`\n"
        "3. Restart the app"
    )
    st.stop()

# ── Lazy imports (after env is loaded) ──
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from data_analyst.agent import root_agent
from data_analyst.utils.state_manager import OUTPUTS_DIR, UPLOADS_DIR

# ── Constants ──
APP_NAME = "data_analyst_ai"
USER_ID = "streamlit_user"


# ── Session state initialization ──
def _create_runner_and_session(session_id: str):
    """Create a Runner + session in one go."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    # Actually create the session inside the service so it exists when run_async looks for it
    loop = asyncio.new_event_loop()
    loop.run_until_complete(
        session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    )
    loop.close()
    return runner


def init_session():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session_001"
    if "runner" not in st.session_state:
        st.session_state.runner = _create_runner_and_session(st.session_state.session_id)


init_session()


# ── Helper: Run agent ──
async def run_agent_async(user_message: str) -> str:
    """Send a message to the agent and collect the response."""
    from google.genai import types

    runner: Runner = st.session_state.runner

    content = types.Content(
        role="user", parts=[types.Part(text=user_message)]
    )

    response_parts = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=st.session_state.session_id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_parts.append(part.text)

    return "\n".join(response_parts) if response_parts else "I couldn't generate a response. Please try again."


def run_agent(user_message: str) -> str:
    """Synchronous wrapper for running the agent."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run_agent_async(user_message))
    finally:
        loop.close()


# ── Sidebar ──
with st.sidebar:
    st.title("📊 Data Analyst AI")
    st.markdown("---")

    # File uploader
    st.subheader("📁 Upload Data Files")
    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, or PDF files",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Auto-load into memory so the agent can use it immediately
            fname = uploaded_file.name.lower()
            dataset_name = os.path.splitext(uploaded_file.name)[0]
            try:
                if fname.endswith(".csv"):
                    import pandas as pd
                    from data_analyst.utils.state_manager import store_dataframe
                    df = pd.read_csv(file_path)
                    store_dataframe(dataset_name, df)
                    st.success(f"✅ {uploaded_file.name} — loaded ({len(df)} rows)")
                elif fname.endswith((".xlsx", ".xls")):
                    import pandas as pd
                    from data_analyst.utils.state_manager import store_dataframe
                    df = pd.read_excel(file_path)
                    store_dataframe(dataset_name, df)
                    st.success(f"✅ {uploaded_file.name} — loaded ({len(df)} rows)")
                else:
                    st.success(f"✅ {uploaded_file.name} — saved (use 'load' to parse)")
            except Exception as e:
                st.warning(f"⚠️ {uploaded_file.name} saved but auto-load failed: {e}")

    st.markdown("---")

    # Database connection
    st.subheader("🗄️ Database Connection")
    db_string = st.text_input(
        "Connection string",
        placeholder="sqlite:///my_database.db",
        help="SQLite: sqlite:///path.db | PostgreSQL: postgresql://user:pass@host/db",
    )

    if db_string:
        st.session_state["db_connection_string"] = db_string
        st.info(f"📎 Connection string saved")

    st.markdown("---")

    # Outputs section
    st.subheader("📥 Generated Outputs")
    if os.path.exists(OUTPUTS_DIR):
        output_files = [
            f for f in os.listdir(OUTPUTS_DIR)
            if os.path.isfile(os.path.join(OUTPUTS_DIR, f)) and f != ".gitkeep"
        ]
        if output_files:
            for fname in sorted(output_files, reverse=True)[:10]:
                fpath = os.path.join(OUTPUTS_DIR, fname)
                with open(fpath, "rb") as f:
                    st.download_button(
                        label=f"⬇️ {fname}",
                        data=f.read(),
                        file_name=fname,
                        key=f"dl_{fname}",
                    )
        else:
            st.caption("No outputs generated yet.")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        new_sid = f"session_{abs(hash(os.urandom(8)))}"
        st.session_state.session_id = new_sid
        st.session_state.runner = _create_runner_and_session(new_sid)
        st.rerun()

    # Quick actions — dynamic based on loaded data
    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    from data_analyst.utils.state_manager import list_datasets
    loaded = list_datasets()

    if not loaded:
        # No data loaded yet — show upload/load actions
        quick_actions = [
            ("📁 List uploaded files", "List uploaded files and loaded datasets"),
            ("📂 Load sample_data.csv", "load sample_data.csv"),
        ]
    else:
        ds = loaded[0]  # Use first loaded dataset
        quick_actions = [
            ("📁 List uploaded files", "List uploaded files and loaded datasets"),
            (f"📊 Describe {ds}", f"describe {ds}"),
            (f"📈 Bar chart (Salary by Dept)", f"create a bar chart of Salary by Department from {ds}"),
            (f"🔍 Top 5 highest salaries", f"show top 5 employees with highest Salary from {ds}"),
            (f"📉 Correlation heatmap", f"create a heatmap for {ds}"),
        ]

    for label, action_text in quick_actions:
        if st.button(label, use_container_width=True, key=f"qa_{label}"):
            st.session_state.pending_message = action_text
            st.rerun()


# ── Main chat area ──
st.title("📊 Data Analyst AI")
st.caption("Powered by Google ADK + Groq (Llama 4 Scout) · Data Analysis System")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display charts inline if referenced
        if msg["role"] == "assistant" and "chart_path" in msg.get("extra", {}):
            chart_path = msg["extra"]["chart_path"]
            if os.path.exists(chart_path):
                st.image(chart_path)

# Determine the prompt: either from chat input or from a queued quick action
prompt = st.chat_input("Ask me anything about your data...")
if not prompt and "pending_message" in st.session_state:
    prompt = st.session_state.pop("pending_message")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):
            try:
                response = run_agent(prompt)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower():
                    response = "⏳ Rate limit hit. Please wait 30 seconds and try again."
                else:
                    response = f"❌ Error: {error_msg}"

        st.markdown(response)

        # Check for chart files in the response and display them
        if os.path.exists(OUTPUTS_DIR):
            for fname in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
                if fname.endswith(".png") and fname != ".gitkeep":
                    fpath = os.path.join(OUTPUTS_DIR, fname)
                    # Show if created recently (within last 60 seconds)
                    import time
                    if time.time() - os.path.getmtime(fpath) < 60:
                        st.image(fpath, caption=fname)
                        break

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Auto-intro on first load ──
if not st.session_state.messages:
    intro = (
        "👋 Hi! I'm your **Data Analyst AI** — I can help you load data, "
        "run analyses, create charts, and generate reports.\n\n"
        "Here's what I can do:\n"
        "- 📁 **Load Data** — CSV, Excel, PDF files\n"
        "- 🗄️ **Query Databases** — SQL with natural language\n"
        "- 📊 **Analyze** — Statistics, correlations, trends, outliers\n"
        "- 📈 **Visualize** — Bar, line, scatter, pie charts & heatmaps\n"
        "- 📝 **Generate Reports** — Markdown & PDF\n"
        "- 🌐 **Scrape Web Data** — Websites, tables, APIs\n\n"
        "**Upload a file** in the sidebar or **tell me what you'd like to analyze!**"
    )
    st.session_state.messages.append({"role": "assistant", "content": intro})
    st.rerun()
