"""
SQL tools — Database connection, query execution, and schema inspection.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

# Module-level engine cache
_engine = None


def connect_database(connection_string: str) -> dict:
    """Connect to a database using a SQLAlchemy connection string.

    Supported formats:
      - SQLite:     sqlite:///path/to/database.db
      - PostgreSQL: postgresql://user:pass@host:5432/dbname
      - MySQL:      mysql+pymysql://user:pass@host:3306/dbname

    Args:
        connection_string: A SQLAlchemy-compatible database connection string.

    Returns:
        dict: Connection status and available tables.
    """
    global _engine
    try:
        _engine = create_engine(connection_string)
        # Test connection
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        inspector = inspect(_engine)
        tables = inspector.get_table_names()

        return {
            "status": "success",
            "message": "Connected to database successfully.",
            "tables": tables,
        }
    except Exception as e:
        _engine = None
        return {"status": "error", "message": f"Connection failed: {e}"}


def execute_sql(query: str) -> dict:
    """Execute a SQL query and return the results.

    Use SELECT for reading data, or INSERT/UPDATE/DELETE for modifications.
    Results are returned as a list of row dictionaries.

    Args:
        query: The SQL query to execute.

    Returns:
        dict: Query results with column names and rows.
    """
    if _engine is None:
        return {"status": "error", "message": "No database connected. Use connect_database first."}

    try:
        with _engine.connect() as conn:
            result = conn.execute(text(query))

            # Check if query returns rows (SELECT)
            if result.returns_rows:
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return {
                    "status": "success",
                    "columns": columns,
                    "row_count": len(rows),
                    "rows": rows[:200],  # Limit to 200 rows for display
                    "truncated": len(rows) > 200,
                }
            else:
                conn.commit()
                return {
                    "status": "success",
                    "message": f"Query executed successfully. Rows affected: {result.rowcount}",
                }
    except Exception as e:
        return {"status": "error", "message": f"Query failed: {e}"}


def list_tables() -> dict:
    """List all tables in the connected database.

    Returns:
        dict: List of table names.
    """
    if _engine is None:
        return {"status": "error", "message": "No database connected. Use connect_database first."}

    try:
        inspector = inspect(_engine)
        tables = inspector.get_table_names()
        return {"status": "success", "tables": tables}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def describe_table(table_name: str) -> dict:
    """Describe a table's schema — columns, types, primary keys, and row count.

    Args:
        table_name: Name of the table to describe.

    Returns:
        dict: Column details, primary keys, and row count.
    """
    if _engine is None:
        return {"status": "error", "message": "No database connected. Use connect_database first."}

    try:
        inspector = inspect(_engine)
        columns = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)

        col_info = []
        for col in columns:
            col_info.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
            })

        # Get row count
        with _engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()

        return {
            "status": "success",
            "table_name": table_name,
            "columns": col_info,
            "primary_keys": pk.get("constrained_columns", []) if pk else [],
            "row_count": row_count,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
