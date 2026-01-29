"""
Store each analysis run in SQLite (so results persist)
Export doc-topic assignments as a downloadable CSV by run_id

"""

import os
import sqlite3

# Store DB inside backend/app/ so it’s easy to find
DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")

def get_connection() -> sqlite3.Connection:
    """
    Creates a SQLite connection.
    check_same_thread=False allows usage across FastAPI threads.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """
    Create tables if they don't exist.
    """
    conn = get_connection()
    cur = conn.cursor()

    # A "run" is one analysis execution
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        source_type TEXT NOT NULL,           -- "csv" or "txt"
        n_docs INTEGER NOT NULL,
        n_topics INTEGER NOT NULL,
        preprocess_options_json TEXT,
        topic_options_json TEXT,
        ingest_options_json TEXT
    );
    """)
    
    # Store topic terms per run
    cur.execute("""
    CREATE TABLE IF NOT EXISTS run_topics (
        run_id TEXT NOT NULL,
        topic_id INTEGER NOT NULL,
        top_terms_json TEXT NOT NULL,
        PRIMARY KEY (run_id, topic_id),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );
    """)
    
    # Store doc-level assignments (doc_id + group + dominant topic + cleaned preview)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS run_docs (
        run_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        doc_group TEXT NOT NULL,
        dominant_topic INTEGER NOT NULL,
        cleaned_text TEXT NOT NULL,
        PRIMARY KEY (run_id, doc_id),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    );
    
    """)
    
    conn.commit()
    conn.close()
