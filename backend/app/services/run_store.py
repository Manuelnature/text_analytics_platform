import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

from app.db import get_connection

def create_run(
    source_type: str,
    n_docs: int,
    n_topics: int,
    ingest_options_json: Optional[str],
    preprocess_options_json: Optional[str],
    topic_options_json: Optional[str],
) -> str:
    run_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        """
        INSERT INTO runs(run_id, created_at, source_type, n_docs, n_topics,
                         preprocess_options_json, topic_options_json, ingest_options_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at,
            source_type,
            n_docs,
            n_topics,
            preprocess_options_json,
            topic_options_json,
            ingest_options_json,
        )
    )
    conn.commit()
    conn.close()
    return run_id


def save_topics(run_id: str, topics: List[Dict]) -> None:
    """
    topics = [{"topic_id": i, "top_terms": [...]}, ...]
    """
    conn = get_connection()
    cur = conn.cursor()

    for t in topics:
        cur.execute(
            """
            INSERT OR REPLACE INTO run_topics(run_id, topic_id, top_terms_json)
            VALUES (?, ?, ?)
            """,
            (run_id, int(t["topic_id"]), json.dumps(t["top_terms"]))
        )
        
    conn.commit()
    conn.close()
    
    

def save_doc_assignments(run_id: str, rows: List[Dict]) -> None:
    """
    rows = [{"doc_id":..., "group":..., "dominant_topic":..., "cleaned_text":...}, ...]
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT OR REPLACE INTO run_docs(run_id, doc_id, doc_group, dominant_topic, cleaned_text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (run_id, r["doc_id"], r["group"], int(r["dominant_topic"]), r["cleaned_text"])
            for r in rows
        ]
    )
    
    conn.commit()
    conn.close()
