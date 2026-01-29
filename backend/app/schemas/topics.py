from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TopicModelResponse(BaseModel):
    run_id: Optional[str] = None
    
    n_docs: int
    n_topics: int

    # topic_id -> top terms
    topics: List[Dict] = Field(..., description="List of {topic_id, top_terms}")

    # doc results preview
    doc_preview: List[Dict] = Field(..., description="List of {doc_id, group, dominant_topic, cleaned_text_preview}")

    # distribution summary
    topic_counts: Dict[int, int]
    empty_after_cleaning: int