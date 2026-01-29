from pydantic import BaseModel
from typing import List, Dict, Optional


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    source_type: str
    n_docs: int
    n_topics: int
    

class RunsListResponse(BaseModel):
    #paging metadata - for pagination
    total: int
    offset: int
    limit: int
    runs: List[RunSummary]


class RunDetailResponse(BaseModel):
    run_id: str
    created_at: str
    source_type: str
    n_docs: int
    n_topics: int
    


    ingest_options_json: Optional[str] = None
    preprocess_options_json: Optional[str] = None
    topic_options_json: Optional[str] = None
    
    topics: List[Dict]                 # [{topic_id, top_terms}, ...]
    doc_preview: List[Dict]            # [{doc_id, group, dominant_topic, cleaned_text_preview}, ...]
    topic_counts: Dict[int, int]
    
    
# Compare two runs
class RunCompareResponse(BaseModel):
    run_id_a: str
    run_id_b: str
    meta: Dict
    topic_overlap: List[Dict]   # [{topic_a, topic_b, overlap_count, overlap_terms}, ...]