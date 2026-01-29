from pydantic import BaseModel, Field
from typing import List, Optional

from app.schemas.ingestion import DocumentRecord


class AnalyzeUploadResponse(BaseModel):
    # ingestion summary
    n_docs: int
    n_groups: int
    groups_preview: List[str]
    
    
    # preprocessing summary
    empty_after_cleaning: int
    avg_cleaned_tokens: float
    
    # previews to sanity-check results
    preview_docs_raw: List[DocumentRecord]
    preview_docs_clean: List[dict] = Field(
        ..., description="List of {doc_id, group, cleaned_text}"
    ) 