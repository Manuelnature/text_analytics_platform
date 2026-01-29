from pydantic import BaseModel, Field  
from typing import List, Optional

class PreprocessRequest(BaseModel):
    # Accept raw texts (simple for now)
    texts: List[str] = Field(..., description="List of raw documents to preprocess")
    
    # Options (I will map to PreprocessConfig)
    min_token_len: int = 4
    remove_stopwords: bool = True
    lemmatize: bool = True
    remove_urls: bool = True
    keep_only_letters: bool = True
    lowercase: bool = True

    extra_stopwords: Optional[List[str]] = None
    
    
class PreprocessResponse(BaseModel):
    n_docs: int
    preview_raw: List[str]
    preview_clean: List[str]
    empty_after_cleaning: int