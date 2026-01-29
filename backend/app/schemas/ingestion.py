"""
### Define ingestion schemas ###

This part defines the data contracts for your backend:

What options a user can pass when uploading TXT/CSV (e.g., grouping strategy, header pattern, which CSV column contains text).

The standard internal document format your app will use everywhere (doc_id, text, group, source).

The response format returned by your ingestion endpoints (number of docs, detected groups, preview records).
"""

"""
Ingestion Schemas (Pydantic models)

Why this file exists:
- Defines the *validated* input options for ingestion endpoints (TXT/CSV).
- Defines the canonical internal representation of a "document".
- Defines the response schema returned by ingestion endpoints.

This helps:
- Avoid messy dictionaries everywhere
- Ensure your API contracts are consistent
- Make the frontend integration much easier
"""

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class GroupStrategy(str, Enum):
    # For TXT: group header lines begin with a pattern, e.g. "#####"
    HEADER = "header"
    # For CSV: group is taken from a specified column
    CSV_COLUMN = "csv_column"
    # No grouping; everything is assigned to a default group
    NONE = "none"
    

########## For Text files #############
"""
class TxtIngestOptions(BaseModel):
    #Options for ingesting TXT files
    
    # How grouping is determined for TXT files
    group_strategy: GroupStrategy = Field(
        default=GroupStrategy.NONE,
        description="How to assign groups for TXT ingestion: header-based, none, etc."
    )
    
    # Used only when group_strategy == HEADER
    # Example: header_pattern="#####"
    header_pattern: Optional[str] = Field(
        default="#####",
        description="Prefix that identifies a group header line. Used only for HEADER strategy."
    )
    
    # How to split documents within each group.
    # - blank_lines: split on empty lines (best for paragraphs-as-documents)
    # - lines: treat each non-empty line as a document
    split_mode: str = Field(
        default="blank_lines",
        description="How to split documents: 'blank_lines' (paragraphs) or 'lines'."
    )
    
    # If a dataset has no groups, or group header is empty,
    # The default group label is assigned.
    default_group: str = Field(
        default="default",
        description="Fallback group name when groups are missing."
    )
"""



class TxtIngestOptions(BaseModel):
    group_strategy: GroupStrategy = Field(default=GroupStrategy.NONE)

    # NEW: how headers are represented when group_strategy=HEADER
    # - inline: "##### University Name"
    # - banner: "#####" line, then title line, then "#####" line
    header_mode: str = Field(
        default="inline",
        description="Header style when group_strategy=HEADER: 'inline' or 'banner'"
    )

    header_pattern: Optional[str] = Field(default="#####")
    split_mode: str = Field(default="blank_lines")
    default_group: str = Field(default="default")

    # NEW: helps prevent garbage group names (URLs/paragraphs)
    max_group_name_len: int = Field(default=120)


########## For CSV files #############
class CsvIngestOptions(BaseModel):
    # Options for ingesting CSV files.
    
    # REQUIRED: which column contains the raw text?
    text_column: str = Field(
        ...,
        description="Name of the CSV column that contains the raw text to analyze."
    )
    
    # OPTIONAL: which column contains a unique ID?
    # If missing, we auto-generate an ID.
    id_column: Optional[str] = Field(
        default=None,
        description="Optional: column that contains unique document IDs."
    )
    
    # OPTIONAL: which column contains the group/category?
    # If missing, we assign default_group.
    group_column: Optional[str] = Field(
        default=None,
        description="Optional: column that contains group/category names."
    )
    
    # Default group name if group_column is absent or empty
    default_group: str = Field(
        default="default",
        description="Fallback group name when group_column is missing or empty."
    )
    
    
    
    

class DocumentRecord(BaseModel):
    """
    Canonical internal representation of a document.

    No matter how the dataset is uploaded (TXT/CSV),
    we convert it into this format so downstream steps are consistent.

    Fields:
    - doc_id: unique identifier
    - text: raw document text (preprocessing happens later)
    - group: optional grouping label (university/category/etc.)
    - source: where the document came from (thesis_txt, txt_upload, csv_upload...)
    """
    doc_id: str
    text: str
    group: str
    source: str
    



class IngestResponse(BaseModel):
    """
    What the ingestion endpoints return.

    We return:
    - how many docs we extracted
    - which groups we detected
    - a small preview list of documents (first 10) for debugging/UI

    This is extremely useful for:
    - validating ingestion worked
    - allowing the frontend to show a preview before analysis
    """
    n_docs: int
    groups: List[str]
    preview: List[DocumentRecord]