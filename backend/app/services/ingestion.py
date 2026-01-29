from __future__ import annotations

"""
### Ingestion service ###

It provides two main functions:

ingest_txt(...):
Reads a TXT upload and converts it into documents, with optional grouping:

HEADER grouping (thesis format): group headers like ##### University Name

NONE: no grouping; everything goes into a default group
It also splits the text into documents (paragraphs by blank lines, or line-by-line).

ingest_csv(...):
Reads a CSV upload and extracts documents from a chosen text column, optionally using an ID column and a group column.

Output is always the same canonical schema:
doc_id | text | group | source

"""


"""
Ingestion Service

Purpose:
- Convert uploaded TXT/CSV files into a standard list of DocumentRecord objects.
- Support optional grouping (e.g., university/category).
- Keep all downstream ML logic dataset-agnostic by enforcing one canonical schema.

Key design idea:
- Regardless of input type (TXT/CSV), we always output:
    doc_id, text, group, source

This file does NOT do preprocessing (cleaning/tokenization).
Preprocessing should live in ml/preprocessing.py and be applied later.
"""



import io
import re
import uuid
from typing import List

import pandas as pd

from app.schemas.ingestion import (
    GroupStrategy,
    TxtIngestOptions,
    CsvIngestOptions,
    DocumentRecord,
)


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def _new_id() -> str:
    """
    Generate a short unique ID for documents when an ID is not provided by the dataset.
    """
    return uuid.uuid4().hex[:12]


def _normalize_line_endings(text:string) -> str:
    """
    Normalize line endings so parsing behaves consistently across Windows/Mac/Linux.

    - Windows often uses \r\n
    - Old Mac sometimes uses \r
    - Linux uses \n
    We convert everything to '\n'.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_blocks_blank_lines(text: str) -> List[str]:
    """
    Split text into "documents" by blank lines.

    Example:
        paragraph 1

        paragraph 2

    becomes:
        ["paragraph 1", "paragraph 2"]

    This is ideal for thesis-like corpora where paragraphs represent distinct documents.
    """
    blocks = re.split(r"\n\s*\n+", text.strip())  # split on 1+ empty lines
    return [b.strip() for b in blocks if b.strip()]


def _split_blocks_lines(text: str) -> List[str]:
    """
    Split text into "documents" line-by-line (each non-empty line is a doc).

    This is useful for datasets where each line is one entry, e.g.:
        review 1
        review 2
        review 3
    """
    blocks = [ln.strip() for ln in text.split("\n")]
    return [b for b in blocks if b]


def _safe_decode(file_bytes: bytes) -> str:
    """
    Decode uploaded files safely.

    We try UTF-8 first (most common). If decoding fails,
    fallback to latin-1 and replace problematic characters.

    This prevents ingestion from crashing due to encoding issues.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")



# ---------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------
def ingest_txt(
    file_bytes: bytes,
    options: TxtIngestOptions,
    source: str = "txt_upload") -> List[DocumentRecord]:
     
    """
    Ingest a TXT file and convert it into a list of DocumentRecord objects.

    Supports two main styles:
    1) NO GROUPING:
       - The whole file is split into documents and assigned to default_group.

    2) HEADER GROUPING (thesis-style):
       - A group header line starts with a pattern (e.g. "#####").
       - Everything after that header belongs to that group until next header.
       - The group content is split into documents.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded TXT file.
    options:
        TxtIngestOptions controlling grouping and splitting behavior.
    source:
        A string tag to track origin (e.g., "thesis_txt", "txt_upload").

    Returns
    -------
    List[DocumentRecord]
        Standardized documents ready for preprocessing and modeling.
    """
    
    # Decode and normalize the raw text
    raw = _safe_decode(file_bytes)
    raw = _normalize_line_endings(raw)

    # Choose how to split docs:
    # - blank_lines: split by paragraphs
    # - lines: split by line
    splitter = _split_blocks_blank_lines if options.split_mode == "blank_lines" else _split_blocks_lines

    docs: List[DocumentRecord] = []
    
    
    # -----------------------------
    # Case 1: No grouping
    # -----------------------------
    if options.group_strategy == GroupStrategy.NONE:
        blocks = splitter(raw)
        for block in blocks:
            docs.append(
                DocumentRecord(
                    doc_id=_new_id(),
                    text=block,
                    group=options.default_group,
                    source=source,
                )
            )
        return docs

    # -----------------------------
    # Case 2: Grouping via headers
    # -----------------------------
    #expecting_group_name = False

    """
    if options.group_strategy == GroupStrategy.HEADER:
        if not options.header_pattern:
            raise ValueError("header_pattern must be provided when group_strategy=HEADER")

        current_group = options.default_group
        buffer_lines: List[str] = []  # stores lines belonging to the current group

        def flush_buffer_as_docs(group_name: str):
            
            # Convert the accumulated lines for a group into documents and append to `docs`.
           
            joined = "\n".join(buffer_lines).strip()
            if not joined:
                return

            # Split group content into documents
            for block in splitter(joined):
                docs.append(
                    DocumentRecord(
                        doc_id=_new_id(),
                        text=block,
                        group=group_name,
                        source=source,
                    )
                )

        hash_line_re = re.compile(r"^\s*#{5,}\s*$")  # a line made of 5+ hashes (optionally with spaces)
        
        # Iterate line-by-line to detect group headers
        for line in raw.split("\n"):
            stripped = line.strip()

            # If a line starts with the header pattern, it indicates a new group
            if stripped.startswith(options.header_pattern):
                # Flush the previous group's buffered content into docs
                flush_buffer_as_docs(current_group)
                buffer_lines = []

                # Extract group name after the pattern
                group_name = stripped[len(options.header_pattern):].strip()

                # If group name is empty, fallback to default
                current_group = group_name if group_name else options.default_group

            else:
                # Regular content line: add to current group's buffer
                buffer_lines.append(line)

        # Flush the last group after finishing the loop
        flush_buffer_as_docs(current_group)
        return docs
        
    
        # Case 1: Inline header like "##### University Name"
        if options.header_pattern and stripped.startswith(options.header_pattern) and not hash_line_re.match(stripped):
            flush_buffer_as_docs(current_group)
            buffer_lines = []
            group_name = stripped[len(options.header_pattern):].strip()
            current_group = group_name if group_name else options.default_group
            continue

        # Case 2: Banner header line like "####################"
        if hash_line_re.match(stripped):
            # Flush previous group content
            flush_buffer_as_docs(current_group)
            buffer_lines = []
            # We don't set the group yet — next non-empty line will be the group name
            current_group = options.default_group
            expecting_group_name = True
            continue

        # If we just saw a hash banner, the next non-empty line is the group name
        if 'expecting_group_name' in locals() and expecting_group_name and stripped:
            current_group = stripped
            expecting_group_name = False
            continue

        buffer_lines.append(line)
    
    # If we reach here, the strategy is not supported for TXT
    raise ValueError(f"Unsupported group strategy for TXT: {options.group_strategy}")

    """
    
    """
    # =========== Another Worked but not extracting the correct groups ==================
    # Case 2: header-based grouping (supports both inline and banner styles)
    if options.group_strategy == GroupStrategy.HEADER:
        # A "banner" header line is a line made only of hashes, e.g. "####################"
        hash_line_re = re.compile(r"^\s*#{5,}\s*$")

        current_group = options.default_group

        # We temporarily store lines for the current group here, then "flush" them into docs
        buffer_lines: List[str] = []

        # State flag:
        # - When we see a banner hash line, the next non-empty line is treated as the group name
        expecting_group_name = False

        def flush_buffer_as_docs(group_name: str) -> None:
            
            #Convert the currently buffered lines into documents and append them to `docs`.

            #We join buffered lines into one string, then split into documents using the chosen split mode
            #(blank lines or line-by-line).
            
            joined = "\n".join(buffer_lines).strip()
            if not joined:
                return

            for block in splitter(joined):
                docs.append(
                    DocumentRecord(
                        doc_id=_new_id(),
                        text=block,
                        group=group_name,
                        source=source,
                    )
                )

        for line in raw.split("\n"):
            stripped = line.strip()

            # -------------------------------------------------------
            # Banner style headers (your dataset)
            #
            # Example:
            #   ########################
            #   University of Turin
            #   ########################
            # -------------------------------------------------------
            if hash_line_re.match(stripped):
                # A banner hash line marks a boundary between groups.
                # Flush whatever we collected so far under the current group.
                flush_buffer_as_docs(current_group)
                buffer_lines = []

                # Next non-empty line should be the group name
                expecting_group_name = True
                continue

            if expecting_group_name:
                # Skip empty lines between banner and group name (if any)
                if not stripped:
                    continue

                # The first non-empty line after a banner becomes the group name
                current_group = stripped
                expecting_group_name = False
                continue

            # -------------------------------------------------------
            # Inline headers (optional support)
            #
            # Example:
            #   ##### University of Milan
            #
            # NOTE:
            # If someone uploads a dataset that uses inline headers, this will work too.
            # We also guard against the header_pattern accidentally matching banner-only hash lines.
            # -------------------------------------------------------
            if options.header_pattern and stripped.startswith(options.header_pattern) and not hash_line_re.match(stripped):
                flush_buffer_as_docs(current_group)
                buffer_lines = []

                group_name = stripped[len(options.header_pattern):].strip()
                current_group = group_name if group_name else options.default_group
                continue

            # Normal content line: keep accumulating it for the current group
            buffer_lines.append(line)

        # Flush remaining buffered text at end of file
        flush_buffer_as_docs(current_group)
        return docs
    
    
    # If we reach here, the strategy is not supported for TXT
    raise ValueError(f"Unsupported group strategy for TXT: {options.group_strategy}")

    """
    
    
    if options.group_strategy == GroupStrategy.HEADER:
        hash_line_re = re.compile(r"^\s*#{5,}\s*$")
        url_re = re.compile(r"^https?://", re.IGNORECASE)

        def looks_like_group_name(s: str) -> bool:
            """Heuristic filter to avoid URLs/paragraphs being treated as group names."""
            if not s:
                return False
            if url_re.match(s):
                return False
            if len(s) > options.max_group_name_len:
                return False
            # Must contain at least one letter
            if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", s):
                return False
            # Avoid lines that are mostly punctuation
            letters = sum(ch.isalpha() for ch in s)
            if letters < 3:
                return False
            return True

        current_group = options.default_group
        buffer_lines: List[str] = []
        docs: List[DocumentRecord] = []

        def flush_buffer_as_docs(group_name: str) -> None:
            joined = "\n".join(buffer_lines).strip()
            if not joined:
                return
            for block in splitter(joined):
                docs.append(
                    DocumentRecord(
                        doc_id=_new_id(),
                        text=block,
                        group=group_name,
                        source=source,
                    )
                )

        lines = raw.split("\n")
        i = 0

        # --------------------------
        # HEADER MODE: INLINE
        # --------------------------
        if options.header_mode == "inline":
            for line in lines:
                stripped = line.strip()

                # Inline header: "##### Group Name"
                if options.header_pattern and stripped.startswith(options.header_pattern) and not hash_line_re.match(stripped):
                    flush_buffer_as_docs(current_group)
                    buffer_lines = []

                    group_name = stripped[len(options.header_pattern):].strip()
                    current_group = group_name if looks_like_group_name(group_name) else options.default_group
                    continue

                buffer_lines.append(line)

            flush_buffer_as_docs(current_group)
            return docs

        # --------------------------
        # HEADER MODE: BANNER
        # Pattern:
        #   ###### (hash line)
        #   Group Name
        #   ###### (hash line)
        # --------------------------
        if options.header_mode == "banner":
            while i < len(lines):
                stripped = lines[i].strip()

                # Detect banner start
                if hash_line_re.match(stripped):
                    # Candidate title is the next non-empty line
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    # Need a title line and a closing banner line
                    if j < len(lines):
                        title = lines[j].strip()

                        k = j + 1
                        while k < len(lines) and not lines[k].strip():
                            k += 1

                        # Only accept as a "real group header" if:
                        # - title looks like a group name
                        # - and next non-empty line after title is another hash banner
                        if k < len(lines) and hash_line_re.match(lines[k].strip()) and looks_like_group_name(title):
                            # Flush content collected for the previous group
                            flush_buffer_as_docs(current_group)
                            buffer_lines = []

                            current_group = title
                            i = k + 1
                            continue

                # Normal content line
                buffer_lines.append(lines[i])
                i += 1

            flush_buffer_as_docs(current_group)
            return docs

        raise ValueError("header_mode must be 'inline' or 'banner' when group_strategy=HEADER")



# ---------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------

def ingest_csv(
    file_bytes: bytes,
    options: CsvIngestOptions,
    source: str = "csv_upload"
) -> List[DocumentRecord]:
    """
    Ingest a CSV file and convert it into a list of DocumentRecord objects.

    CSV ingestion is based on explicit columns:
    - text_column (required): column containing the text to analyze
    - id_column (optional): column used as doc_id; otherwise auto-generated
    - group_column (optional): column used as group label; otherwise default_group

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded CSV.
    options:
        CsvIngestOptions describing which columns to use.
    source:
        A string tag for provenance.

    Returns
    -------
    List[DocumentRecord]
        Standardized documents.
    """
    
    
    # Decode bytes into a string, then wrap as a StringIO buffer for pandas

    text = _safe_decode(file_bytes)
    buf = io.StringIO(text)

    # Load CSV into DataFrame
    df = pd.read_csv(buf)
    
    # Validate text column presence
    # if options.text_column not in df.columns:
    #     raise ValueError(
    #         f"text_column '{options.text_column}' not found. Available columns: {list(df.columns)}"
    #     )
    
    columns_map = {c.lower(): c for c in df.columns}

    requested_text = options.text_column.lower()
    if requested_text not in columns_map:
        raise ValueError(f"text_column '{options.text_column}' not found. Available columns: {list(df.columns)}")

    text_col = columns_map[requested_text]
    
    # Validate optional columns (only use them if they exist)
    # id_col = options.id_column if options.id_column and options.id_column in df.columns else None
    # group_col = options.group_column if options.group_column and options.group_column in df.columns else None
    
    id_col = columns_map.get(options.id_column.lower()) if options.id_column else None
    group_col = columns_map.get(options.group_column.lower()) if options.group_column else None
    
    docs: List[DocumentRecord] = []

    # Iterate each row and build DocumentRecords
    for _, row in df.iterrows():
        raw_text = row.get(options.text_column, "")

        # Skip empty / NaN text
        if pd.isna(raw_text) or str(raw_text).strip() == "":
            continue

        # Use provided ID column if available, otherwise generate one
        doc_id = str(row[id_col]) if id_col and not pd.isna(row[id_col]) else _new_id()

        # Use provided group column if available, otherwise default
        group = (
            str(row[group_col]).strip()
            if group_col and not pd.isna(row[group_col]) and str(row[group_col]).strip() != ""
            else options.default_group
        )

        docs.append(
            DocumentRecord(
                doc_id=doc_id,
                text=str(raw_text),
                group=group,
                source=source,
            )
        )

    return docs


