import json
from typing import Optional
from fastapi import Form

import csv
import io

"""
## API routes ##

This file defines the HTTP endpoints that the frontend (or Swagger UI / curl) will call. It:
Receives uploaded files (UploadFile)
Receives ingestion options (TXT options or CSV options)
Calls the ingestion service (ingest_txt, ingest_csv)

Returns a clean, validated IngestResponse with:
n_docs
groups
preview (first 10 docs)

It also converts errors into proper HTTP responses (status 400 with a message).
"""


"""
API Routes for Ingestion

Purpose:
- Provide HTTP endpoints for uploading datasets (TXT/CSV).
- Validate options using Pydantic schemas.
- Call ingestion service functions that convert uploads into DocumentRecord objects.
- Return a consistent response format that the frontend can consume.

Routes in this file:
- POST /api/ingest/txt  -> ingest a TXT file (thesis-style grouped or generic)
- POST /api/ingest/csv  -> ingest a CSV file (requires choosing the text column)
"""


from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.ingestion import(
    TxtIngestOptions,
    CsvIngestOptions,
    IngestResponse,
)

from app.services.ingestion import ingest_txt, ingest_csv

from app.schemas.ingestion import CsvIngestOptions
from app.schemas.topics import TopicModelResponse


# Create a router to group related endpoints together
router = APIRouter()

@router.post("/ingest/txt", response_model=IngestResponse)
async def ingest_txt_endpoint(
    #file: UploadFile = File(...),
    #options: TxtIngestOptions = None,
    file: UploadFile = File(...),
    options_json: Optional[str] = Form(None),  # <-- options come as a JSON string in multipart
):
    """
    Ingest a TXT upload.

    Supports:
    - Thesis-style grouping (e.g., lines starting with "##### University Name")
    - Generic TXT without groups

    Parameters
    ----------
    file:
        Uploaded TXT file.
    options:
        Optional TxtIngestOptions controlling:
        - grouping strategy (header / none)
        - header pattern (e.g., "#####")
        - split mode (blank_lines / lines)
        - default group label

    Returns
    -------
    IngestResponse:
        - n_docs: number of extracted documents
        - groups: unique detected group names
        - preview: first 10 DocumentRecord objects
    """
    
    try:
        # Read entire uploaded file into memory as bytes
        file_bytes = await file.read()
        
        # If no options were provided, use defaults:
        # - group_strategy = NONE
        # - split_mode = blank_lines
        
        #opts = options or TxtIngestOptions()
        
        # If options_json is not provided, use defaults
        if options_json:
            opts = TxtIngestOptions(**json.loads(options_json))
        else:
            opts = TxtIngestOptions()
        
        # Convert upload into canonical DocumentRecord list
        docs = ingest_txt(file_bytes, opts, source="txt_upload")

        # Collect unique group names for UI (sorted for stability)
        groups = sorted({d.group for d in docs})
        
        # Return summary + first few docs for preview/debugging
        return IngestResponse(
            n_docs=len(docs),
            groups=groups,
            preview=docs[:10],
        )
    except Exception as e:
        # Convert any ingestion error into HTTP 400 for the client
        raise HTTPException(status_code=400, detail=str(e))
    
    
    

@router.post("/ingest/csv", response_model=IngestResponse)
async def ingest_csv_endpoint(
    file: UploadFile = File(...),
    options: CsvIngestOptions = None,
):
    
    """
    Ingest a CSV upload.

    CSV ingestion requires at minimum:
    - text_column: name of column containing the text

    Optional:
    - id_column: unique ID column
    - group_column: grouping/category column

    Parameters
    ----------
    file:
        Uploaded CSV file.
    options:
        CsvIngestOptions (REQUIRED):
        - text_column must be provided

    Returns
    -------
    IngestResponse with n_docs, groups, preview
    """
    
    try:
        file_bytes = await file.read()

        # CSV ingestion needs at least text_column.
        # So we force the caller to provide options.
        if options is None:
            raise ValueError(
                "CSV ingestion requires options: text_column (and optional id_column/group_column)."
            )
        
        docs = ingest_csv(file_bytes, options, source="csv_upload")
        groups = sorted({d.group for d in docs})

        return IngestResponse(
            n_docs=len(docs),
            groups=groups,
            preview=docs[:10],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    

#---------
# Calling the Preprocessing
#---------
from app.schemas.analyze import PreprocessRequest, PreprocessResponse

@router.post("/analyze/preprocess", response_model=PreprocessResponse)
async def preprocess_endpoint(req: PreprocessRequest):
    
    """
    Quick endpoint to validate preprocessing logic from Swagger UI.
    Later, our /analyze endpoint will accept ingested DocumentRecords and run:
    ingestion -> preprocessing -> topic model -> results
    """
    
    try:
        from ml.preprocessing import PreprocessConfig, preprocess_many  # import from repo root

        configuration = PreprocessConfig(
            min_token_len=req.min_token_len,
            remove_stopwords=req.remove_stopwords,
            lemmatize=req.lemmatize,
            remove_urls=req.remove_urls,
            keep_only_letters=req.keep_only_letters,
            lowercase=req.lowercase,
            extra_stopwords=req.extra_stopwords or [],
        )
        
        cleaned = preprocess_many(req.texts, configuration)
        
        empty_count = sum(1 for x in cleaned if not x.strip())
        
        return PreprocessResponse(
            n_docs=len(req.texts),
            preview_raw=req.texts[:5],
            preview_clean=cleaned[:5],
            empty_after_cleaning=empty_count,
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    




#---------
# For Form Upload - Txt Upload === TXT upload → ingest → preprocess
#---------
import json
from app.schemas.analyze_upload import AnalyzeUploadResponse

@router.post("/analyze/from-upload/txt", response_model=AnalyzeUploadResponse)
async def analyze_from_txt_upload(
    file: UploadFile = File(...),
    ingest_options_json: Optional[str] = Form(None),
    preprocess_options_json: Optional[str] = Form(None),
):
    """
    Upload TXT -> ingest -> preprocess.

    ingest_options_json: TxtIngestOptions as JSON string
    preprocess_options_json: PreprocessRequest-like fields as JSON string (optional)
    """
    
    try:
        # 1) Read file
        file_bytes = await file.read()
        
        # 2) Parse ingest options (multipart needs JSON string)
        if ingest_options_json:
            ingest_options = TxtIngestOptions(**json.loads(ingest_options_json))
        else:
            ingest_options = TxtIngestOptions()
            
        # 3) Ingest into canonical docs
        docs = ingest_txt(file_bytes, ingest_options, source="txt_upload")
        
        # 4) Parse preprocess options (optional)
        preprocess_payload = json.loads(preprocess_options_json) if preprocess_options_json else {}
        extra_stopwords = preprocess_payload.get("extra_stopwords") or []
        
        from ml.preprocessing import PreprocessConfig, preprocess_many  # requires PYTHONPATH set to repo root

        configuration = PreprocessConfig(
            min_token_len=int(preprocess_payload.get("min_token_len", 4)),
            remove_stopwords=bool(preprocess_payload.get("remove_stopwords", True)),
            lemmatize=bool(preprocess_payload.get("lemmatize", True)),
            remove_urls=bool(preprocess_payload.get("remove_urls", True)),
            keep_only_letters=bool(preprocess_payload.get("keep_only_letters", True)),
            lowercase=bool(preprocess_payload.get("lowercase", True)),
            extra_stopwords=extra_stopwords,
        )
        
        # 5) Preprocess all document texts
        raw_texts = [d.text for d in docs]
        cleaned_texts = preprocess_many(raw_texts, configuration)
        
        empty_count = sum(1 for t in cleaned_texts if not t.strip())
        token_counts = [len(t.split()) for t in cleaned_texts if t.strip()]
        avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
        
        # groups summary
        groups = sorted({d.group for d in docs})
        groups_preview = groups[:25]
        
        # preview records
        preview_raw = docs[:10]
        preview_clean = [
            {"doc_id": d.doc_id, "group": d.group, "cleaned_text": cleaned_texts[i]}
            for i, d in enumerate(preview_raw)
        ]
        
        return AnalyzeUploadResponse(
            n_docs=len(docs),
            n_groups=len(groups),
            groups_preview=groups_preview,
            empty_after_cleaning=empty_count,
            avg_cleaned_tokens=avg_tokens,
            preview_docs_raw=preview_raw,
            preview_docs_clean=preview_clean,
        )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    

#---------
# For Form Upload - CSV Upload === CSV upload → ingest → preprocess
#---------

@router.post("/analyze/from-upload/csv", response_model=AnalyzeUploadResponse)
async def analyze_from_csv_upload(
    file: UploadFile = File(...),
    ingest_options_json: str = Form(...),
    preprocess_options_json: Optional[str] = Form(None),
):
    """
    Upload CSV -> ingest -> preprocess.

    ingest_options_json: CsvIngestOptions as JSON string (required, must include text_column)
    preprocess_options_json: optional preprocessing config JSON string
    """
    
    try:
        file_bytes = await file.read()

        ingest_options = CsvIngestOptions(**json.loads(ingest_options_json))
        docs = ingest_csv(file_bytes, ingest_options, source="csv_upload")

        preprocess_payload = json.loads(preprocess_options_json) if preprocess_options_json else {}
        extra_stopwords = preprocess_payload.get("extra_stopwords") or []

        from ml.preprocessing import PreprocessConfig, preprocess_many

        configuration = PreprocessConfig(
            min_token_len=int(preprocess_payload.get("min_token_len", 4)),
            remove_stopwords=bool(preprocess_payload.get("remove_stopwords", True)),
            lemmatize=bool(preprocess_payload.get("lemmatize", True)),
            remove_urls=bool(preprocess_payload.get("remove_urls", True)),
            keep_only_letters=bool(preprocess_payload.get("keep_only_letters", True)),
            lowercase=bool(preprocess_payload.get("lowercase", True)),
            extra_stopwords=extra_stopwords,
        )
        
        raw_texts = [d.text for d in docs]
        cleaned_texts = preprocess_many(raw_texts, configuration)

        empty_count = sum(1 for t in cleaned_texts if not t.strip())
        token_counts = [len(t.split()) for t in cleaned_texts if t.strip()]
        avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0

        groups = sorted({d.group for d in docs})
        groups_preview = groups[:25]
        
        preview_raw = docs[:10]
        preview_clean = [
            {"doc_id": d.doc_id, "group": d.group, "cleaned_text": cleaned_texts[i]}
            for i, d in enumerate(preview_raw)
        ]
        
        return AnalyzeUploadResponse(
            n_docs=len(docs),
            n_groups=len(groups),
            groups_preview=groups_preview,
            empty_after_cleaning=empty_count,
            avg_cleaned_tokens=avg_tokens,
            preview_docs_raw=preview_raw,
            preview_docs_clean=preview_clean,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
    
#-----------
# TXT + CSV → ingest → preprocess → topics ===== Topic Endpoints
#-----------
from app.schemas.topics import TopicModelResponse


# Helper function for Topics 
def _topic_counts(topic_ids: list[int]) -> dict[int, int]:
    counts = {}
    for t in topic_ids:
        counts[t] = counts.get(t, 0) + 1
    return counts



# Endpoints

## TXT Topics
@router.post("/analyze/topics/from-upload/txt", response_model=TopicModelResponse)
async def topics_from_txt_upload(
    file: UploadFile = File(...),
    ingest_options_json: Optional[str] = Form(None),
    preprocess_options_json: Optional[str] = Form(None),
    topic_options_json: Optional[str] = Form(None),
):
    """
    Upload TXT -> ingest -> preprocess -> topic modeling (NMF + TF-IDF)

    ingest_options_json (required):
    {"group_strategy":"header","header_mode":"banner","split_mode":"blank_lines","default_group":"unknown"}

    preprocess_options_json (optional):
      

    topic_options_json (optional):
    {"n_topics":3,"n_top_words":12,"max_features":5000,"min_df":2,"max_df":0.95}

    """
    try:
        file_bytes = await file.read()

        ingest_options = TxtIngestOptions(**json.loads(ingest_options_json)) if ingest_options_json else TxtIngestOptions()
        docs = ingest_txt(file_bytes, ingest_options, source="txt_upload")

        preprocess_payload = json.loads(preprocess_options_json) if preprocess_options_json else {}
        
        ## Import
        from ml.preprocessing import PreprocessConfig, preprocess_many

        configuration = PreprocessConfig(
            min_token_len=int(preprocess_payload.get("min_token_len", 4)),
            remove_stopwords=bool(preprocess_payload.get("remove_stopwords", True)),
            lemmatize=bool(preprocess_payload.get("lemmatize", True)),
            remove_urls=bool(preprocess_payload.get("remove_urls", True)),
            keep_only_letters=bool(preprocess_payload.get("keep_only_letters", True)),
            lowercase=bool(preprocess_payload.get("lowercase", True)),
            extra_stopwords=preprocess_payload.get("extra_stopwords") or [],
        )

        cleaned_texts = preprocess_many([d.text for d in docs], configuration)
        empty_count = sum(1 for t in cleaned_texts if not t.strip())

        topic_payload = json.loads(topic_options_json) if topic_options_json else {}
        
        ## Import
        from ml.topic_modeling import TopicModelConfig, fit_nmf_topics, dominant_topics

        topic_config = TopicModelConfig(
            n_topics=int(topic_payload.get("n_topics", 10)),
            n_top_words=int(topic_payload.get("n_top_words", 12)),
            max_features=int(topic_payload.get("max_features", 5000)),
            min_df=int(topic_payload.get("min_df", 2)),
            max_df=float(topic_payload.get("max_df", 0.95)),
        )

        result = fit_nmf_topics(cleaned_texts, topic_config)

        topic_words = result["topic_words"]
        W = result["doc_topic_matrix"]
        kept_indices = result["kept_doc_indices"]

        dom = dominant_topics(W)
        # dom is aligned with kept_indices, not original docs

        # Build topic list
        topics = [{"topic_id": i, "top_terms": topic_words[i]} for i in range(topic_config.n_topics)]

        # Topic distribution
        counts = _topic_counts(dom)

        # Build preview for first 10 kept docs
        doc_preview = []
        for j in range(min(10, len(kept_indices))):
            original_idx = kept_indices[j]
            d = docs[original_idx]
            doc_preview.append({
                "doc_id": d.doc_id,
                "group": d.group,
                "dominant_topic": dom[j],
                "cleaned_text_preview": cleaned_texts[original_idx][:250],
            })
            
        
        ##--------- this part for the storing in DB
        run_id = None
        if store_run:
            from app.services.run_store import create_run, save_topics, save_doc_assignments

            run_id = create_run(
                source_type="csv",
                n_docs=len(docs),
                n_topics=topic_config.n_topics,
                ingest_options_json=ingest_options_json,
                preprocess_options_json=preprocess_options_json,
                topic_options_json=topic_options_json,
            )
            
            # Store topics
            save_topics(run_id, topics)
            
            # Store doc assignments for ALL kept docs (not just preview)
            assignments = []
            for j, original_idx in enumerate(kept_doc_indices):
                d = docs[original_idx]
                assignments.append({
                    "doc_id": d.doc_id,
                    "group": d.group,
                    "dominant_topic": dominant[j],
                    "cleaned_text": cleaned_texts[original_idx],
                })

            save_doc_assignments(run_id, assignments)
            
        ##--------- End of part for the storing in DB
        

        return TopicModelResponse(
            run_id=run_id,
            n_docs=len(docs),
            n_topics=topic_config.n_topics,
            topics=topics,
            doc_preview=doc_preview,
            topic_counts=counts,
            empty_after_cleaning=empty_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



## CSV Topics Endpoint
@router.post("/analyze/topics/from-upload/csv", response_model=TopicModelResponse)
async def topics_from_csv_upload(
    file: UploadFile = File(...),
    ingest_options_json: str = Form(...),                # REQUIRED (must include text_column)
    preprocess_options_json: Optional[str] = Form(None), # OPTIONAL
    topic_options_json: Optional[str] = Form(None),      # OPTIONAL
    
    store_run: bool = Form(False)
):
    """
    Upload CSV -> ingest -> preprocess -> topic modeling (NMF + TF-IDF)

    ingest_options_json (required):
      {"text_column":"Text","id_column":"Id","group_column":"Score","default_group":"unknown"}

    preprocess_options_json (optional):
      {"min_token_len":4,"remove_stopwords":true,"lemmatize":true,"remove_urls":true,
       "keep_only_letters":true,"lowercase":true,"extra_stopwords":["counseling","service"]}

    topic_options_json (optional):
      {"n_topics":10,"n_top_words":12,"max_features":5000,"min_df":2,"max_df":0.95}
    """
    
    try:
        # 1) Read file bytes
        file_bytes = await file.read()

        # 2) Parse ingestion options (required for CSV)
        ingest_options = CsvIngestOptions(**json.loads(ingest_options_json))

        # 3) Ingest documents (canonical schema: doc_id, group, text)
        docs = ingest_csv(file_bytes, ingest_options, source="csv_upload")

        # 4) Preprocess config (optional override)
        preprocess_payload = json.loads(preprocess_options_json) if preprocess_options_json else {}

        ##Import
        from ml.preprocessing import PreprocessConfig, preprocess_many

        configuration = PreprocessConfig(
            min_token_len=int(preprocess_payload.get("min_token_len", 4)),
            remove_stopwords=bool(preprocess_payload.get("remove_stopwords", True)),
            lemmatize=bool(preprocess_payload.get("lemmatize", True)),
            remove_urls=bool(preprocess_payload.get("remove_urls", True)),
            keep_only_letters=bool(preprocess_payload.get("keep_only_letters", True)),
            lowercase=bool(preprocess_payload.get("lowercase", True)),
            extra_stopwords=preprocess_payload.get("extra_stopwords") or [],
        )
        
        raw_texts = [d.text for d in docs]
        cleaned_texts = preprocess_many(raw_texts, configuration)

        empty_count = sum(1 for t in cleaned_texts if not t.strip())

        # 5) Topic modeling config (optional override)
        topic_payload = json.loads(topic_options_json) if topic_options_json else {}

        ##Import
        from ml.topic_modeling import TopicModelConfig, fit_nmf_topics, dominant_topics
        
        topic_config = TopicModelConfig(
            n_topics=int(topic_payload.get("n_topics", 10)),
            n_top_words=int(topic_payload.get("n_top_words", 12)),
            max_features=int(topic_payload.get("max_features", 5000)),
            min_df=int(topic_payload.get("min_df", 2)),
            max_df=float(topic_payload.get("max_df", 0.95)),
        )
        
         # 6) Fit topics (only on non-empty cleaned docs)
        result = fit_nmf_topics(cleaned_texts, topic_config)

        topic_words = result["topic_words"]
        doc_topic_matrix = result["doc_topic_matrix"]
        kept_doc_indices = result["kept_doc_indices"]

        dominant = dominant_topics(doc_topic_matrix)  # aligned to kept_doc_indices

        # 7) Prepare response payloads
        topics = [{"topic_id": i, "top_terms": topic_words[i]} for i in range(topic_config.n_topics)]
        counts = _topic_counts(dominant)
        
         # Preview first 10 kept docs (not original docs that became empty)
        doc_preview = []
        for j in range(min(10, len(kept_doc_indices))):
            original_idx = kept_doc_indices[j]
            d = docs[original_idx]
            doc_preview.append({
                "doc_id": d.doc_id,
                "group": d.group,
                "dominant_topic": dominant[j],
                "cleaned_text_preview": cleaned_texts[original_idx][:250],
            })
        
        ##--------- this part for the storing in DB
        run_id = None
        if store_run:
            from app.services.run_store import create_run, save_topics, save_doc_assignments

            run_id = create_run(
                source_type="csv",
                n_docs=len(docs),
                n_topics=topic_config.n_topics,
                ingest_options_json=ingest_options_json,
                preprocess_options_json=preprocess_options_json,
                topic_options_json=topic_options_json,
            )
            
            # Store topics
            save_topics(run_id, topics)
            
            # Store doc assignments for ALL kept docs (not just preview)
            assignments = []
            for j, original_idx in enumerate(kept_doc_indices):
                d = docs[original_idx]
                assignments.append({
                    "doc_id": d.doc_id,
                    "group": d.group,
                    "dominant_topic": dominant[j],
                    "cleaned_text": cleaned_texts[original_idx],
                })

            save_doc_assignments(run_id, assignments)
            
        ##--------- End of part for the storing in DB
            
            return TopicModelResponse(
            run_id=run_id,
            n_docs=len(docs),
            n_topics=topic_config.n_topics,
            topics=topics,
            doc_preview=doc_preview,
            topic_counts=counts,
            empty_after_cleaning=empty_count,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

#------------
# Add export endpoint (download CSV)
#------------
from fastapi.responses import StreamingResponse
from app.db import get_connection

@router.get("/runs/{run_id}/export")
def export_run_csv(run_id: str):
    """
    Download doc-topic assignments as CSV for a stored run_id.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT doc_id, doc_group, dominant_topic, cleaned_text
        FROM run_docs
        WHERE run_id = ?
        ORDER BY doc_id
        """,
        (run_id,)
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Run not found or no documents stored for this run_id.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    # Header
    writer.writerow(["doc_id", "group", "dominant_topic", "cleaned_text"])

    # Data
    for r in rows:
        writer.writerow([r["doc_id"], r["doc_group"], r["dominant_topic"], r["cleaned_text"]])

    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="run_{run_id}_assignments.csv"'}
    )




#------------
# Run dashboard Endpoint
#------------
import json
from app.db import get_connection
from app.schemas.runs import RunsListResponse, RunSummary, RunDetailResponse

# GET /api/runs
@router.get("/runs", response_model=RunsListResponse)
def list_runs(limit: int = 20, offset: int = 0):
    """
    List stored runs with offset-limit pagination.

    - limit: how many runs to return
    - offset: how many runs to skip (for paging)
    """
    
    # Basic guardrails
    limit = max(1, min(limit, 100))
    offset = max(0, offset)


    conn = get_connection()
    cur = conn.cursor()
    
    # total count
    cur.execute("SELECT COUNT(*) AS cnt FROM runs")
    total = int(cur.fetchone()["cnt"])
    
    # page
    cur.execute(
        """
        SELECT run_id, created_at, source_type, n_docs, n_topics
        FROM runs
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset)
    )
    rows = cur.fetchall()
    conn.close()
    
    runs = [
        RunSummary(
            run_id=r["run_id"],
            created_at=r["created_at"],
            source_type=r["source_type"],
            n_docs=r["n_docs"],
            n_topics=r["n_topics"],
        )
        for r in rows
    ]
    
    #return RunsListResponse(runs=runs)
    return RunsListResponse(total=total, offset=offset, limit=limit, runs=runs)




# Compare two runs Endpoint
from app.schemas.runs import RunCompareResponse

@router.get("/runs/compare", response_model=RunCompareResponse)
def compare_runs(run_id_a: str, run_id_b: str, top_k_overlap_terms: int = 8):
    """
    Compare two runs by topic top-terms overlap.
    This is a simple but useful baseline comparison for topic stability.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    def fetch_run(run_id: str):
        cur.execute(
            "SELECT run_id, created_at, source_type, n_docs, n_topics FROM runs WHERE run_id = ?",
            (run_id,)
        )
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return dict(run)

    run_a = fetch_run(run_id_a)
    run_b = fetch_run(run_id_b)
    
    # topics
    cur.execute("SELECT topic_id, top_terms_json FROM run_topics WHERE run_id = ? ORDER BY topic_id", (run_id_a,))
    topics_a = {int(r["topic_id"]): set(json.loads(r["top_terms_json"])) for r in cur.fetchall()}

    cur.execute("SELECT topic_id, top_terms_json FROM run_topics WHERE run_id = ? ORDER BY topic_id", (run_id_b,))
    topics_b = {int(r["topic_id"]): set(json.loads(r["top_terms_json"])) for r in cur.fetchall()}
    
    conn.close()
    
    overlaps = []
    for topic_a, terms_a in topics_a.items():
        for topic_b, terms_b in topics_b.items():
            common = list(terms_a.intersection(terms_b))
            if common:
                common_sorted = sorted(common)[:max(1, top_k_overlap_terms)]
                overlaps.append({
                    "topic_a": topic_a,
                    "topic_b": topic_b,
                    "overlap_count": len(common),
                    "overlap_terms": common_sorted,
                })
                
                
    # sort best matches first
    overlaps.sort(key=lambda x: x["overlap_count"], reverse=True)

    return RunCompareResponse(
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        meta={"run_a": run_a, "run_b": run_b},
        topic_overlap=overlaps[:50],
    )

    




# GET /api/runs/{run_id}
@router.get("/runs/{run_id}", response_model=RunDetailResponse)
def get_run_details(run_id: str, doc_preview_limit: int = 10):
    """
    Get details for a run:
    - metadata/options
    - topics
    - preview of doc-topic assignments
    - topic distribution counts
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # 1) Fetch run metadata
    cur.execute(
        """
        SELECT run_id, created_at, source_type, n_docs, n_topics,
               ingest_options_json, preprocess_options_json, topic_options_json
        FROM runs
        WHERE run_id = ?
        """,
        (run_id,)
    )
    run = cur.fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found.")
    
    # 2) Fetch topics for this run
    cur.execute(
        """
        SELECT topic_id, top_terms_json
        FROM run_topics
        WHERE run_id = ?
        ORDER BY topic_id ASC
        """,
        (run_id,)
    )
    topic_rows = cur.fetchall()
    
    #topics = []
    # for tr in topic_rows:
    #     topics.append({
    #         "topic_id": tr["topic_id"],
    #         #"top_terms": json.loads(tr["top_terms_json"]),
    #         top_terms": json.loads(tr["top_terms_json"]) if tr["top_terms_json"] else []
    #     })
    topics = []
    for tr in topic_rows:
        raw_terms = tr["top_terms_json"]
        topics.append({
            "topic_id": int(tr["topic_id"]),
            "top_terms": json.loads(raw_terms) if raw_terms else [],
        })
        
        
    # 3) Fetch doc assignments preview
    cur.execute(
        """
        SELECT doc_id, doc_group, dominant_topic, cleaned_text
        FROM run_docs
        WHERE run_id = ?
        LIMIT ?
        """,
        (run_id, doc_preview_limit)
    )
    doc_rows = cur.fetchall()
    
    doc_preview = []
    topic_counts: dict[int, int] = {}
    
    for dr in doc_rows:
        dominant_topic = int(dr["dominant_topic"])
        topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1

        cleaned_text = dr["cleaned_text"] or ""
        doc_preview.append({
            "doc_id": dr["doc_id"],
            "group": dr["doc_group"],
            "dominant_topic": dominant_topic,
            "cleaned_text_preview": cleaned_text[:250],
        })
        
    # NOTE:
    # The topic_counts above are computed only from the preview subset.
    # If we want full counts, do a full GROUP BY query (we can add it below).
    #
    # Let's compute full counts properly:
    
    cur.execute(
        """
        SELECT dominant_topic, COUNT(*) as cnt
        FROM run_docs
        WHERE run_id = ?
        GROUP BY dominant_topic
        ORDER BY dominant_topic
        """,
        (run_id,)
    )
    count_rows = cur.fetchall()
    topic_counts = {int(r["dominant_topic"]): int(r["cnt"]) for r in count_rows}
    
    conn.close()

    return RunDetailResponse(
        run_id=run["run_id"],
        created_at=run["created_at"],
        source_type=run["source_type"],
        n_docs=run["n_docs"],
        n_topics=run["n_topics"],
        ingest_options_json=run["ingest_options_json"],
        preprocess_options_json=run["preprocess_options_json"],
        topic_options_json=run["topic_options_json"],
        topics=topics,
        doc_preview=doc_preview,
        topic_counts=topic_counts,
    )
    
    
    
# Delete run Endpoint
@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    """
    Delete a run and all associated stored results (topics + doc assignments).
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # check exists
    cur.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Run not found.")
    
     # delete children first (FK-safe even if FKs not enforced)
    cur.execute("DELETE FROM run_docs WHERE run_id = ?", (run_id,))
    cur.execute("DELETE FROM run_topics WHERE run_id = ?", (run_id,))
    cur.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    conn.commit()
    conn.close()

    return {"status": "deleted", "run_id": run_id}










