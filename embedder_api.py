# embedder_api.py
"""
Personal LLM Embedding Service

A FastAPI-based service that generates vector embeddings from text.
Supports local models via SentenceTransformers.
"""

# ----------------- STANDARD LIBRARIES -----------------
import os          # For reading environment variables
import re          # Regular expressions for text cleaning
import uuid        # To generate unique IDs for embeddings
import time        # To measure processing time
from datetime import datetime   # Timestamps for metadata
from typing import List, Optional  # Type hints
from contextlib import asynccontextmanager  # For FastAPI lifespan context

# ----------------- THIRD-PARTY LIBRARIES -----------------
from fastapi import FastAPI, HTTPException  # Web framework and exception handling
from pydantic import BaseModel, field_validator, model_validator  # Data validation for request/response models
from bs4 import BeautifulSoup  # HTML parsing and stripping
import numpy as np  # Numerical operations

# SentenceTransformer is optional; raise error if missing
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

# ----------------- CONFIGURATION -----------------
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")  # "local" / "openai" / "gemini"
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "all-mpnet-base-v2")  # Default local model
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda" (GPU)
HTML_STRIP = os.getenv("HTML_STRIP", "true").strip().lower() in ("1", "true", "yes")
 # Whether to strip HTML from text

# Global variable to hold the loaded embedding model
_model: Optional[SentenceTransformer] = None

# Precompiled regex for simple HTML tag detection (fast path)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# ----------------- FASTAPI APP WITH LIFESPAN -----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic for the FastAPI app.
    Loads the embedding model at startup based on configuration.
    """
    global _model
    if EMBEDDING_MODE == "local":
        print(f"[Embedding] Loading local model: {LOCAL_MODEL} on {DEVICE}")
        _model = SentenceTransformer(LOCAL_MODEL, device=DEVICE)
    elif EMBEDDING_MODE in ["openai", "gemini"]:
        print(f"[Embedding] {EMBEDDING_MODE.capitalize()} mode selected (implementation placeholder).")
    else:
        raise ValueError(f"Unknown EMBEDDING_MODE: {EMBEDDING_MODE}")
    yield  # FastAPI will run the app during this yield
    # Optional shutdown/cleanup code can be added here

# Initialize FastAPI app with custom lifespan
app = FastAPI(title="Personal LLM Embedding Service", version="1.0", lifespan=lifespan)

# ----------------- DATA SCHEMAS -----------------
class EmbedRequest(BaseModel):
    """
    Request schema for embedding endpoint.
    Accepts either a single text or a list of texts.
    """
    
    texts: Optional[List[str]] = None
    source: str = "user_prompt"


    # ---- Field-level cleaning : trim strings early ----
    @field_validator("texts", mode="before")
    @classmethod
    def ensure_list_and_trim(cls, v):
        # Case 1: single string → wrap into list
        if isinstance(v, str):
            return [v.strip()]
        # Case 2: list of strings → trim each
        if isinstance(v, list):
            return [s.strip() for s in v if isinstance(s, str)]
        # Invalid input
        raise ValueError("texts must be a string or list of strings")
    
    # ---- Model-level rule: enforce exactly one of text/texts ----
    @model_validator(mode="after")
    def check_texts_not_empty(self):
        """
        Ensure `texts` exists and has at least one non-empty string.
        """
        if not self.texts or all(len(s) == 0 for s in self.texts):
            raise ValueError("At least one non-empty string is required in 'texts'.")
        return self

class EmbedResponseItem(BaseModel):
    """
    Each embedding item returned from the API.
    """
    id: str
    vector: List[float]
    metadata: dict

class EmbedBatchResponse(BaseModel):
    """
    Full response returned by the embedding endpoint.
    Contains metadata, processing time, and embedding vectors.
    """
    mode: str
    model_name: str
    vector_size: int
    count: int
    processing_time_sec: float
    items: List[EmbedResponseItem]

# ----------------- HELPER FUNCTIONS -----------------


def _strip_html_if_present(s: str) -> str:
    """
    Strip HTML only if:
      1) HTML_STRIP is enabled AND
      2) The input likely contains HTML tags (regex detection)
    """
    if not HTML_STRIP:
        return s
    if not s:
        return s
    # Fast-path check: only call BeautifulSoup if tags are found
    if _HTML_TAG_RE.search(s):
        return BeautifulSoup(s, "html.parser").get_text(" ")
    return s

def _clean(s: str) -> str:
    """
    Clean text by stripping HTML and normalizing whitespace.
    """
    s = _strip_html_if_present(s)
    return _WHITESPACE_RE.sub(" ", s.strip())

def _ensure_list(req: EmbedRequest) -> List[str]:
    """
    Convert request input into a list of strings.
    """
    return  req.texts or []

def _ensure_python_list(vectors):
    """
    Ensure numpy arrays are converted to plain Python lists for JSON serialization.
    """
    if isinstance(vectors, np.ndarray):
        return vectors.tolist()
    elif isinstance(vectors, list):
        return [v.tolist() if isinstance(v, np.ndarray) else v for v in vectors]
    else:
        raise TypeError(f"Unsupported vector type: {type(vectors)}")

def _embed_local(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using a local SentenceTransformer model.
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded.")
    # Batch encode; convert_to_numpy gives np.float32 -> cast to Python list
    raw_vectors = _model.encode(texts, convert_to_numpy=True)
    vectors=_ensure_python_list(raw_vectors)

    # Debug log
    print(f"[Embedding] Generated {len(texts)} embeddings, dim={len(vectors[0])}")

    return vectors

def _embed(texts: List[str]) -> List[List[float]]:
    """
    Route to the correct embedding method based on EMBEDDING_MODE.
    Currently only supports local embeddings.
    """
    if EMBEDDING_MODE == "local":
        return _embed_local(texts)
    raise HTTPException(status_code=501, detail=f"Unknown EMBEDDING_MODE: {EMBEDDING_MODE}")

# ----------------- ROUTES -----------------
@app.get("/")
def health():
    """
    Health check endpoint.
    Returns service status, mode, model, device, and HTML stripping setting.
    """
    return {
        "status": "ok",
        "mode": EMBEDDING_MODE,
        "model": LOCAL_MODEL if EMBEDDING_MODE == "local" else EMBEDDING_MODE,
        "device": DEVICE if EMBEDDING_MODE == "local" else "cloud",
        "html_strip": HTML_STRIP,
    }

@app.post("/embed", response_model=EmbedBatchResponse)
def embed(req: EmbedRequest):
    """
    Generate embeddings for one or multiple texts.
    Returns embedding vectors with metadata and processing time.
    """
    raw_texts: List[str] = _ensure_list(req)
    cleaned_texts: List[str] = [_clean(x) for x in raw_texts if len(x.strip()) > 0]

    if not cleaned_texts:
        raise HTTPException(status_code=400, detail="All inputs are empty after cleaning.")

    # Generate embeddings and measure processing time
    t0 = time.time()
    vectors = _embed(cleaned_texts)
    dt = time.time() - t0

    if not vectors or not vectors[0]:
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

    # Build response items
    items = [
        EmbedResponseItem(
            id=str(uuid.uuid4()),
            vector=v,
            metadata={
                "text": c,
                "timestamp": datetime.utcnow().isoformat(),
                "source": req.source,
                "embedding_mode": EMBEDDING_MODE,
            },
        )
        for c, v in zip(cleaned_texts, vectors)
    ]

    # Return full response
    return EmbedBatchResponse(
        mode=EMBEDDING_MODE,
        model_name=(LOCAL_MODEL if EMBEDDING_MODE == "local" else EMBEDDING_MODE),
        vector_size=len(vectors[0]),
        count=len(items),
        processing_time_sec=round(dt, 4),
        items=items,
    )
