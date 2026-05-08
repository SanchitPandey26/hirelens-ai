"""
build_index.py — Offline Pre-processing & Indexing Script
=========================================================
Admin tool that pre-processes all PDF resumes in resume_data/ and builds
persistent indexes for high-performance hybrid retrieval.

Produces three artifacts in index_store/:
  1. faiss.index  — FAISS IndexFlatIP (inner-product on L2-normed vectors = cosine sim)
  2. bm25.pkl     — Pickled BM25Okapi object for keyword retrieval
  3. metadata.pkl — Pickled list of dicts with candidate_id, resume_json, and token_doc

Usage:
  python build_index.py                          # Index all PDFs in resume_data/
  python build_index.py --resume-dir my_folder   # Index PDFs from a custom folder
  python build_index.py --skip-existing           # Only process new (un-indexed) resumes
  python build_index.py --limit 5                 # Process only the first 5 PDFs (for testing)
"""

import os
import sys
import json
import time
import pickle
import string
import argparse
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Add the project root to sys.path so we can import resume_model modules
# regardless of where this script is executed from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from resume_model.text_extract import extract_text_and_links
from resume_model.resume_api_integration import call_gemini_api
from resume_model.embedding_matching import ResumeJDMatcher

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
API_KEY = os.getenv("GEMINI_API_KEY")

# Default paths
DEFAULT_RESUME_DIR = os.path.join(PROJECT_ROOT, "resume_data")
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "index_store")


# ===========================================================================
# Tokenizer — shared with HybridResumeRetriever in embedding_matching.py
# ===========================================================================
def tokenize_text(text: str) -> list:
    """
    Lowercase, strip punctuation, and split into word tokens.
    This MUST stay in sync with HybridResumeRetriever._tokenize()
    so that the BM25 index and query use identical tokenization.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return tokens


# ===========================================================================
# Main indexing logic
# ===========================================================================
def build_indexes(resume_dir: str, index_dir: str, skip_existing: bool = False, limit: int = None):
    """
    Scan resume_dir for PDFs, parse each with Gemini, generate embeddings,
    tokenize for BM25, and persist all artifacts to index_dir.
    """
    os.makedirs(index_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover PDF files
    # ------------------------------------------------------------------
    pdf_files = sorted(
        [f for f in os.listdir(resume_dir) if f.lower().endswith(".pdf")],
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f
    )
    print(f"[INFO] Found {len(pdf_files)} PDF files in {resume_dir}")

    if limit:
        pdf_files = pdf_files[:limit]
        print(f"[INFO] --limit {limit}: processing only first {len(pdf_files)} files")

    # ------------------------------------------------------------------
    # 2. Load existing metadata (for --skip-existing)
    # ------------------------------------------------------------------
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    existing_metadata = []
    existing_ids = set()

    if skip_existing and os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            existing_metadata = pickle.load(f)
        existing_ids = {m["candidate_id"] for m in existing_metadata}
        print(f"[INFO] --skip-existing: {len(existing_ids)} resumes already indexed")

    # ------------------------------------------------------------------
    # 3. Initialize the embedding client (reuses existing ResumeJDMatcher)
    # ------------------------------------------------------------------
    matcher = ResumeJDMatcher(API_KEY)

    # ------------------------------------------------------------------
    # 4. Process each resume
    # ------------------------------------------------------------------
    new_metadata = []          # List of dicts: {candidate_id, resume_json, token_doc}
    new_embeddings = []        # List of np.array vectors
    errors = []                # Track failures

    start_time = time.time()

    for i, pdf_name in enumerate(pdf_files):
        # Skip if already indexed
        if pdf_name in existing_ids:
            print(f"  [{i+1}/{len(pdf_files)}] SKIP (already indexed): {pdf_name}")
            continue

        print(f"  [{i+1}/{len(pdf_files)}] Processing: {pdf_name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            pdf_path = os.path.join(resume_dir, pdf_name)

            # --- Step A: Extract raw text from PDF ---
            resume_text, _ = extract_text_and_links(pdf_path)

            # --- Step B: Parse structured JSON via Gemini (unchanged prompt) ---
            # filename becomes candidate_id inside the Gemini prompt
            parsed_json_str = call_gemini_api(resume_text, pdf_name, API_KEY)
            resume_json = json.loads(parsed_json_str)

            # --- Step C: Generate embedding vector ---
            embedding_text = matcher.prepare_text_for_embedding(resume_json)
            embedding_vector = matcher.generate_embedding(embedding_text)

            # Rate-limit: sleep 30s between embedding API calls to avoid 503 overload
            time.sleep(30)

            # --- Step D: Tokenize for BM25 ---
            token_doc = tokenize_text(embedding_text)

            # --- Step E: Collect results ---
            new_metadata.append({
                "candidate_id": pdf_name,
                "resume_json": resume_json,
                "token_doc": token_doc,
            })
            new_embeddings.append(np.array(embedding_vector, dtype=np.float32))

            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s) — {e}")
            errors.append((pdf_name, str(e)))
            continue

        # Progress report every 10 resumes
        if (i + 1) % 10 == 0:
            total_elapsed = time.time() - start_time
            processed = len(new_metadata)
            remaining = len(pdf_files) - (i + 1)
            if processed > 0:
                avg_per_resume = total_elapsed / processed
                eta = avg_per_resume * remaining
                print(f"  --- Progress: {processed} indexed, {remaining} remaining, "
                      f"ETA: {eta:.0f}s ({eta/60:.1f}min) ---")

    # ------------------------------------------------------------------
    # 5. Merge with existing data
    # ------------------------------------------------------------------
    all_metadata = existing_metadata + new_metadata

    # Rebuild embeddings: load existing FAISS index vectors if present
    if existing_metadata:
        # Re-generate embeddings for existing entries by reading from existing FAISS
        faiss_path = os.path.join(index_dir, "faiss.index")
        if os.path.exists(faiss_path):
            old_index = faiss.read_index(faiss_path)
            old_vectors = np.array([old_index.reconstruct(j) for j in range(old_index.ntotal)])
            all_embeddings = list(old_vectors) + new_embeddings
        else:
            # Fallback: only new embeddings (shouldn't happen with valid metadata)
            all_embeddings = new_embeddings
    else:
        all_embeddings = new_embeddings

    if not all_embeddings:
        print("[ERROR] No embeddings to index. Exiting.")
        return

    # ------------------------------------------------------------------
    # 6. Build FAISS index
    #    Using IndexFlatIP (Inner Product) on L2-normalized vectors.
    #    Inner product of L2-normed vectors = cosine similarity.
    # ------------------------------------------------------------------
    embedding_matrix = np.vstack(all_embeddings).astype(np.float32)

    # L2-normalize each vector so inner product = cosine similarity
    faiss.normalize_L2(embedding_matrix)

    dimension = embedding_matrix.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product index
    faiss_index.add(embedding_matrix)

    faiss_path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(faiss_index, faiss_path)
    print(f"\n[SAVED] FAISS index: {faiss_path} ({faiss_index.ntotal} vectors, dim={dimension})")

    # ------------------------------------------------------------------
    # 7. Build BM25 index from ALL token documents
    # ------------------------------------------------------------------
    all_token_docs = [m["token_doc"] for m in all_metadata]
    bm25 = BM25Okapi(all_token_docs)

    bm25_path = os.path.join(index_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[SAVED] BM25 index:  {bm25_path} ({len(all_token_docs)} documents)")

    # ------------------------------------------------------------------
    # 8. Save metadata (candidate_id, resume_json, token_doc for each resume)
    # ------------------------------------------------------------------
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)
    print(f"[SAVED] Metadata:    {metadata_path} ({len(all_metadata)} entries)")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Indexing Complete!")
    print(f"  New resumes processed : {len(new_metadata)}")
    print(f"  Previously indexed    : {len(existing_metadata)}")
    print(f"  Total in index        : {len(all_metadata)}")
    print(f"  Errors                : {len(errors)}")
    print(f"  Total time            : {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Index directory       : {index_dir}")
    print(f"{'='*60}")

    if errors:
        print("\n[ERRORS] The following resumes failed:")
        for fname, err in errors:
            print(f"  - {fname}: {err}")


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 indexes from PDF resumes for hybrid retrieval."
    )
    parser.add_argument(
        "--resume-dir",
        default=DEFAULT_RESUME_DIR,
        help=f"Directory containing PDF resumes (default: {DEFAULT_RESUME_DIR})"
    )
    parser.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to store index files (default: {DEFAULT_INDEX_DIR})"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip resumes that are already indexed (default: True)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force re-index ALL resumes, ignoring existing index"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PDFs (for testing)"
    )

    args = parser.parse_args()

    if not API_KEY:
        print("[FATAL] GEMINI_API_KEY not found in environment. Set it in .env file.")
        sys.exit(1)

    # If force-rebuild, disable skip-existing
    skip = args.skip_existing and not args.force_rebuild

    print(f"{'='*60}")
    print(f"  Resume Hybrid Index Builder")
    print(f"  Resume dir   : {args.resume_dir}")
    print(f"  Index dir    : {args.index_dir}")
    print(f"  Skip existing: {skip}")
    print(f"  Limit        : {args.limit or 'None (all)'}")
    print(f"{'='*60}\n")

    build_indexes(
        resume_dir=args.resume_dir,
        index_dir=args.index_dir,
        skip_existing=skip,
        limit=args.limit,
    )
