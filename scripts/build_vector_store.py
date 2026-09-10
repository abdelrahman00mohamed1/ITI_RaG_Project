import os
import re
import sys
import time
from pathlib import Path
import fitz
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# Locate project root
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "raw" / "harrypotter.pdf"
if not PDF_PATH.exists():
    PDF_PATH = BASE_DIR / "harrypotter.pdf"

BOOK_RANGES = [
    ("Harry Potter and the Sorcerer's Stone", 12, 274),
    ("Harry Potter and the Chamber of Secrets", 282, 565),
    ("Harry Potter and the Prisoner of Azkaban", 573, 939),
    ("Harry Potter and the Goblet of Fire", 949, 1560),
    ("Harry Potter and the Order of the Phoenix", 1570, 2406),
    ("Harry Potter and the Half-Blood Prince", 2409, 2964),
    ("Harry Potter and the Deathly Hallows", 2974, 3622),
]


def get_book_name(page_num: int) -> str:
    for name, start, end in BOOK_RANGES:
        if start <= page_num <= end:
            return name
    return "Harry Potter Anthology"


def main():
    print(f"📖 Opening PDF: {PDF_PATH}")
    if not PDF_PATH.exists():
        print(f"❌ Error: PDF not found at {PDF_PATH}")
        sys.exit(1)

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"📄 Total pages in document: {total_pages}")

    # Allow custom page limit or step via environment variable if needed
    max_pages = int(os.getenv("MAX_PAGES", total_pages))
    max_pages = min(max_pages, total_pages)

    print(f"⚡ Extracting and chunking text across pages 1 to {max_pages}...")
    chunks_data = []

    for p in range(1, max_pages + 1):
        page_text = doc[p - 1].get_text()
        page_text = re.sub(r"\s+", " ", page_text).strip()

        # Skip blank, cover, or purely numerical/header pages
        if len(page_text) < 60:
            continue

        book_title = get_book_name(p)
        start = 0
        idx = 1
        chunk_size = 700
        overlap = 100

        while start < len(page_text):
            chunk = page_text[start : start + chunk_size].strip()
            if len(chunk) > 60:
                chunks_data.append({
                    "id": f"hp_p{p}_c{idx}",
                    "book": book_title,
                    "page": p,
                    "text": chunk,
                })
                idx += 1
            start += (chunk_size - overlap)

    doc.close()
    total_chunks = len(chunks_data)
    print(f"✅ Extracted {total_chunks} text chunks from {max_pages} pages.")

    # Detect compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64 if device == "cuda" else 16
    print(f"🚀 Using compute device: {device.upper()} (Batch size: {batch_size})")

    print("⏳ Loading embedding model 'intfloat/multilingual-e5-large'...")
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

    destinations = [
        BASE_DIR / "backend" / "data" / "vector_store",
        BASE_DIR / "data" / "vector_store",
    ]

    # Initialize Chroma clients
    chroma_collections = []
    for dest in destinations:
        dest.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(dest))
        col = client.get_or_create_collection("harry_potter_books", metadata={"hnsw:space": "cosine"})
        chroma_collections.append((dest, col))

    # Process and upsert in progressive batches of 256 chunks
    chunk_batch_size = 256
    print(f"📦 Encoding and indexing {total_chunks} chunks in batches of {chunk_batch_size}...")

    start_time = time.time()
    for b_idx in range(0, total_chunks, chunk_batch_size):
        batch = chunks_data[b_idx : b_idx + chunk_batch_size]
        passages = [f"passage: {c['text']}" for c in batch]

        # Generate embeddings
        embeddings = model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {"book_name": c["book"], "page_number": c["page"], "chunk_id": c["id"]}
            for c in batch
        ]

        # Upsert into all Chroma destinations
        for _, col in chroma_collections:
            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        processed = min(b_idx + chunk_batch_size, total_chunks)
        pct = (processed / total_chunks) * 100
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 1
        eta = (total_chunks - processed) / rate
        print(f"  Progress: {processed}/{total_chunks} ({pct:.1f}%) | ETA: {eta:.0f}s")

    for dest, col in chroma_collections:
        print(f"🎉 Successfully populated {dest} (Total records in vector store: {col.count()})")


if __name__ == "__main__":
    main()

