import os
import re
import fitz
from pathlib import Path
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
    print(f"Opening PDF: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    # Representative pages across key events in all 7 books
    pages_to_index = [
        15, 50, 100, 120, 270,        # Book 1 (Intro, Diagon Alley, Sorting, Mirror of Erised)
        301, 302, 314, 400, 520,      # Book 2 (Flying Ford Anglia, Dobby, Basilisk)
        580, 650, 750, 800, 920,      # Book 3 (Dementors, Marauder's Map, Patronus, Sirius)
        960, 1100, 1250, 1400, 1550,  # Book 4 (Triwizard Tournament, Moody, Graveyard)
        1600, 1800, 2000, 2200, 2390, # Book 5 (Order of Phoenix, DA, Ministry, Prophecy)
        2420, 2600, 2800, 2950, 2954, # Book 6 (Horcruxes, Slughorn, Half-Blood Prince)
        2990, 3100, 3200, 3400, 3600  # Book 7 (Seven Potters, Deathly Hallows, Final Battle)
    ]

    chunks_data = []
    for p in pages_to_index:
        if p <= total_pages:
            text = doc[p - 1].get_text()
            text = re.sub(r"\s+", " ", text).strip()
            book = get_book_name(p)
            start = 0
            idx = 1
            while start < len(text):
                chunk = text[start : start + 700].strip()
                if len(chunk) > 60:
                    chunks_data.append({
                        "id": f"hp_p{p}_c{idx}",
                        "book": book,
                        "page": p,
                        "text": chunk
                    })
                    idx += 1
                start += 600
    doc.close()
    print(f"Extracted {len(chunks_data)} chunks from {len(pages_to_index)} reference pages.")

    print("Loading intfloat/multilingual-e5-large...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    print("Encoding passages with E5 prefix...")
    passages = [f"passage: {c['text']}" for c in chunks_data]
    embeddings = model.encode(passages, normalize_embeddings=True, show_progress_bar=True).tolist()

    destinations = [
        BASE_DIR / "backend" / "data" / "vector_store",
        BASE_DIR / "data" / "vector_store"
    ]

    for dest in destinations:
        dest.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(dest))
        col = client.get_or_create_collection("harry_potter_books", metadata={"hnsw:space": "cosine"})
        col.upsert(
            ids=[c["id"] for c in chunks_data],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks_data],
            metadatas=[{"book_name": c["book"], "page_number": c["page"], "chunk_id": c["id"]} for c in chunks_data]
        )
        print(f"Successfully populated {dest} (Total records: {col.count()})")

if __name__ == "__main__":
    main()
