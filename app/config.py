from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
INDEX_DIR = STORAGE_DIR / "index"
INDEX_PATH = INDEX_DIR / "mindvault.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

for directory in (STORAGE_DIR, UPLOAD_DIR, INDEX_DIR):
    directory.mkdir(parents=True, exist_ok=True)
