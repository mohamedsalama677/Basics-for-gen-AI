"""One-shot: chunk the docs, embed them locally, save a FAISS index.

Run once (or whenever docs/ changes):
    python src/ingest.py
"""

from collections import defaultdict

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)


def main() -> None:
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # tag each chunk with a per-source running index so citations are stable
    per_source_index: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source_name = _short_source(chunk.metadata.get("source", "unknown"))
        chunk.metadata["source"] = source_name
        chunk.metadata["chunk_id"] = per_source_index[source_name]
        per_source_index[source_name] += 1

    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads ~90MB)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    store = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    print(f"Saved FAISS index to {VECTORSTORE_DIR}")


def _short_source(path: str) -> str:
    """Reduce a full file path to just the filename for cleaner citations."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


if __name__ == "__main__":
    main()
