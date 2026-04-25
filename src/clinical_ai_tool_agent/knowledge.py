"""Local FAISS knowledge base over the clinical corpus.

Embeds the markdown corpus with a small Sentence-Transformers model
(`all-MiniLM-L6-v2`), then exposes a thin retriever. All heavy imports
are deferred to `build_retriever()` so unit tests on parser/calculators
don't pull LangChain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


CORPUS_DIR = Path(__file__).parent / "corpus"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[str]:
    """Read all markdown files in `corpus_dir`, return non-empty lines.

    Lines that look like markdown headers are skipped.
    """
    facts: list[str] = []
    for md in sorted(corpus_dir.glob("*.md")):
        for raw in md.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("**"):
                continue
            facts.append(line)
    return facts


def build_retriever(
    facts: Iterable[str] | None = None,
    *,
    chunk_size: int = 320,
    chunk_overlap: int = 40,
    k: int = 3,
    embed_model: str = DEFAULT_EMBED_MODEL,
):
    """Build a FAISS retriever over the corpus."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    facts = list(facts) if facts is not None else load_corpus()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text("\n".join(facts))
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectordb = FAISS.from_texts(chunks, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": k})
