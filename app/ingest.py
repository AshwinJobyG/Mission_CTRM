"""Ingestion pipeline: folder -> load -> chunk -> embed -> Chroma."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import vectorstore
from .chunker import chunk_documents
from .config import DATA_DIR
from .loader import load_folder


@dataclass
class IngestReport:
    folder: str
    documents: int
    chunks: int
    stored: int
    skipped: list[str]
    total_in_db: int

    def __str__(self) -> str:
        lines = [
            f"Folder        : {self.folder}",
            f"Documents     : {self.documents}",
            f"Chunks        : {self.chunks}",
            f"Stored        : {self.stored}",
            f"Total in DB   : {self.total_in_db}",
        ]
        if self.skipped:
            lines.append(f"Skipped files : {len(self.skipped)}")
            for s in self.skipped[:10]:
                lines.append(f"  - {s}")
            if len(self.skipped) > 10:
                lines.append(f"  ... and {len(self.skipped) - 10} more")
        return "\n".join(lines)


def ingest_folder(
    folder: str | Path | None = None,
    recursive: bool = True,
    reset: bool = False,
) -> IngestReport:
    folder = Path(folder).expanduser() if folder else DATA_DIR
    if reset:
        vectorstore.reset()

    docs, skipped = load_folder(folder, recursive=recursive)
    chunks = chunk_documents(docs)
    stored = vectorstore.add_chunks(chunks)

    return IngestReport(
        folder=str(folder),
        documents=len(docs),
        chunks=len(chunks),
        stored=stored,
        skipped=skipped,
        total_in_db=vectorstore.count(),
    )
