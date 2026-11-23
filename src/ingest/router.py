"""Router: decide how each raw file is loaded and chunked, and run ingestion.

Each file type has a Route: which loader reads it and which chunk settings apply.
A file's type is decided by its extension; if the extension is missing or
unknown, the first bytes are checked (a PDF saved as .bin is still a PDF).

    from ingest.router import ingest_directory
    chunks, report = ingest_directory("data/raw")              # flat chunks
    chunks, report = ingest_directory("data/raw", "parent_child")

    python -m ingest.router [data/raw] [--strategy parent_child] [--out data/processed/chunks.jsonl]
                                                               (run from src/)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

from .chunkers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_documents,
    parent_child_chunks,
)
from .loaders import (
    AUDIO_EXTS,
    DEFAULT_RAW_DIR,
    HTML_EXTS,
    PDF_EXTS,
    Document,
    _rel,
    load_audio,
    load_html,
    load_pdf,
)

logger = logging.getLogger(__name__)

Strategy = Literal["flat", "parent_child"]


@dataclass(frozen=True)
class Route:
    doc_type: str
    extensions: frozenset[str]
    loader: Callable[[Path], list[Document]]
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP
    parent_size: int = 1536
    child_size: int = 256
    child_overlap: int = 32


ROUTES: list[Route] = [
    Route("pdf", frozenset(PDF_EXTS), load_pdf),
    Route("html", frozenset(HTML_EXTS), load_html),
    Route("audio", frozenset(AUDIO_EXTS), load_audio),
]

_BY_EXT = {ext: r for r in ROUTES for ext in r.extensions}
_BY_TYPE = {r.doc_type: r for r in ROUTES}


# --------------------------------------------------------------------------- routing


def _sniff(path: Path) -> str | None:
    """Guess the type from the file's first bytes. None if unrecognized."""
    try:
        with path.open("rb") as f:
            head = f.read(512)
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "pdf"
    if (
        head.startswith((b"ID3", b"fLaC", b"OggS"))
        or (head[:4] == b"RIFF" and head[8:12] == b"WAVE")
        or head[4:8] == b"ftyp"  # mp4 / m4a
        or head.startswith(b"\x1a\x45\xdf\xa3")  # webm / mkv
        or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)  # raw mp3 frame
    ):
        return "audio"
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if text.startswith((b"<!doctype html", b"<html")) or b"<html" in text:
        return "html"
    return None


def route_for(path: str | Path) -> Route | None:
    """The Route for a file, or None if it is not a supported type."""
    path = Path(path)
    route = _BY_EXT.get(path.suffix.lower())
    if route is None:
        doc_type = _sniff(path)
        route = _BY_TYPE.get(doc_type) if doc_type else None
    return route


def iter_files(root: str | Path = DEFAULT_RAW_DIR) -> Iterator[tuple[Path, Route | None]]:
    """Every file under root with its Route (None = unsupported), sorted.

    Skips hidden files/folders and the '<page>_files/' asset folders that a
    browser's "Save page as" writes next to an .html file.
    """
    root = Path(root)
    if root.is_file():
        yield root, route_for(root)
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and not d.endswith("_files"))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            yield path, route_for(path)


# --------------------------------------------------------------------------- ingestion


def load_file(path: str | Path) -> list[Document]:
    """Load one file with its route's loader. Raises ValueError if unsupported."""
    path = Path(path)
    route = route_for(path)
    if route is None:
        raise ValueError(f"Unsupported file type: {path}")
    return route.loader(path)


def chunk_file_documents(docs: list[Document], route: Route, strategy: Strategy = "flat") -> list[Chunk]:
    """Chunk one file's Documents with that route's settings."""
    if strategy == "flat":
        return chunk_documents(docs, route.chunk_size, route.overlap)
    if strategy == "parent_child":
        return parent_child_chunks(docs, route.parent_size, route.child_size, route.child_overlap)
    raise ValueError(f"Unknown strategy: {strategy!r}")


@dataclass
class FileResult:
    source: str
    doc_type: str | None
    status: Literal["ok", "empty", "unsupported", "error"]
    n_docs: int = 0
    n_chunks: int = 0
    seconds: float = 0.0
    error: str | None = None


@dataclass
class IngestReport:
    files: list[FileResult] = field(default_factory=list)

    def by_status(self, status: str) -> list[FileResult]:
        return [f for f in self.files if f.status == status]

    def summary(self) -> str:
        lines = []
        for f in self.files:
            if f.error:
                detail = f.error
            elif f.status == "unsupported":
                detail = "skipped"
            else:
                detail = f"{f.n_docs:4d} docs {f.n_chunks:5d} chunks {f.seconds:7.1f}s"
            lines.append(f"  {f.status:<11} {f.doc_type or '-':<6} {detail}  {f.source}")
        ok = self.by_status("ok")
        lines.append(
            f"{len(ok)} files ingested, {sum(f.n_chunks for f in ok)} chunks; "
            f"{len(self.by_status('empty'))} empty, {len(self.by_status('unsupported'))} unsupported, "
            f"{len(self.by_status('error'))} failed"
        )
        return "\n".join(lines)


def ingest_file(path: str | Path, strategy: Strategy = "flat") -> tuple[list[Chunk], FileResult]:
    """Route, load and chunk one file. Never raises: failures go into the FileResult."""
    path = Path(path)
    route = route_for(path)
    if route is None:
        return [], FileResult(_rel(path), None, "unsupported")
    t0 = time.perf_counter()
    try:
        docs = route.loader(path)
        chunks = chunk_file_documents(docs, route, strategy) if docs else []
    except Exception as exc:
        logger.exception("Failed to ingest %s", path)
        return [], FileResult(_rel(path), route.doc_type, "error", seconds=time.perf_counter() - t0,
                              error=f"{type(exc).__name__}: {exc}")
    status = "ok" if chunks else "empty"
    result = FileResult(_rel(path), route.doc_type, status, len(docs), len(chunks), time.perf_counter() - t0)
    logger.info("%-5s %-6s %4d docs %5d chunks  %s", status, route.doc_type, len(docs), len(chunks), result.source)
    return chunks, result


def ingest_directory(
    root: str | Path = DEFAULT_RAW_DIR,
    strategy: Strategy = "flat",
    types: set[str] | None = None,
) -> tuple[list[Chunk], IngestReport]:
    """Load and chunk every supported file under root.

    types limits ingestion to some doc types, e.g. {"pdf", "html"} to skip
    slow audio transcription. Unsupported files are listed in the report.
    """
    chunks: list[Chunk] = []
    report = IngestReport()
    for path, route in iter_files(root):
        if route is None:
            report.files.append(FileResult(_rel(path), None, "unsupported"))
            continue
        if types and route.doc_type not in types:
            continue
        file_chunks, result = ingest_file(path, strategy)
        chunks.extend(file_chunks)
        report.files.append(result)
    return chunks, report


def load_directory(root: str | Path = DEFAULT_RAW_DIR, types: set[str] | None = None) -> list[Document]:
    """Load (without chunking) every supported file under root; failures are logged and skipped."""
    docs: list[Document] = []
    for path, route in iter_files(root):
        if route is None or (types and route.doc_type not in types):
            continue
        try:
            docs.extend(route.loader(path))
        except Exception:
            logger.exception("Failed to load %s", path)
    return docs


if __name__ == "__main__":
    import argparse

    from .chunkers import save_jsonl

    ap = argparse.ArgumentParser(description="Route, load and chunk raw files")
    ap.add_argument("root", nargs="?", default=DEFAULT_RAW_DIR)
    ap.add_argument("--strategy", choices=["flat", "parent_child"], default="flat")
    ap.add_argument("--types", help="comma-separated subset of: " + ",".join(_BY_TYPE))
    ap.add_argument("--out", help="write chunks as JSONL, e.g. data/processed/chunks.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    types = set(args.types.split(",")) if args.types else None
    if types and not types <= set(_BY_TYPE):
        ap.error(f"unknown types: {types - set(_BY_TYPE)}")
    all_chunks, rep = ingest_directory(args.root, args.strategy, types)
    print("\n" + rep.summary())
    if args.out:
        save_jsonl(all_chunks, args.out)
        print(f"wrote {len(all_chunks)} chunks to {args.out}")
