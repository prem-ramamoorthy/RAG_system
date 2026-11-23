"""Chunkers: split loaded Documents into token-bounded, overlapping Chunks.

Pages of a PDF and time windows of an audio transcript are joined back into one
text per file before splitting, so a chunk can cross a page/window boundary
instead of being cut short there. Each chunk records which pages
(page, page_end) or which time range (start, end) it came from.

Splitting is recursive: paragraphs first, then lines, sentences, words, so a
chunk breaks at the most natural boundary that keeps it under chunk_size tokens.

Two strategies:
    chunk_documents(docs)        flat ~512-token chunks with 64-token overlap
    parent_child_chunks(docs)    small child chunks for retrieval, each carrying
                                 the larger parent text it came from (for
                                 src/retrieval/parent.py)

Token counts use tiktoken's cl100k_base. It is not bge-m3's tokenizer, but it is
close enough for sizing and much faster; bge-m3 accepts up to 8192 tokens.

To chunk a whole folder from the command line, use ingest.router:
    python -m ingest.router [data/raw] [--strategy parent_child] [--out data/processed/chunks.jsonl]
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Iterable

from .loaders import Document

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
# A last chunk smaller than this is folded into the one before it.
MIN_CHUNK_TOKENS = 32

# Most natural boundary first. Each separator stays attached to the piece before it.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

# Joins the parts (pages / windows) of one document.
PART_JOINER = "\n\n"

# Fixed namespace so chunk ids are stable across runs (re-ingesting overwrites, not duplicates).
_ID_NAMESPACE = uuid.UUID("6f1c2b0e-4a57-4d43-9a0e-5b8f3c1d2e7a")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata["chunk_id"]

    def to_payload(self) -> dict:
        """Flat dict for the Qdrant payload / BM25 corpus.

        Matches what src/retrieval/dense.py expects: text, source, type, page.
        """
        payload = {"text": self.text, **self.metadata}
        payload["type"] = payload.get("doc_type")
        return payload


# --------------------------------------------------------------------------- tokens


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


# --------------------------------------------------------------------------- splitting

Span = tuple[int, int]  # [start, end) character offsets into the source text


def _split_spans(text: str, start: int, end: int, max_tokens: int, seps: list[str]) -> list[Span]:
    """Recursively split text[start:end] into spans of at most max_tokens tokens."""
    if count_tokens(text[start:end]) <= max_tokens:
        return [(start, end)]

    segment = text[start:end]
    for i, sep in enumerate(seps):
        if sep not in segment:
            continue
        spans: list[Span] = []
        pos = start
        for piece in segment.split(sep)[:-1]:
            piece_end = pos + len(piece) + len(sep)
            spans.extend(_split_spans(text, pos, piece_end, max_tokens, seps[i + 1 :]))
            pos = piece_end
        if pos < end:
            spans.extend(_split_spans(text, pos, end, max_tokens, seps[i + 1 :]))
        return spans

    # No separator left (e.g. a long URL, a formula, a base64 blob): halve by characters.
    if end - start <= 1:
        return [(start, end)]
    mid = (start + end) // 2
    return _split_spans(text, start, mid, max_tokens, []) + _split_spans(text, mid, end, max_tokens, [])


def split_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Span]:
    """Split text into overlapping spans of about chunk_size tokens.

    Returns (start, end) offsets into text, trimmed of surrounding whitespace.
    No span exceeds chunk_size tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Split into small pieces (about sentence size) so the overlap can carry whole
    # sentences over; contiguous pieces are merged back up to chunk_size below.
    piece_max = min(chunk_size, max(overlap, chunk_size // 8, 16))
    pieces = [s for s in _split_spans(text, 0, len(text), piece_max, SEPARATORS) if text[s[0] : s[1]].strip()]
    def ntok(a: int, b: int) -> int:
        """Tokens in the text covered by pieces a..b inclusive (0 if empty).

        Counted on the joined text: summing per-piece counts undercounts, because
        tokens merge differently across piece boundaries.
        """
        return count_tokens(text[pieces[a][0] : pieces[b][1]]) if a <= b else 0

    windows: list[tuple[int, int]] = []  # inclusive piece-index ranges
    first = 0
    for i in range(len(pieces)):
        if i > first and ntok(first, i) > chunk_size:
            windows.append((first, i - 1))
            # Keep trailing pieces of the last chunk as overlap, but leave room for this piece.
            while first < i and (ntok(first, i - 1) > overlap or ntok(first, i) > chunk_size):
                first += 1
    if pieces:
        windows.append((first, len(pieces) - 1))

    # Fold a tiny tail into the previous chunk if the result still fits.
    if len(windows) > 1:
        (pa, pb), (ta, tb) = windows[-2], windows[-1]
        if ntok(pb + 1, tb) < MIN_CHUNK_TOKENS and ntok(pa, tb) <= chunk_size:
            windows[-2] = (pa, tb)
            windows.pop()

    spans = []
    for a, b in windows:
        s, e = pieces[a][0], pieces[b][1]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        spans.append((s, e))
    return spans


# --------------------------------------------------------------------------- documents


@dataclass
class _Joined:
    text: str
    parts: list[tuple[int, int, dict]]  # (start, end, metadata) of each source Document
    metadata: dict  # shared per-file metadata


_PART_KEYS = {"page", "start", "end", "timestamp", "ocr"}


def _join(docs: list[Document]) -> _Joined:
    """Concatenate the parts of one file, remembering where each part landed."""
    texts, parts, pos = [], [], 0
    for d in docs:
        text = _dehyphenate(d.text) if d.metadata.get("doc_type") == "pdf" else d.text
        parts.append((pos, pos + len(text), d.metadata))
        texts.append(text)
        pos += len(text) + len(PART_JOINER)
    shared = {k: v for k, v in docs[0].metadata.items() if k not in _PART_KEYS}
    return _Joined(PART_JOINER.join(texts), parts, shared)


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def _dehyphenate(text: str) -> str:
    """Rejoin words broken across PDF lines: 'adapta-\\ntion' -> 'adaptation'."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def _locate(joined: _Joined, start: int, end: int) -> dict:
    """Page range / time range covered by the span [start, end)."""
    covered = [m for s, e, m in joined.parts if s < end and e > start]
    if not covered:
        return {}
    first, last = covered[0], covered[-1]
    loc: dict = {}
    if "page" in first:
        loc["page"] = first["page"]
        loc["page_end"] = last["page"]
    if "start" in first:
        loc["start"] = first["start"]
        loc["end"] = last["end"]
        if "timestamp" in first:
            loc["timestamp"] = f'{first["timestamp"].split("-")[0]}-{last["timestamp"].split("-")[-1]}'
    return loc


def _group_by_file(docs: Iterable[Document]) -> list[list[Document]]:
    """Group consecutive Documents that come from the same file (same doc_id)."""
    return [list(g) for _, g in groupby(docs, key=lambda d: d.metadata.get("doc_id") or d.metadata.get("source"))]


def _chunk_id(doc_id: str, kind: str, index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{kind}:{index}"))


def chunk_documents(
    docs: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Flat chunking: every chunk is about chunk_size tokens, overlapping by `overlap`."""
    chunks: list[Chunk] = []
    for group in _group_by_file(docs):
        joined = _join(group)
        spans = split_text(joined.text, chunk_size, overlap)
        doc_id = joined.metadata.get("doc_id", joined.metadata.get("source", ""))
        for i, (s, e) in enumerate(spans):
            text = joined.text[s:e]
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        **joined.metadata,
                        **_locate(joined, s, e),
                        "chunk_id": _chunk_id(doc_id, "chunk", i),
                        "chunk_index": i,
                        "n_chunks": len(spans),
                        "n_tokens": count_tokens(text),
                    },
                )
            )
    return chunks


def parent_child_chunks(
    docs: Iterable[Document],
    parent_size: int = 1536,
    child_size: int = 256,
    child_overlap: int = 32,
) -> list[Chunk]:
    """Small child chunks to embed and search; each carries its parent's id and text.

    Search on children (precise matches), then hand the LLM the parent (more context).
    Parents do not overlap; children overlap within their parent.
    """
    chunks: list[Chunk] = []
    for group in _group_by_file(docs):
        joined = _join(group)
        doc_id = joined.metadata.get("doc_id", joined.metadata.get("source", ""))
        child_index = 0
        for p, (ps, pe) in enumerate(split_text(joined.text, parent_size, 0)):
            parent_text = joined.text[ps:pe]
            parent_id = _chunk_id(doc_id, "parent", p)
            for cs, ce in split_text(parent_text, child_size, child_overlap):
                text = parent_text[cs:ce]
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            **joined.metadata,
                            **_locate(joined, ps + cs, ps + ce),
                            "chunk_id": _chunk_id(doc_id, "child", child_index),
                            "chunk_index": child_index,
                            "n_tokens": count_tokens(text),
                            "parent_id": parent_id,
                            "parent_index": p,
                            "parent_text": parent_text,
                        },
                    )
                )
                child_index += 1
    return chunks


def save_jsonl(chunks: Iterable[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_payload(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Chunk]:
    chunks = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            text = payload.pop("text")
            payload.pop("type", None)
            chunks.append(Chunk(text=text, metadata=payload))
    return chunks
