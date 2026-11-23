"""File loaders: turn raw files under data/raw/ into Document objects.

One loader per format; ingest/router.py decides which one a file goes to.
Layout expected:

    data/raw/
        pdf/    *.pdf                 -> one Document per page (OCR fallback for scanned pages)
        html/   *.html, *.htm         -> one Document per page (main text via trafilatura)
                <name>_files/         -> browser "Save page as" assets, skipped
        audio/  *.mp3, *.wav, ...     -> one Document per time window of a Whisper transcript

Audio transcripts are cached as JSON in data/transcripts/ so re-ingesting
does not re-run Whisper.

Usage:
    from ingest.loaders import load_pdf
    docs = load_pdf("data/raw/pdf/LORA.pdf")

To load a whole folder, use ingest.router.load_directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"

PDF_EXTS = {".pdf"}
HTML_EXTS = {".html", ".htm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}

# A PDF page with fewer extracted characters than this is treated as scanned and OCR'd.
MIN_PAGE_CHARS = 50


@dataclass
class Document:
    """A unit of loaded text plus where it came from.

    metadata always has: source (path relative to the project root), doc_id
    (stable hash of the source path), doc_type ("pdf" | "html" | "audio"),
    and title. Type-specific locators: page / total_pages for PDFs,
    start / end (seconds) for audio.
    """

    text: str
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _doc_id(path: Path) -> str:
    return hashlib.sha1(_rel(path).encode("utf-8")).hexdigest()[:16]


def _base_meta(path: Path, doc_type: str, title: str | None = None) -> dict:
    return {
        "source": _rel(path),
        "doc_id": _doc_id(path),
        "doc_type": doc_type,
        "title": (title or path.stem.replace("_", " ")).strip(),
    }


def _clean(text: str) -> str:
    """Normalize whitespace: strip each line, collapse runs of blank lines."""
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# --------------------------------------------------------------------------- PDF


def load_pdf(path: str | Path, ocr: bool = True, ocr_lang: str = "eng") -> list[Document]:
    """One Document per non-empty page. Pages with almost no text layer are OCR'd."""
    import fitz  # pymupdf

    path = Path(path)
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or None
        meta = _base_meta(path, "pdf", title)
        n_pages = pdf.page_count
        for i, page in enumerate(pdf):
            text = page.get_text("text", sort=True)
            ocr_used = False
            if ocr and len(text.strip()) < MIN_PAGE_CHARS:
                ocr_text = _ocr_page(page, ocr_lang)
                if len(ocr_text.strip()) > len(text.strip()):
                    text, ocr_used = ocr_text, True
            text = _clean(text)
            if not text:
                continue
            docs.append(
                Document(
                    text=text,
                    metadata={**meta, "page": i + 1, "total_pages": n_pages, "ocr": ocr_used},
                )
            )
    return docs


def _ocr_page(page, lang: str, dpi: int = 300) -> str:
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:  # tesseract binary missing, bad page, ...
        logger.warning("OCR failed on page %s: %s", page.number + 1, exc)
        return ""


# --------------------------------------------------------------------------- HTML


def load_html(path: str | Path) -> list[Document]:
    """Main-content text of a saved web page (boilerplate, nav and scripts removed)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")

    text, title = "", None
    try:
        import trafilatura

        text = trafilatura.extract(
            raw,
            include_tables=True,
            include_comments=False,
            include_formatting=False,
            favor_recall=True,
        ) or ""
        md = trafilatura.extract_metadata(raw)
        title = md.title if md else None
    except Exception as exc:
        logger.warning("trafilatura failed on %s: %s", path.name, exc)

    if len(text.strip()) < MIN_PAGE_CHARS:
        text, bs_title = _html_fallback(raw)
        title = title or bs_title

    text = _clean(text)
    if not text:
        logger.warning("No text extracted from %s", path)
        return []
    return [Document(text=text, metadata=_base_meta(path, "html", title))]


def _html_fallback(raw: str) -> tuple[str, str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    root = soup.find("article") or soup.find("main") or soup.body or soup
    return root.get_text("\n"), title


# --------------------------------------------------------------------------- Audio

_whisper_model = None


def _get_whisper():
    """Load faster-whisper once. Configure with WHISPER_MODEL / WHISPER_DEVICE / WHISPER_COMPUTE."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "auto")
        compute = os.getenv("WHISPER_COMPUTE", "default")
        logger.info("Loading Whisper model %s (device=%s)", name, device)
        try:
            _whisper_model = WhisperModel(name, device=device, compute_type=compute)
        except Exception as exc:  # e.g. CTranslate2 has no ROCm build -> fall back to CPU
            logger.warning("Whisper on %s failed (%s); using CPU int8", device, exc)
            _whisper_model = WhisperModel(name, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(path: str | Path, transcript_dir: str | Path = DEFAULT_TRANSCRIPT_DIR) -> dict:
    """Return {"language", "duration", "segments": [{"start","end","text"}]}, cached as JSON.

    The cache is keyed by file name and invalidated when the audio file's size or mtime changes.
    """
    path = Path(path)
    transcript_dir = Path(transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    cache = transcript_dir / f"{path.stem}.json"
    stat = path.stat()
    fingerprint = {"size": stat.st_size, "mtime": int(stat.st_mtime)}

    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fingerprint:
                return data
        except json.JSONDecodeError:
            pass

    logger.info("Transcribing %s", path.name)
    segments, info = _get_whisper().transcribe(str(path), vad_filter=True, beam_size=5)
    data = {
        "source": _rel(path),
        "fingerprint": fingerprint,
        "language": info.language,
        "duration": info.duration,
        "segments": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ],
    }
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_audio(
    path: str | Path,
    window_seconds: float = 120.0,
    transcript_dir: str | Path = DEFAULT_TRANSCRIPT_DIR,
) -> list[Document]:
    """Transcribe and group segments into ~window_seconds Documents with start/end timestamps."""
    path = Path(path)
    data = transcribe(path, transcript_dir)
    meta = {**_base_meta(path, "audio"), "language": data["language"], "duration": data["duration"]}

    docs: list[Document] = []
    buf: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        start, end = buf[0]["start"], buf[-1]["end"]
        docs.append(
            Document(
                text=" ".join(s["text"] for s in buf),
                metadata={
                    **meta,
                    "start": start,
                    "end": end,
                    "timestamp": f"{_fmt_ts(start)}-{_fmt_ts(end)}",
                },
            )
        )
        buf.clear()

    for seg in data["segments"]:
        if buf and seg["end"] - buf[0]["start"] > window_seconds:
            flush()
        buf.append(seg)
    flush()
    return docs
