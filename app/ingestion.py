"""Document ingestion: parse PDF / TXT / CSV / Excel into text chunks."""
import re
from pathlib import Path

import pandas as pd

from . import config


class IngestionError(Exception):
    pass


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware chunking with character overlap."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paras:                      # hard-split paragraphs that are too long
        while len(p) > size:
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n\n" + piece).strip() if tail else piece
        else:
            cur = f"{cur}\n\n{piece}".strip() if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- parsers
def _parse_pdf(path: Path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise IngestionError("PDF is password protected.")
    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Could not open PDF: {e}")
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        for c in chunk_text(txt):
            out.append((c, f"page {i}"))
    if not out:
        raise IngestionError("No extractable text found in PDF (scanned image PDFs need OCR, which is not supported).")
    return out


def _parse_text(path: Path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    chunks = chunk_text(text)
    if not chunks:
        raise IngestionError("File is empty.")
    return [(c, f"part {i}") for i, c in enumerate(chunks, 1)]


def _table_chunks(df: pd.DataFrame, label: str, location_prefix: str):
    """Turn a DataFrame into (a) a schema/summary chunk and (b) row-group chunks."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return []
    truncated = len(df) > config.MAX_TABLE_ROWS
    df = df.head(config.MAX_TABLE_ROWS)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    summary = [f"TABLE SUMMARY for {label}", f"Rows: {len(df)}{' (truncated)' if truncated else ''}",
               f"Columns: {', '.join(cols)}"]
    num = df.select_dtypes(include="number")
    for c in num.columns[:15]:
        s = num[c]
        summary.append(f"- {c}: sum={s.sum():,.2f}, mean={s.mean():,.2f}, min={s.min():,.2f}, max={s.max():,.2f}")
    out = [("\n".join(summary), f"{location_prefix} summary")]

    for start in range(0, len(df), config.ROWS_PER_CHUNK):
        part = df.iloc[start:start + config.ROWS_PER_CHUNK]
        lines = [f"Data from {label} (rows {start + 1}-{start + len(part)})"]
        for _, row in part.iterrows():
            lines.append(" | ".join(f"{c}: {row[c]}" for c in cols if pd.notna(row[c])))
        out.append(("\n".join(lines), f"{location_prefix} rows {start + 1}-{start + len(part)}"))
    return out


def _parse_csv(path: Path, label: str):
    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception:
            continue
    if df is None:
        raise IngestionError("Could not parse CSV file.")
    out = _table_chunks(df, label, "csv")
    if not out:
        raise IngestionError("CSV contains no data.")
    return out


def _parse_excel(path: Path, label: str):
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        raise IngestionError(f"Could not read Excel file: {e}")
    out = []
    for name, df in sheets.items():
        out += _table_chunks(df, f"{label} / sheet '{name}'", f"sheet '{name}'")
    if not out:
        raise IngestionError("Excel file contains no data.")
    return out


PARSERS = {".pdf": _parse_pdf, ".txt": _parse_text, ".md": _parse_text,
           ".csv": _parse_csv, ".xlsx": _parse_excel, ".xlsm": _parse_excel}


def parse_file(path: Path, display_name: str | None = None) -> list[tuple[str, str]]:
    """Returns list of (chunk_text, location_label)."""
    ext = path.suffix.lower()
    if ext not in PARSERS:
        raise IngestionError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXT))}")
    if ext in (".csv", ".xlsx", ".xlsm"):
        return PARSERS[ext](path, display_name or path.name)
    return PARSERS[ext](path)
