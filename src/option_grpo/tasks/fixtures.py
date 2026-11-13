"""Data fixtures materialised in the sandbox working directory.

Tasks refer to small files (``data.csv``, ``data.json``, ...) whose content is defined
here.  Every fixture exists in three variants used by the edge-case property tests:
``base`` (six rows), ``empty`` (no rows) and ``single`` (one row).
"""

from __future__ import annotations

import base64
import csv
import io
import json
from dataclasses import dataclass
from typing import Any

ROWS: list[dict[str, Any]] = [
    {"id": 1, "name": "Ada", "department": "eng", "age": 36, "salary": 120000, "score": 0.91},
    {"id": 2, "name": "Grace", "department": "eng", "age": 45, "salary": 135000, "score": 0.87},
    {"id": 3, "name": "Linus", "department": "ops", "age": 29, "salary": 82000, "score": 0.64},
    {"id": 4, "name": "Edsger", "department": "research", "age": 52, "salary": 99000, "score": 0.78},
    {"id": 5, "name": "Barbara", "department": "research", "age": 41, "salary": 104000, "score": 0.83},
    {"id": 6, "name": "Ken", "department": "ops", "age": 33, "salary": 76000, "score": 0.59},
]
COLUMNS: list[str] = list(ROWS[0])
NUMERIC_COLUMNS = ["age", "salary", "score", "id"]
CATEGORICAL_COLUMNS = ["department", "name"]

_TEXT = (
    "Option-GRPO fixture document.\n\n"
    "The first paragraph introduces the topic in two short sentences. It is plain text.\n\n"
    "The second paragraph lists three items: alpha, beta and gamma.\n"
)

# 1x1 transparent PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 20 100 Td (Fixture page) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000054 00000 n \n0000000101 00000 n \n"
    b"0000000208 00000 n \n0000000299 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n362\n%%EOF\n"
)

VARIANTS = ("base", "empty", "single")


@dataclass(frozen=True)
class Fixture:
    """A file or directory created before a task runs."""

    path: str
    kind: str  # csv | tsv | json | jsonl | txt | md | html | yaml | png | pdf | dir
    variant: str = "base"

    def to_json(self) -> dict[str, Any]:
        return materialise(self)

    def with_variant(self, variant: str) -> Fixture:
        return Fixture(self.path, self.kind, variant)


# extension -> fixture kind
EXTENSIONS: dict[str, str] = {
    "csv": "csv",
    "tsv": "tsv",
    "json": "json",
    "jsonl": "jsonl",
    "ndjson": "jsonl",
    "txt": "txt",
    "text": "txt",
    "md": "md",
    "markdown": "md",
    "html": "html",
    "yaml": "yaml",
    "yml": "yaml",
    "png": "png",
    "image": "png",
    "img": "png",
    "pdf": "pdf",
}


def rows_for(variant: str) -> list[dict[str, Any]]:
    if variant == "empty":
        return []
    if variant == "single":
        return [dict(ROWS[0])]
    return [dict(row) for row in ROWS]


def _delimited(rows: list[dict[str, Any]], delimiter: str) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def materialise(fixture: Fixture) -> dict[str, Any]:
    """Serialisable description consumed by the execution harness."""
    rows = rows_for(fixture.variant)
    kind = fixture.kind
    if kind == "dir":
        return {"path": fixture.path, "type": "dir"}
    if kind == "csv":
        text = _delimited(rows, ",")
    elif kind == "tsv":
        text = _delimited(rows, "\t")
    elif kind == "json":
        text = json.dumps(rows, indent=1) + "\n"
    elif kind == "jsonl":
        text = "".join(json.dumps(row) + "\n" for row in rows)
    elif kind == "yaml":
        text = "".join(
            "- " + "\n  ".join(f"{k}: {json.dumps(v)}" for k, v in row.items()) + "\n" for row in rows
        ) or "[]\n"
    elif kind in {"txt", "md", "html"}:
        body = "" if fixture.variant == "empty" else (_TEXT.split("\n\n")[0] + "\n" if fixture.variant == "single" else _TEXT)
        if kind == "md":
            body = "# Fixture\n\n" + body
        elif kind == "html":
            paragraphs = "".join(f"<p>{p.strip()}</p>" for p in body.split("\n\n") if p.strip())
            body = f"<html><head><title>Fixture</title></head><body>{paragraphs}</body></html>\n"
        text = body
    elif kind == "png":
        return {"path": fixture.path, "type": "bytes", "b64": base64.b64encode(_PNG).decode("ascii")}
    elif kind == "pdf":
        return {"path": fixture.path, "type": "bytes", "b64": base64.b64encode(_PDF).decode("ascii")}
    else:
        raise ValueError(f"unknown fixture kind: {kind}")
    return {"path": fixture.path, "type": "text", "content": text}
