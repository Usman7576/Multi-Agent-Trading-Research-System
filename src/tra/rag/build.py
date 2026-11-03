"""Read material-information PDFs and build the search corpus.

uv run --all-extras python -m tra.rag.build --read-material --since 2024-01-01
uv run --all-extras python -m tra.rag.build --chunks
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.rag.corpus import build_chunks
from tra.scrapers.common.http import HttpClient, HttpError
from tra.scrapers.psx.client import USER_AGENT, PSXClient
from tra.scrapers.psx.fetch_reports import report_text

log = logging.getLogger(__name__)
MATERIAL = re.compile(r"material\s+information", re.I)


def read_material(client: PSXClient, lake: ParquetLake, raw_dir: Path, since: str) -> int:
    """Download and read material-information PDFs. Text is cached next to each PDF, so this resumes."""
    ann = lake.read("announcements", "PSX")
    ann = ann[
        ann["title"].map(lambda t: bool(MATERIAL.search(t)))
        & ann["doc_path"].notna()
        & (ann["announced_local"] >= pd.Timestamp(since))
    ].sort_values("announced_local", ascending=False)
    done = 0
    for a in ann.itertuples(index=False):
        path = raw_dir / "reports" / a.symbol / f"{a.announcement_id}.pdf"
        if not path.exists():
            try:
                body = client.download(a.doc_path)
            except HttpError as exc:
                log.warning("%s %s: download failed, skipping (%s)", a.symbol, a.announcement_id, exc)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        report_text(path)
        done += 1
        if done % 25 == 0:
            log.info("Read %d of %d material-information PDFs", done, len(ann))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--read-material", action="store_true")
    parser.add_argument("--chunks", action="store_true")
    parser.add_argument("--since", default="2024-01-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    lake = ParquetLake(settings.data_dir)
    raw = settings.data_dir / "raw" / "psx"
    if args.read_material:
        http = HttpClient(USER_AGENT, requests_per_second=1.0, timeout_seconds=120)
        read_material(PSXClient(http), lake, raw, args.since)
    if args.chunks:
        chunks = build_chunks(lake, raw)
        log.info("Chunks: %d, by type %s", len(chunks), chunks["doc_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
