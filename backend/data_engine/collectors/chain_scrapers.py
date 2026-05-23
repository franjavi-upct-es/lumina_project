# backend/data_engine/collectors/chain_scrapers.py
"""Supply-chain graph builder via SEC EDGAR 10-K filings."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from prometheus_client import Counter

from backend.config.constants import TARGET_TICKERS
from backend.config.settings import get_settings
from backend.data_engine.pipelines.cleaning import validate_supply_chain_edge
from backend.data_engine.storage.timescale import SupplyChainEdge, TimescaleStore

SCC_EDGES_EXTRACTED = Counter("supply_chain_edges_extracted_total", "Edges extracted")
SCC_FILINGS_PROCESSED = Counter(
    "supply_chain_filings_processed_total",
    "10-K filings processed",
    labelnames=("ticker",),
)

_SEC_HEADERS = {"User-Agent": "LuminaV3 research@lumina.local"}
_RELATIONSHIP_KEYWORDS = {
    "customer": "customer_of",
    "customers": "customer_of",
    "supplier": "supplier_of",
    "suppliers": "supplier_of",
    "distributor": "distributor_of",
}
_TICKER_TAG_RE = re.compile(r"\(([A-Z]{1,5}):\s*([A-Z]{1,5})\)")


class SupplyChainBuilder:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def build_graph(self, as_of: date, store: TimescaleStore) -> int:
        edges_total: list[SupplyChainEdge] = []
        async with httpx.AsyncClient(timeout=30.0, headers=_SEC_HEADERS) as client:
            for ticker in TARGET_TICKERS:
                try:
                    edges = await self._process_ticker(client, ticker, as_of)
                    edges_total.extend(edges)
                    SCC_FILINGS_PROCESSED.labels(ticker=ticker).inc()
                except Exception as exc:
                    logger.warning(f"[{ticker}] 10-K parse failed: {exc}")
                await asyncio.sleep(0.2)
        if edges_total:
            n = await store.insert_supply_chain_edges(edges_total)
            logger.success(f"Persisted {n} supply-chain edges")
        return len(edges_total)

    async def _process_ticker(self, client, ticker: str, as_of: date) -> list[SupplyChainEdge]:
        cik = await self._lookup_cik(client, ticker)
        if not cik:
            return []
        filing_url = await self._latest_10k_url(client, cik)
        if not filing_url:
            return []
        text = await self._fetch_filing_text(client, filing_url)
        return self._extract_edges(ticker, text, as_of)

    async def _lookup_cik(self, client, ticker: str) -> str | None:
        resp = await client.get("https://www.sec.gov/files/company_tickers.json")
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
        return None

    async def _latest_10k_url(self, client, cik: str) -> str | None:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = await client.get(url)
        resp.raise_for_status()
        recent = resp.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary = recent.get("primaryDocument", [])
        for form, acc, doc in zip(forms, accessions, primary, strict=False):
            if form == "10-K":
                acc_clean = acc.replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
        return None

    async def _fetch_filing_text(self, client, url: str) -> str:
        resp = await client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "table"]):
            tag.decompose()
        return soup.get_text(separator=" ")[:500_000]

    def _extract_edges(self, source_ticker: str, text: str, as_of: date) -> list[SupplyChainEdge]:
        edges: list[SupplyChainEdge] = []
        valid_from = datetime.combine(as_of, datetime.min.time())
        lower = text.lower()
        for keyword, rel_type in _RELATIONSHIP_KEYWORDS.items():
            idx = 0
            while True:
                pos = lower.find(keyword, idx)
                if pos == -1:
                    break
                window = text[max(0, pos - 200) : pos + 400]
                for _, ticker_candidate in _TICKER_TAG_RE.findall(window):
                    edge_dict = {
                        "source_ticker": source_ticker,
                        "target_ticker": ticker_candidate,
                        "relationship_type": rel_type,
                    }
                    if validate_supply_chain_edge(edge_dict):
                        edges.append(
                            SupplyChainEdge(
                                source_ticker=source_ticker,
                                target_ticker=ticker_candidate,
                                relationship_type=rel_type,
                                weight=0.5,
                                valid_from=valid_from,
                            )
                        )
                        SCC_EDGES_EXTRACTED.inc()
                idx = pos + len(keyword)
        seen = set()
        unique: list[SupplyChainEdge] = []
        for e in edges:
            k = (e.source_ticker, e.target_ticker, e.relationship_type)
            if k not in seen:
                seen.add(k)
                unique.append(e)
        return unique
