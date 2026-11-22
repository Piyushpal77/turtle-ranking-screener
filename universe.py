"""Universe builder for S&P 500, Nasdaq 100, and custom tickers."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

from config import (
    CUSTOM_TICKERS_PATH,
    UNIVERSE_CACHE_HOURS,
    UNIVERSE_CACHE_PATH,
)

logger = logging.getLogger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def normalize_ticker(symbol: str) -> str:
    """Normalize ticker symbol for Yahoo Finance (e.g., BRK.B -> BRK-B)."""
    cleaned = str(symbol).strip().upper()
    return cleaned.replace(".", "-")


def _scrape_sp500() -> List[Dict[str, str]]:
    """Scrape S&P 500 constituents from Wikipedia."""
    resp = requests.get(SP500_WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        symbol = normalize_ticker(row.get("Symbol", ""))
        if not symbol:
            continue
        records.append(
            {
                "ticker": symbol,
                "company_name": str(row.get("Security", symbol)).strip(),
                "sector": str(row.get("GICS Sector", "Unknown")).strip(),
                "source_index": "S&P 500",
            }
        )
    return records


NASDAQ100_FALLBACK_EXTRAS = [
    {"ticker": "ARM", "company_name": "Arm Holdings", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "ASML", "company_name": "ASML Holding", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "AZN", "company_name": "AstraZeneca", "sector": "Health Care", "source_index": "Nasdaq 100"},
    {"ticker": "CCEP", "company_name": "Coca-Cola Europacific Partners", "sector": "Consumer Staples", "source_index": "Nasdaq 100"},
    {"ticker": "FER", "company_name": "Ferrovial", "sector": "Industrials", "source_index": "Nasdaq 100"},
    {"ticker": "GFS", "company_name": "GlobalFoundries", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "MDB", "company_name": "MongoDB", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "MELI", "company_name": "MercadoLibre", "sector": "Consumer Discretionary", "source_index": "Nasdaq 100"},
    {"ticker": "MRVL", "company_name": "Marvell Technology", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "MSTR", "company_name": "MicroStrategy", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "PDD", "company_name": "PDD Holdings", "sector": "Consumer Discretionary", "source_index": "Nasdaq 100"},
    {"ticker": "TEAM", "company_name": "Atlassian", "sector": "Technology", "source_index": "Nasdaq 100"},
    {"ticker": "TTD", "company_name": "The Trade Desk", "sector": "Communication Services", "source_index": "Nasdaq 100"},
    {"ticker": "ZS", "company_name": "Zscaler", "sector": "Technology", "source_index": "Nasdaq 100"},
]


def _scrape_nasdaq100() -> List[Dict[str, str]]:
    """Scrape Nasdaq 100 constituents from Wikipedia with resilient table detection and fallback."""
    resp = requests.get(NASDAQ100_WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    target_df = None
    for tbl in tables:
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [" ".join(str(x) for x in col).strip() for col in tbl.columns]
        cols = [str(c).lower() for c in tbl.columns]
        if any("ticker" in c or "symbol" in c for c in cols) and len(tbl) >= 50:
            target_df = tbl
            break
    if target_df is None:
        return list(NASDAQ100_FALLBACK_EXTRAS)

    cols_str = [str(c) for c in target_df.columns]
    ticker_col = next(
        (c for c in target_df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()),
        target_df.columns[0],
    )
    company_col = next(
        (c for c in target_df.columns if "company" in str(c).lower() or "security" in str(c).lower()),
        target_df.columns[1] if len(cols_str) > 1 else ticker_col,
    )
    sector_col = next(
        (
            c
            for c in target_df.columns
            if "sector" in str(c).lower() or "industry" in str(c).lower()
        ),
        target_df.columns[2] if len(cols_str) > 2 else None,
    )

    records: List[Dict[str, str]] = []
    for _, row in target_df.iterrows():
        symbol = normalize_ticker(row.get(ticker_col, ""))
        if not symbol or len(symbol) > 6:
            continue
        records.append(
            {
                "ticker": symbol,
                "company_name": str(row.get(company_col, symbol)).strip(),
                "sector": str(row.get(sector_col, "Technology")).strip() if sector_col else "Unknown",
                "source_index": "Nasdaq 100",
            }
        )
    return records


def _load_custom_tickers(custom_path: Path = CUSTOM_TICKERS_PATH) -> List[Dict[str, str]]:
    """Load optional user-defined tickers from custom_tickers.txt."""
    if not custom_path.exists():
        return []
    records: List[Dict[str, str]] = []
    for line in custom_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        symbol = normalize_ticker(cleaned.split()[0])
        records.append(
            {
                "ticker": symbol,
                "company_name": symbol,
                "sector": "Custom",
                "source_index": "Custom",
            }
        )
    return records


def build_universe(force_refresh: bool = False) -> List[Dict[str, str]]:
    """Return deduplicated S&P 500 + Nasdaq 100 + Custom universe metadata list."""
    now = datetime.now(timezone.utc)

    if not force_refresh and UNIVERSE_CACHE_PATH.exists():
        try:
            cached = json.loads(UNIVERSE_CACHE_PATH.read_text(encoding="utf-8"))
            cached_time = datetime.fromisoformat(cached["updated_at"])
            if now - cached_time < timedelta(hours=UNIVERSE_CACHE_HOURS) and cached.get("constituents"):
                constituents = cached["constituents"]
                # Merge any custom tickers added since cache was built
                by_ticker = {item["ticker"]: item for item in constituents}
                for item in _load_custom_tickers():
                    if item["ticker"] not in by_ticker:
                        by_ticker[item["ticker"]] = item
                return sorted(by_ticker.values(), key=lambda x: x["ticker"])
        except Exception as exc:
            logger.warning("Failed reading universe cache, refreshing: %s", exc)

    by_ticker: Dict[str, Dict[str, str]] = {}
    try:
        sp500 = _scrape_sp500()
        for item in sp500:
            by_ticker[item["ticker"]] = item
        logger.info("Loaded %d S&P 500 tickers.", len(sp500))
    except Exception as exc:
        logger.warning("Error scraping S&P 500 list: %s", exc)

    try:
        ndx100 = _scrape_nasdaq100()
        for item in ndx100:
            tkr = item["ticker"]
            if tkr in by_ticker:
                by_ticker[tkr]["source_index"] = "S&P 500 + Nasdaq 100"
            else:
                by_ticker[tkr] = item
        logger.info("Loaded %d Nasdaq 100 tickers.", len(ndx100))
    except Exception as exc:
        logger.warning("Error scraping Nasdaq 100 list: %s", exc)

    # If scraping failed completely and cache exists, fallback to stale cache
    if not by_ticker and UNIVERSE_CACHE_PATH.exists():
        logger.warning("Using existing universe_cache.json fallback.")
        cached = json.loads(UNIVERSE_CACHE_PATH.read_text(encoding="utf-8"))
        for item in cached.get("constituents", []):
            by_ticker[item["ticker"]] = item

    # Merge custom tickers
    for item in _load_custom_tickers():
        tkr = item["ticker"]
        if tkr not in by_ticker:
            by_ticker[tkr] = item

    merged = sorted(by_ticker.values(), key=lambda x: x["ticker"])
    if merged:
        payload = {
            "updated_at": now.isoformat(),
            "count": len(merged),
            "constituents": merged,
        }
        UNIVERSE_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return merged
