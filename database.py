"""SQLite persistence and incremental EOD market data synchronization."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    DB_PATH,
    INCREMENTAL_OVERLAP_DAYS,
    INITIAL_LOOKBACK_PERIOD,
    MIN_TRADING_DAYS_REQUIRED,
    SPLIT_TOLERANCE_PCT,
)

logger = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with WAL mode enabled for performance."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite schema for tickers, daily_prices, and recommendations."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                source_index TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                PRIMARY KEY (ticker, date)
            );

            CREATE INDEX IF NOT EXISTS idx_daily_prices_date
                ON daily_prices (date);

            CREATE TABLE IF NOT EXISTS recommendations (
                rec_id TEXT PRIMARY KEY,
                rec_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                company_name TEXT,
                sector TEXT,
                direction TEXT NOT NULL,
                system TEXT NOT NULL,
                upgrade_note TEXT DEFAULT '',
                grade TEXT NOT NULL,
                turtle_score REAL NOT NULL,
                rec_price REAL NOT NULL,
                breakout_level REAL NOT NULL,
                initial_exit_level REAL NOT NULL,
                atr_n REAL NOT NULL,
                atr_pct REAL NOT NULL,
                breakout_n REAL NOT NULL,
                hma_20 REAL NOT NULL,
                hma_slope_n REAL NOT NULL,
                rsi_14 REAL NOT NULL,
                macd_hist_n REAL NOT NULL,
                stoch_k REAL NOT NULL,
                stoch_d REAL NOT NULL,
                adx_14 REAL NOT NULL,
                rvol REAL NOT NULL,
                roc_60 REAL NOT NULL,
                dist_sma200_pct REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                latest_date TEXT NOT NULL,
                latest_price REAL NOT NULL,
                current_exit_level REAL NOT NULL,
                return_pct REAL NOT NULL DEFAULT 0.0,
                max_favorable_pct REAL NOT NULL DEFAULT 0.0,
                max_adverse_pct REAL NOT NULL DEFAULT 0.0,
                days_held INTEGER NOT NULL DEFAULT 0,
                exit_date TEXT DEFAULT '',
                exit_price REAL DEFAULT NULL,
                chart_path TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_recommendations_status
                ON recommendations (status);
            CREATE INDEX IF NOT EXISTS idx_recommendations_ticker_dir
                ON recommendations (ticker, direction, status);
            """
        )


def upsert_tickers(constituents: List[Dict[str, str]], db_path: Path = DB_PATH) -> None:
    """Insert or update ticker metadata in SQLite."""
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            c["ticker"],
            c.get("company_name", c["ticker"]),
            c.get("sector", "Unknown"),
            c.get("source_index", "Unknown"),
            now_iso,
        )
        for c in constituents
    ]
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO tickers (ticker, company_name, sector, source_index, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name=excluded.company_name,
                sector=excluded.sector,
                source_index=excluded.source_index,
                updated_at=excluded.updated_at;
            """,
            rows,
        )


def get_ticker_metadata(db_path: Path = DB_PATH) -> Dict[str, Dict[str, str]]:
    """Return mapping of ticker -> {company_name, sector, source_index}."""
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT ticker, company_name, sector, source_index FROM tickers;")
        return {
            row["ticker"]: {
                "company_name": row["company_name"] or row["ticker"],
                "sector": row["sector"] or "Unknown",
                "source_index": row["source_index"] or "Unknown",
            }
            for row in cur.fetchall()
        }


def _extract_ticker_frames(raw_df: pd.DataFrame, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Normalize yfinance multi-index or single-ticker DataFrame into per-ticker OHLCV DataFrames."""
    result: Dict[str, pd.DataFrame] = {}
    if raw_df is None or raw_df.empty:
        return result

    is_multi = isinstance(raw_df.columns, pd.MultiIndex)
    for tkr in tickers:
        try:
            if is_multi:
                if tkr in raw_df.columns.get_level_values(0):
                    sub = raw_df[tkr].copy()
                elif tkr in raw_df.columns.get_level_values(1):
                    sub = raw_df.xs(tkr, axis=1, level=1).copy()
                else:
                    continue
            else:
                if len(tickers) == 1:
                    sub = raw_df.copy()
                else:
                    continue

            sub = sub.rename(columns={c: str(c).lower() for c in sub.columns})
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in sub.columns for col in required_cols):
                continue

            sub = sub[required_cols].dropna(subset=["open", "high", "low", "close"])
            if sub.empty:
                continue

            sub["volume"] = sub["volume"].fillna(0).astype(np.int64)
            idx = pd.to_datetime(sub.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            sub.index = idx.strftime("%Y-%m-%d")
            sub = sub[~sub.index.duplicated(keep="last")].sort_index()
            result[tkr] = sub
        except Exception as exc:
            logger.debug("Skipping malformed download frame for %s: %s", tkr, exc)
    return result


def _download_batch(
    tickers: List[str],
    period: Optional[str] = None,
    start: Optional[str] = None,
    batch_size: int = 100,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV bars from yfinance in batches."""
    frames: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        kwargs: Dict[str, Any] = {
            "tickers": chunk,
            "auto_adjust": True,
            " group_by": "ticker",
            "progress": False,
            "threads": True,
        }
        # Clean key name
        kwargs.pop(" group_by", None)
        kwargs["group_by"] = "ticker"
        if start:
            kwargs["start"] = start
        else:
            kwargs["period"] = period or INITIAL_LOOKBACK_PERIOD

        try:
            raw = yf.download(**kwargs)
            frames.update(_extract_ticker_frames(raw, chunk))
        except Exception as exc:
            logger.warning("Batch download failed for chunk %d..%d: %s", i, i + len(chunk), exc)
    return frames


def _save_ticker_frames(
    frames: Dict[str, pd.DataFrame],
    replace_full_history: bool = False,
    db_path: Path = DB_PATH,
) -> int:
    """Persist per-ticker OHLCV DataFrames into daily_prices."""
    if not frames:
        return 0

    total_rows = 0
    with get_connection(db_path) as conn:
        if replace_full_history:
            conn.executemany(
                "DELETE FROM daily_prices WHERE ticker = ?;",
                [(tkr,) for tkr in frames.keys()],
            )

        rows_to_upsert: List[Tuple[str, str, float, float, float, float, int]] = []
        for tkr, df in frames.items():
            for dt_str, row in df.iterrows():
                rows_to_upsert.append(
                    (
                        tkr,
                        str(dt_str),
                        round(float(row["open"]), 4),
                        round(float(row["high"]), 4),
                        round(float(row["low"]), 4),
                        round(float(row["close"]), 4),
                        int(row["volume"]),
                    )
                )
        conn.executemany(
            """
            INSERT INTO daily_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume;
            """,
            rows_to_upsert,
        )
        total_rows = len(rows_to_upsert)
    return total_rows


def sync_market_data(
    tickers: List[str],
    force_full_refresh: bool = False,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Synchronize daily OHLCV prices into SQLite.
    - When SQLite already has historical data, never re-downloads 2-year data; instead checks if
      new EOD dates exist after the latest stored SQLite date and downloads ONLY new EOD bars.
    - Full 2y download runs only on an empty database (first-time setup) or when --force-full-sync is passed.
    """
    init_db(db_path)

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT ticker, COUNT(*) AS bar_count, MAX(date) AS max_date
            FROM daily_prices
            GROUP BY ticker;
            """
        )
        existing_stats = {
            row["ticker"]: (int(row["bar_count"]), str(row["max_date"]))
            for row in cur.fetchall()
        }

    full_rows_written = 0
    inc_rows_written = 0

    # Only download full 2-year history if database is completely empty or force_full_refresh=True
    if force_full_refresh or not existing_stats:
        logger.info("Initial database setup: downloading %s history for %d tickers...", INITIAL_LOOKBACK_PERIOD, len(tickers))
        cold_frames = _download_batch(tickers, period=INITIAL_LOOKBACK_PERIOD)
        full_rows_written += _save_ticker_frames(cold_frames, replace_full_history=True, db_path=db_path)
        return {
            "cold_tickers": len(tickers),
            "incremental_tickers": 0,
            "split_refreshed_tickers": 0,
            "full_rows_written": full_rows_written,
            "inc_rows_written": 0,
        }

    # Database already has historical data -> search for and download ONLY new EOD prices after db_max_date
    db_max_date = max(max_dt for _, max_dt in existing_stats.values())
    next_day = (datetime.strptime(db_max_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.now().strftime("%Y-%m-%d")

    if next_day > today_str:
        logger.info("SQLite database is already up to date through EOD %s (no new EOD download required).", db_max_date)
        return {
            "cold_tickers": 0,
            "incremental_tickers": 0,
            "split_refreshed_tickers": 0,
            "full_rows_written": 0,
            "inc_rows_written": 0,
        }

    # Quick 1-ticker probe (AAPL) to check if a new EOD trading session has closed on Yahoo Finance after db_max_date
    logger.info("Checking for new EOD closing prices after %s...", db_max_date)
    probe_frames = _download_batch(["AAPL"], start=next_day, batch_size=1)
    probe_df = probe_frames.get("AAPL")
    if probe_df is None or probe_df.empty or str(probe_df.index[-1]) <= db_max_date:
        logger.info("No new EOD trading session found after %s. Using existing SQLite database.", db_max_date)
        return {
            "cold_tickers": 0,
            "incremental_tickers": 0,
            "split_refreshed_tickers": 0,
            "full_rows_written": 0,
            "inc_rows_written": 0,
        }

    # New EOD session(s) exist after db_max_date -> download ONLY the new EOD dates (start=next_day)
    active_db_tickers = [tkr for tkr in tickers if tkr in existing_stats]
    logger.info(
        "New EOD data available (%s -> %s). Downloading EOD prices only for %d tickers...",
        next_day,
        str(probe_df.index[-1]),
        len(active_db_tickers),
    )
    inc_frames = _download_batch(active_db_tickers, start=next_day)
    # Filter strictly to dates > db_max_date
    new_eod_frames: Dict[str, pd.DataFrame] = {}
    for tkr, df in inc_frames.items():
        sub = df[df.index > db_max_date]
        if not sub.empty:
            new_eod_frames[tkr] = sub

    inc_rows_written += _save_ticker_frames(new_eod_frames, replace_full_history=False, db_path=db_path)

    return {
        "cold_tickers": 0,
        "incremental_tickers": len(new_eod_frames),
        "split_refreshed_tickers": 0,
        "full_rows_written": 0,
        "inc_rows_written": inc_rows_written,
    }


def load_price_histories(
    tickers: Optional[Iterable[str]] = None,
    as_of_date: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV histories from SQLite into a dictionary of DataFrames indexed by date."""
    query = "SELECT ticker, date, open, high, low, close, volume FROM daily_prices"
    params: List[Any] = []
    if as_of_date:
        query += " WHERE date <= ?"
        params.append(as_of_date)
    query += " ORDER BY ticker, date ASC;"

    with get_connection(db_path) as conn:
        df_all = pd.read_sql_query(query, conn, params=params)

    if df_all.empty:
        return {}

    if tickers is not None:
        allowed = set(tickers)
        df_all = df_all[df_all["ticker"].isin(allowed)]

    histories: Dict[str, pd.DataFrame] = {}
    for tkr, group in df_all.groupby("ticker", sort=False):
        sub = group.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
        histories[str(tkr)] = sub
    return histories


def get_trading_dates(limit: int = 60, db_path: Path = DB_PATH) -> List[str]:
    """Return the most recent distinct trading dates present in daily_prices (ascending)."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT date FROM (
                SELECT date, COUNT(*) as cnt
                FROM daily_prices
                GROUP BY date
                HAVING cnt >= 10
                ORDER BY date DESC
                LIMIT ?
            )
            ORDER BY date ASC;
            """,
            (limit,),
        )
        return [str(r["date"]) for r in cur.fetchall()]


def get_active_recommendations(
    before_or_on_date: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Fetch all currently ACTIVE recommendations from SQLite."""
    query = "SELECT * FROM recommendations WHERE status = 'ACTIVE'"
    params: List[Any] = []
    if before_or_on_date:
        query += " AND rec_date < ?"
        params.append(before_or_on_date)
    query += " ORDER BY rec_date ASC, turtle_score DESC;"
    with get_connection(db_path) as conn:
        cur = conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def delete_recommendations_on_or_after(start_date: str, db_path: Path = DB_PATH) -> int:
    """Remove recommendations on or after `start_date` so same-day re-runs and backfills are idempotent."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM recommendations WHERE rec_date >= ?;", (start_date,))
        return int(cur.rowcount or 0)


def get_date_wise_closing_prices(
    recommendations: List[Dict[str, Any]],
    db_path: Path = DB_PATH,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    For all tracked recommendations, fetch date-wise EOD closing prices from the earliest
    `rec_date` through each recommendation's `latest_date` (or `exit_date` if EXITED).
    Returns:
      - sorted list of trading dates (`all_dates`)
      - mapping `rec_id -> {date_str: close_price}` (only populated from rec_date up to latest_date/exit_date)
    """
    if not recommendations:
        return [], {}

    min_rec_date = min(str(r["rec_date"]) for r in recommendations)
    tickers = sorted({str(r["ticker"]) for r in recommendations})
    placeholders = ", ".join(["?"] * len(tickers))

    sql = f"""
        SELECT ticker, date, close
        FROM daily_prices
        WHERE date >= ? AND ticker IN ({placeholders})
        ORDER BY date ASC;
    """
    params: List[Any] = [min_rec_date] + tickers
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

    ticker_date_close: Dict[str, Dict[str, float]] = {}
    all_dates_set = set()
    for row in rows:
        tkr = str(row["ticker"])
        dt = str(row["date"])
        cl = round(float(row["close"]), 2)
        ticker_date_close.setdefault(tkr, {})[dt] = cl
        all_dates_set.add(dt)

    all_dates = sorted(all_dates_set)
    rec_date_closes: Dict[str, Dict[str, float]] = {}
    for rec in recommendations:
        rec_id = str(rec["rec_id"])
        tkr = str(rec["ticker"])
        start_dt = str(rec["rec_date"])
        status = str(rec.get("status", "ACTIVE"))
        end_dt = str(rec.get("exit_date") if status == "EXITED" and rec.get("exit_date") else rec.get("latest_date", start_dt))

        tkr_map = ticker_date_close.get(tkr, {})
        rec_map: Dict[str, float] = {}
        for dt in all_dates:
            if start_dt <= dt <= end_dt and dt in tkr_map:
                rec_map[dt] = tkr_map[dt]
        rec_date_closes[rec_id] = rec_map

    return all_dates, rec_date_closes


def get_all_recommendations(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Fetch all recommendations (both ACTIVE and EXITED) ordered by rec_date DESC, direction, score DESC."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM recommendations
            ORDER BY rec_date DESC, direction ASC, turtle_score DESC;
            """
        )
        return [dict(row) for row in cur.fetchall()]


def upsert_recommendations(records: List[Dict[str, Any]], db_path: Path = DB_PATH) -> None:
    """Insert or update recommendation records in SQLite."""
    if not records:
        return
    columns = [
        "rec_id",
        "rec_date",
        "ticker",
        "company_name",
        "sector",
        "direction",
        "system",
        "upgrade_note",
        "grade",
        "turtle_score",
        "rec_price",
        "breakout_level",
        "initial_exit_level",
        "atr_n",
        "atr_pct",
        "breakout_n",
        "hma_20",
        "hma_slope_n",
        "rsi_14",
        "macd_hist_n",
        "stoch_k",
        "stoch_d",
        "adx_14",
        "rvol",
        "roc_60",
        "dist_sma200_pct",
        "status",
        "latest_date",
        "latest_price",
        "current_exit_level",
        "return_pct",
        "max_favorable_pct",
        "max_adverse_pct",
        "days_held",
        "exit_date",
        "exit_price",
        "chart_path",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    col_names = ", ".join(columns)
    update_clause = ", ".join([f"{col}=excluded.{col}" for col in columns if col != "rec_id"])

    sql = f"""
        INSERT INTO recommendations ({col_names})
        VALUES ({placeholders})
        ON CONFLICT(rec_id) DO UPDATE SET
            {update_clause};
    """
    rows = [tuple(rec.get(col) for col in columns) for rec in records]
    with get_connection(db_path) as conn:
        conn.executemany(sql, rows)
