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
    - Downloads full 2y history for new tickers (or when force_full_refresh=True).
    - Downloads only incremental EOD prices (with 7-day overlap) for existing tickers.
    - Automatically detects split/dividend adjustments on overlap bars and re-fetches full history for affected tickers.
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

    cold_tickers: List[str] = []
    incremental_tickers: List[str] = []

    for tkr in tickers:
        if force_full_refresh or tkr not in existing_stats:
            cold_tickers.append(tkr)
        else:
            count, _ = existing_stats[tkr]
            if count < MIN_TRADING_DAYS_REQUIRED:
                cold_tickers.append(tkr)
            else:
                incremental_tickers.append(tkr)

    full_rows_written = 0
    inc_rows_written = 0
    split_refreshed_tickers: List[str] = []

    if cold_tickers:
        logger.info("Downloading full (%s) history for %d tickers...", INITIAL_LOOKBACK_PERIOD, len(cold_tickers))
        cold_frames = _download_batch(cold_tickers, period=INITIAL_LOOKBACK_PERIOD)
        full_rows_written += _save_ticker_frames(cold_frames, replace_full_history=True, db_path=db_path)

    if incremental_tickers:
        min_max_date = min(existing_stats[tkr][1] for tkr in incremental_tickers)
        overlap_start = (
            datetime.strptime(min_max_date, "%Y-%m-%d") - timedelta(days=INCREMENTAL_OVERLAP_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(
            "Performing incremental EOD sync from %s for %d tickers...",
            overlap_start,
            len(incremental_tickers),
        )
        inc_frames = _download_batch(incremental_tickers, start=overlap_start)

        # Load stored overlap closes to detect overnight splits or dividends
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "SELECT ticker, date, close FROM daily_prices WHERE date >= ?;",
                (overlap_start,),
            )
            stored_overlap: Dict[Tuple[str, str], float] = {
                (row["ticker"], row["date"]): float(row["close"]) for row in cur.fetchall()
            }

        clean_inc_frames: Dict[str, pd.DataFrame] = {}
        for tkr, df in inc_frames.items():
            needs_resync = False
            for dt_str, row in df.iterrows():
                key = (tkr, str(dt_str))
                if key in stored_overlap:
                    old_close = stored_overlap[key]
                    new_close = float(row["close"])
                    if old_close > 0 and abs(new_close - old_close) / old_close > SPLIT_TOLERANCE_PCT:
                        needs_resync = True
                        break
            if needs_resync:
                split_refreshed_tickers.append(tkr)
            else:
                clean_inc_frames[tkr] = df

        inc_rows_written += _save_ticker_frames(clean_inc_frames, replace_full_history=False, db_path=db_path)

        if split_refreshed_tickers:
            logger.info(
                "Detected split/dividend adjustment on %d tickers (%s); refreshing full history...",
                len(split_refreshed_tickers),
                ", ".join(split_refreshed_tickers[:10]),
            )
            resync_frames = _download_batch(split_refreshed_tickers, period=INITIAL_LOOKBACK_PERIOD)
            full_rows_written += _save_ticker_frames(resync_frames, replace_full_history=True, db_path=db_path)

    return {
        "cold_tickers": len(cold_tickers),
        "incremental_tickers": len(incremental_tickers),
        "split_refreshed_tickers": len(split_refreshed_tickers),
        "full_rows_written": full_rows_written,
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
