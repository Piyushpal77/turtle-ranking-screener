"""Main CLI entrypoint for the Turtle Trend Following US Stock Ranking Screener."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List, Optional

import pandas as pd

from config import CHARTS_DIR, DB_PATH, TOP_N
from dashboard import DASHBOARD_HTML_PATH, generate_dashboard_html
from database import (
    delete_recommendations_on_or_after,
    get_all_recommendations,
    get_ticker_metadata,
    get_trading_dates,
    init_db,
    load_price_histories,
    sync_market_data,
    upsert_tickers,
)
from indicators import compute_indicators
from ranker import screen_and_rank_universe
from tracker import (
    register_new_recommendations,
    save_daily_csv_reports,
    update_active_recommendations,
)
from universe import build_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("turtle_screener")


DISPLAY_COLS = [
    "grade",
    "turtle_score",
    "ticker",
    "system",
    "rec_price",
    "breakout_level",
    "initial_exit_level",
    "breakout_n",
    "atr_pct",
    "hma_slope_n",
    "rsi_14",
    "macd_hist_n",
    "stoch_k",
    "adx_14",
    "rvol",
    "roc_60",
    "dist_sma200_pct",
]

DISPLAY_HEADERS = {
    "grade": "Grade",
    "turtle_score": "Score",
    "ticker": "Ticker",
    "system": "System",
    "rec_price": "Close",
    "breakout_level": "Brk_Lvl",
    "initial_exit_level": "Exit_Lvl",
    "breakout_n": "Brk_N",
    "atr_pct": "ATR_%",
    "hma_slope_n": "HMA_Slp",
    "rsi_14": "RSI_14",
    "macd_hist_n": "MACD_N",
    "stoch_k": "Stoch_K",
    "adx_14": "ADX_14",
    "rvol": "RVOL",
    "roc_60": "60d_Ret%",
    "dist_sma200_pct": "SMA200_%",
}


def _print_console_table(title: str, df: pd.DataFrame, scan_date: str) -> None:
    print("\n" + "=" * 118)
    print(f" {title}  (Scan Date: {scan_date})")
    print("=" * 118)
    if df.empty:
        print("  No fresh Day-1 breakouts qualified on this session.")
        return

    avail_cols = [c for c in DISPLAY_COLS if c in df.columns]
    view = df[avail_cols].rename(columns=DISPLAY_HEADERS).copy()
    view.index = range(1, len(view) + 1)
    view.index.name = "Rank"
    print(view.to_string())


def run_single_date_scan(
    indicator_frames: Dict[str, pd.DataFrame],
    ticker_meta: Dict[str, Dict[str, str]],
    scan_date: Optional[str] = None,
    top_n: int = TOP_N,
    generate_charts: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Execute the daily lifecycle update + fresh Day-1 breakout screening for `scan_date`:
    1. Update all existing ACTIVE recommendations up through `scan_date` (freezing any that hit Turtle Exit).
    2. Screen universe for fresh Day-1 breakouts on `scan_date`, excluding tickers still ACTIVE in that direction.
    3. Register today's Top-N Long and Top-N Short recommendations in SQLite and generate HTML charts.
    """
    # Resolve effective scan date
    if scan_date is None:
        all_latest = [str(df.index[-1]) for df in indicator_frames.values() if not df.empty]
        effective_date = pd.Series(all_latest).mode().iloc[0] if all_latest else ""
    else:
        effective_date = scan_date

    # Ensure idempotency when re-running on the same scan_date
    if effective_date:
        delete_recommendations_on_or_after(effective_date, db_path=DB_PATH)

    # Step 1: Update existing ACTIVE recommendations up to effective_date
    _, active_longs, active_shorts = update_active_recommendations(
        indicator_frames=indicator_frames,
        as_of_date=effective_date,
        generate_charts=generate_charts,
        db_path=DB_PATH,
        charts_dir=CHARTS_DIR,
    )

    # Step 2: Screen and rank fresh Day-1 breakouts on effective_date
    top_longs, top_shorts, effective_date = screen_and_rank_universe(
        indicator_frames=indicator_frames,
        ticker_meta=ticker_meta,
        scan_date=effective_date,
        active_long_tickers=active_longs,
        active_short_tickers=active_shorts,
        top_n=top_n,
    )

    # Step 3: Persist new picks and build their HTML charts
    register_new_recommendations(
        top_longs=top_longs,
        top_shorts=top_shorts,
        indicator_frames=indicator_frames,
        generate_charts=generate_charts,
        db_path=DB_PATH,
        charts_dir=CHARTS_DIR,
    )

    if verbose and effective_date:
        _print_console_table(f"TOP {top_n} LONG BREAKOUT CANDIDATES", top_longs, effective_date)
        _print_console_table(f"TOP {top_n} SHORT BREAKOUT CANDIDATES", top_shorts, effective_date)

    return top_longs, top_shorts, effective_date


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turtle Trend Following US Stock Ranking Screener & Live Excel/Chart Tracker"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run scan as of a specific YYYY-MM-DD trading date (default: latest EOD).",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=0,
        help="Replay chronological daily scans over the last N trading days to populate tracker history.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"Number of top ranked Long and Short candidates to select per day (default: {TOP_N}).",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Use existing SQLite EOD data without downloading from Yahoo Finance.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only refresh universe and sync EOD market data into SQLite, then exit.",
    )
    parser.add_argument(
        "--force-full-sync",
        action="store_true",
        help="Force full 2-year OHLCV re-download for all universe tickers.",
    )
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Force refresh of S&P 500 & Nasdaq 100 constituent list from Wikipedia.",
    )
    args = parser.parse_args(argv)

    init_db(DB_PATH)

    # 1. Build / load constituent universe
    if not args.skip_sync or args.refresh_universe:
        constituents = build_universe(force_refresh=args.refresh_universe)
        if constituents:
            upsert_tickers(constituents, db_path=DB_PATH)
            logger.info("Universe ready: %d unique liquid US equities.", len(constituents))

    ticker_meta = get_ticker_metadata(db_path=DB_PATH)
    tickers = sorted(ticker_meta.keys())

    if not tickers:
        logger.error("No tickers found in universe. Run without --skip-sync first.")
        return 1

    # 2. Incremental EOD synchronization into SQLite
    if not args.skip_sync:
        sync_stats = sync_market_data(
            tickers=tickers,
            force_full_refresh=args.force_full_sync,
            db_path=DB_PATH,
        )
        logger.info(
            "SQLite EOD sync complete | Full: %d tickers (%d bars) | Incremental: %d tickers (%d bars) | Split-resynced: %d",
            sync_stats["cold_tickers"],
            sync_stats["full_rows_written"],
            sync_stats["incremental_tickers"],
            sync_stats["inc_rows_written"],
            sync_stats["split_refreshed_tickers"],
        )
        if args.sync_only:
            return 0

    # 3. Load OHLCV histories from SQLite & compute indicators
    logger.info("Loading price histories from SQLite and computing technical indicators...")
    raw_histories = load_price_histories(tickers=tickers, as_of_date=args.date, db_path=DB_PATH)
    if not raw_histories:
        logger.error("No price data available in SQLite database.")
        return 1

    indicator_frames: Dict[str, pd.DataFrame] = {}
    for tkr, df in raw_histories.items():
        if len(df) >= 30:
            indicator_frames[tkr] = compute_indicators(df)

    logger.info("Computed indicator suite for %d tickers.", len(indicator_frames))

    # 4. Run single scan or chronological backfill replay
    if args.backfill_days and args.backfill_days > 1:
        trading_dates = get_trading_dates(limit=args.backfill_days, db_path=DB_PATH)
        if args.date:
            trading_dates = [d for d in trading_dates if d <= args.date]
        logger.info(
            "Replaying chronological daily screener across %d trading sessions (%s -> %s)...",
            len(trading_dates),
            trading_dates[0],
            trading_dates[-1],
        )
        top_longs, top_shorts, effective_date = pd.DataFrame(), pd.DataFrame(), ""
        for idx, dt_str in enumerate(trading_dates):
            is_last = idx == len(trading_dates) - 1
            # Defer chart generation until the final date of backfill for speed,
            # except when a pick freezes on an intermediate date (handled on final pass too)
            top_longs, top_shorts, effective_date = run_single_date_scan(
                indicator_frames=indicator_frames,
                ticker_meta=ticker_meta,
                scan_date=dt_str,
                top_n=args.top_n,
                generate_charts=is_last,
                verbose=is_last,
            )
        # Ensure all historical recommendations created during backfill have up-to-date HTML charts
        from charting import generate_recommendation_chart
        all_recs = get_all_recommendations(db_path=DB_PATH)
        logger.info("Generating interactive HTML charts for %d tracked recommendations...", len(all_recs))
        for rec in all_recs:
            tkr = str(rec["ticker"])
            if tkr in indicator_frames:
                generate_recommendation_chart(rec, indicator_frames[tkr], charts_dir=CHARTS_DIR)
    else:
        top_longs, top_shorts, effective_date = run_single_date_scan(
            indicator_frames=indicator_frames,
            ticker_meta=ticker_meta,
            scan_date=args.date,
            top_n=args.top_n,
            generate_charts=True,
            verbose=True,
        )

    if not effective_date:
        logger.warning("No valid trading date found for screening.")
        return 0

    # 5. Export in-system Web UI Dashboard (with built-in Long & Short Spreadsheets) & daily CSV snapshots
    long_csv, short_csv = save_daily_csv_reports(top_longs, top_shorts, effective_date)
    dash_path = generate_dashboard_html(indicator_frames=indicator_frames, db_path=DB_PATH, output_path=DASHBOARD_HTML_PATH)

    all_tracked = get_all_recommendations(db_path=DB_PATH)
    active_count = sum(1 for r in all_tracked if r["status"] == "ACTIVE")
    exited_count = sum(1 for r in all_tracked if r["status"] == "EXITED")

    print("\n" + "-" * 118)
    print(f" [SUMMARY] Scan Date: {effective_date}")
    print(f"   • Today's Qualified Picks : {len(top_longs)} Longs | {len(top_shorts)} Shorts")
    print(f"   • Tracker Database State  : {len(all_tracked)} Total Tracked ({active_count} ACTIVE | {exited_count} EXITED & FROZEN)")
    print(f"   • System UI & Spreadsheets: {dash_path}")
    print(f"   • Interactive HTML Charts : {CHARTS_DIR}")
    print(f"   • Daily CSV Snapshots     : {long_csv.name}, {short_csv.name}")
    print("-" * 118 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
