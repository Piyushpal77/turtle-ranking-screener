"""Unit and integration test suite for the Turtle Trend Following US Stock Ranking Screener."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from charting import generate_recommendation_chart
from config import assign_letter_grade
from database import get_all_recommendations, init_db
from indicators import compute_indicators, hull_moving_average
from ranker import screen_and_rank_universe
from tracker import (
    export_excel_tracker,
    register_new_recommendations,
    update_active_recommendations,
)
from universe import normalize_ticker


def _make_synthetic_ohlcv(n_days: int = 120, trend: str = "flat", seed: int = 42) -> pd.DataFrame:
    """Create deterministic synthetic daily OHLCV DataFrame for indicator & breakout testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n_days).strftime("%Y-%m-%d")

    base = 100.0
    closes = []
    highs = []
    lows = []
    opens = []
    volumes = []

    for i in range(n_days):
        noise = rng.normal(0, 0.35)
        if trend == "bull_breakout_day_100":
            if i < 100:
                c = base + np.sin(i / 5.0) * 1.5 + noise * 0.2
            elif i == 100:
                # Decisive Day-1 55-day breakout
                c = 105.5
            elif i < 110:
                # Continued rally
                c = 105.5 + (i - 100) * 0.9
            else:
                # Sharp drop below 20-day low to trigger Turtle Exit on day 112+
                c = 95.0
        elif trend == "bear_breakout_day_100":
            if i < 100:
                c = base + np.sin(i / 5.0) * 1.5 + noise * 0.2
            elif i == 100:
                c = 94.5
            else:
                c = 94.5 - (i - 100) * 0.7
        else:
            c = base + noise

        o = c - 0.2 if c >= base else c + 0.2
        h = max(o, c) + 0.6
        l = min(o, c) - 0.6
        vol = 1_800_000 if i == 100 else 1_000_000
        opens.append(round(o, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        closes.append(round(c, 2))
        volumes.append(int(vol))

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )


def test_ticker_normalization() -> None:
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker("bf.b ") == "BF-B"
    assert normalize_ticker("AAPL") == "AAPL"


def test_no_lookahead_bias_and_indicators() -> None:
    df = _make_synthetic_ohlcv(120, trend="bull_breakout_day_100")
    ind = compute_indicators(df)

    # Verify High_20 at index 100 equals max(High[80:100]) and excludes row 100
    expected_h20 = df["high"].iloc[80:100].max()
    actual_h20 = ind["high_20"].iloc[100]
    assert np.isclose(actual_h20, expected_h20), f"Expected {expected_h20}, got {actual_h20}"

    # Verify High_55 at index 100 equals max(High[45:100]) and excludes row 100
    expected_h55 = df["high"].iloc[45:100].max()
    actual_h55 = ind["high_55"].iloc[100]
    assert np.isclose(actual_h55, expected_h55)

    # Verify Hull MA, RSI, MACD, Stochastic, ADX, RVOL are populated and finite at index 100
    row100 = ind.iloc[100]
    for col in ("atr_n", "hma_20", "hma_slope_n", "rsi_14", "macd_hist_n", "stoch_k", "adx_14", "rvol"):
        assert pd.notna(row100[col]), f"Indicator {col} should not be NaN"


def test_closing_breakout_excludes_intraday_wick() -> None:
    df = _make_synthetic_ohlcv(80, trend="flat")
    # Force an intraday high spike on the last bar that closes back inside the 20d channel
    prior_20_high = df["high"].iloc[-21:-1].max()
    df.iloc[-1, df.columns.get_loc("high")] = prior_20_high + 5.0
    df.iloc[-1, df.columns.get_loc("close")] = prior_20_high - 0.5

    ind = compute_indicators(df)
    last_row = ind.iloc[-1]
    assert not bool(last_row["fresh_long_s1"])
    assert not bool(last_row["fresh_long_s2"])


def test_end_to_end_ranking_lifecycle_freeze_excel_and_chart(tmp_path: Path) -> None:
    db_file = tmp_path / "test_screener.db"
    charts_dir = tmp_path / "charts"
    xlsx_file = tmp_path / "tracker.xlsx"
    init_db(db_file)

    df_bull = _make_synthetic_ohlcv(120, trend="bull_breakout_day_100", seed=1)
    df_bear = _make_synthetic_ohlcv(120, trend="bear_breakout_day_100", seed=2)

    ind_frames = {
        "NVDA": compute_indicators(df_bull),
        "TSLA": compute_indicators(df_bear),
    }
    meta = {
        "NVDA": {"company_name": "NVIDIA Corp", "sector": "Technology"},
        "TSLA": {"company_name": "Tesla Inc", "sector": "Consumer Discretionary"},
    }

    breakout_date = str(df_bull.index[100])
    top_longs, top_shorts, eff_date = screen_and_rank_universe(
        indicator_frames=ind_frames,
        ticker_meta=meta,
        scan_date=breakout_date,
    )

    assert eff_date == breakout_date
    assert len(top_longs) == 1
    assert top_longs.iloc[0]["ticker"] == "NVDA"
    assert 0.0 <= float(top_longs.iloc[0]["turtle_score"]) <= 100.0
    assert top_longs.iloc[0]["grade"] in {"A+", "A", "B+", "B", "C"}

    assert len(top_shorts) == 1
    assert top_shorts.iloc[0]["ticker"] == "TSLA"

    # Register recommendations on day 100
    recs = register_new_recommendations(
        top_longs=top_longs,
        top_shorts=top_shorts,
        indicator_frames=ind_frames,
        generate_charts=True,
        db_path=db_file,
        charts_dir=charts_dir,
    )
    assert len(recs) == 2

    # Verify HTML chart exists and contains vertical dotted recommendation date line & price
    nvda_chart = Path(recs[0]["chart_path"])
    assert nvda_chart.exists()
    html_text = nvda_chart.read_text(encoding="utf-8")
    assert breakout_date in html_text
    assert "Recommended Price" in html_text

    # Step forward to day 105 (NVDA still rallying -> ACTIVE)
    day_105 = str(df_bull.index[105])
    updated_105, act_longs, _ = update_active_recommendations(
        indicator_frames=ind_frames,
        as_of_date=day_105,
        generate_charts=True,
        db_path=db_file,
        charts_dir=charts_dir,
    )
    nvda_105 = next(r for r in updated_105 if r["ticker"] == "NVDA")
    assert nvda_105["status"] == "ACTIVE"
    assert nvda_105["return_pct"] > 0.0
    assert "NVDA" in act_longs

    # Step forward to day 115 (NVDA drops to 95.0 below 20d low -> EXITED and FROZEN)
    day_115 = str(df_bull.index[115])
    updated_115, act_longs_115, _ = update_active_recommendations(
        indicator_frames=ind_frames,
        as_of_date=day_115,
        generate_charts=True,
        db_path=db_file,
        charts_dir=charts_dir,
    )
    nvda_115 = next(r for r in updated_115 if r["ticker"] == "NVDA")
    assert nvda_115["status"] == "EXITED"
    assert nvda_115["exit_date"] == str(df_bull.index[110])  # First day it breached 20d low
    assert "NVDA" not in act_longs_115

    # Export Excel tracker and verify tabs + hyperlinks
    export_excel_tracker(top_longs, top_shorts, day_115, xlsx_path=xlsx_file, db_path=db_file)
    assert xlsx_file.exists()

    wb = load_workbook(str(xlsx_file))
    assert set(wb.sheetnames) == {"Tracker_Log", "Latest_Top10", "Grade_Performance"}
    ws_log = wb["Tracker_Log"]
    ticker_cell = ws_log.cell(row=2, column=2)
    assert ticker_cell.hyperlink is not None
    assert ticker_cell.hyperlink.target.endswith(".html")
