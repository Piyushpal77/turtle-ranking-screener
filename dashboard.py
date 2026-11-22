"""Professional Web UI Dashboard Generator for the Turtle Trend Following Stock Screener."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    CHARTS_DIR,
    DB_PATH,
    OUTPUT_DIR,
    PILLAR_WEIGHTS,
    TRACKER_XLSX_PATH,
)
from database import (
    get_all_recommendations,
    get_connection,
    get_date_wise_closing_prices,
    get_ticker_metadata,
)
from ranker import compute_candidate_pillars

logger = logging.getLogger(__name__)

DASHBOARD_HTML_PATH = OUTPUT_DIR / "dashboard.html"


def build_universe_snapshot(
    indicator_frames: Dict[str, pd.DataFrame],
    ticker_meta: Dict[str, Dict[str, str]],
    as_of_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a latest-bar snapshot of all universe stocks including proximity to 20d/55d
    Donchian High/Low channels and full technical indicators.
    """
    rows: List[Dict[str, Any]] = []
    for tkr, df in sorted(indicator_frames.items()):
        if df is None or df.empty:
            continue
        if as_of_date and as_of_date in df.index:
            r = df.loc[as_of_date]
            dt_str = as_of_date
        else:
            r = df.iloc[-1]
            dt_str = str(df.index[-1])

        if isinstance(r, pd.DataFrame):
            r = r.iloc[-1]

        def _f(k: str, default: float = 0.0) -> float:
            v = float(r.get(k, default))
            return default if np.isnan(v) else v

        close_v = _f("close")
        if close_v <= 0:
            continue

        h20 = _f("high_20", close_v)
        l20 = _f("low_20", close_v)
        h55 = _f("high_55", close_v)
        l55 = _f("low_55", close_v)

        pct_to_h20 = ((close_v - h20) / h20 * 100.0) if h20 > 0 else 0.0
        pct_to_h55 = ((close_v - h55) / h55 * 100.0) if h55 > 0 else 0.0
        pct_to_l20 = ((close_v - l20) / l20 * 100.0) if l20 > 0 else 0.0
        pct_to_l55 = ((close_v - l55) / l55 * 100.0) if l55 > 0 else 0.0

        meta = ticker_meta.get(tkr, {})
        signal_tag = "Inside Channel"
        if bool(r.get("fresh_long_s2", False)):
            signal_tag = "Fresh Long S2 (55d)"
        elif bool(r.get("fresh_long_s1", False)):
            signal_tag = "Fresh Long S1 (20d)"
        elif bool(r.get("fresh_short_s2", False)):
            signal_tag = "Fresh Short S2 (55d)"
        elif bool(r.get("fresh_short_s1", False)):
            signal_tag = "Fresh Short S1 (20d)"
        elif bool(r.get("is_long_s2_raw", False)):
            signal_tag = "Cont. Long S2 (55d)"
        elif bool(r.get("is_long_s1_raw", False)):
            signal_tag = "Cont. Long S1 (20d)"
        elif bool(r.get("is_short_s2_raw", False)):
            signal_tag = "Cont. Short S2 (55d)"
        elif bool(r.get("is_short_s1_raw", False)):
            signal_tag = "Cont. Short S1 (20d)"

        rows.append(
            {
                "ticker": tkr,
                "company_name": meta.get("company_name", tkr),
                "sector": meta.get("sector", "Unknown"),
                "source_index": meta.get("source_index", "S&P 500"),
                "date": dt_str,
                "close": round(close_v, 2),
                "high_20": round(h20, 2),
                "low_20": round(l20, 2),
                "high_55": round(h55, 2),
                "low_55": round(l55, 2),
                "pct_to_h20": round(pct_to_h20, 2),
                "pct_to_h55": round(pct_to_h55, 2),
                "pct_to_l20": round(pct_to_l20, 2),
                "pct_to_l55": round(pct_to_l55, 2),
                "signal_tag": signal_tag,
                "atr_n": round(_f("atr_n"), 2),
                "atr_pct": round(_f("atr_pct"), 2),
                "hma_20": round(_f("hma_20"), 2),
                "hma_slope_n": round(_f("hma_slope_n"), 2),
                "rsi_14": round(_f("rsi_14", 50.0), 1),
                "macd_hist_n": round(_f("macd_hist_n"), 3),
                "stoch_k": round(_f("stoch_k", 50.0), 1),
                "stoch_d": round(_f("stoch_d", 50.0), 1),
                "adx_14": round(_f("adx_14", 20.0), 1),
                "rvol": round(_f("rvol", 1.0), 2),
                "roc_60": round(_f("roc_60"), 2),
                "dist_sma200_pct": round(_f("dist_sma200_pct"), 2),
            }
        )
    return rows


def build_dashboard_payload(
    indicator_frames: Optional[Dict[str, pd.DataFrame]] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """Assemble complete JSON payload for the web dashboard UI."""
    all_recs = get_all_recommendations(db_path=db_path)
    ticker_meta = get_ticker_metadata(db_path=db_path)
    date_wise_dates, rec_date_closes = get_date_wise_closing_prices(all_recs, db_path=db_path)

    # Enrich each recommendation with its 5-pillar sub-scores, date-wise closes, and relative chart link
    for rec in all_recs:
        rec["date_wise_closes"] = rec_date_closes.get(str(rec["rec_id"]), {})
        # Provide defaults for pillar computation
        rec_for_pillars = {
            "breakout_n": rec.get("breakout_n", 0.5),
            "vol_compression": 0.9,
            "system": rec.get("system", "System 1 (20d)"),
            "vol_adj_mom": (float(rec.get("roc_60", 0.0)) / max(0.5, float(rec.get("atr_pct", 2.0)))),
            "macd_hist_n": rec.get("macd_hist_n", 0.0),
            "macd_accel_n": float(rec.get("macd_hist_n", 0.0)) * 0.5,
            "hma_slope_n": rec.get("hma_slope_n", 0.0),
            "price_to_hma_n": (
                (float(rec.get("rec_price", 0.0)) - float(rec.get("hma_20", 0.0)))
                / max(0.01, float(rec.get("atr_n", 1.0)))
            ),
            "dist_sma200_pct": rec.get("dist_sma200_pct", 0.0),
            "rsi_14": rec.get("rsi_14", 50.0),
            "stoch_k": rec.get("stoch_k", 50.0),
            "stoch_spread": float(rec.get("stoch_k", 50.0)) - float(rec.get("stoch_d", 50.0)),
            "rvol": rec.get("rvol", 1.0),
            "adx_14": rec.get("adx_14", 20.0),
            "di_spread": 10.0 if rec.get("direction") == "LONG" else -10.0,
        }
        pillars = compute_candidate_pillars(rec_for_pillars, str(rec.get("direction", "LONG")))
        rec.update(pillars)
        rec["chart_rel_url"] = f"charts/{rec['rec_id']}.html"

    rec_dates = sorted({str(r["rec_date"]) for r in all_recs}, reverse=True)
    latest_scan_date = rec_dates[0] if rec_dates else ""

    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT COUNT(DISTINCT ticker) AS cnt, MAX(date) AS max_dt FROM daily_prices;")
        row = cur.fetchone()
        db_ticker_count = int(row["cnt"] or 0) if row else 0
        db_max_date = str(row["max_dt"] or latest_scan_date) if row else latest_scan_date

    universe_snapshot: List[Dict[str, Any]] = []
    if indicator_frames:
        universe_snapshot = build_universe_snapshot(indicator_frames, ticker_meta)

    active_recs = [r for r in all_recs if r.get("status") == "ACTIVE"]
    exited_recs = [r for r in all_recs if r.get("status") == "EXITED"]
    wins = [r for r in all_recs if float(r.get("return_pct", 0.0)) > 0.0]
    win_rate = (len(wins) / len(all_recs) * 100.0) if all_recs else 0.0
    avg_return = float(np.mean([float(r.get("return_pct", 0.0)) for r in all_recs])) if all_recs else 0.0
    avg_mfe = float(np.mean([float(r.get("max_favorable_pct", 0.0)) for r in all_recs])) if all_recs else 0.0
    avg_score = float(np.mean([float(r.get("turtle_score", 0.0)) for r in all_recs])) if all_recs else 0.0

    best_pick = max(all_recs, key=lambda x: float(x.get("return_pct", -999.0))) if all_recs else None

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_market_date": db_max_date,
        "latest_scan_date": latest_scan_date,
        "scan_dates": rec_dates,
        "universe_count": db_ticker_count or len(ticker_meta),
        "kpis": {
            "total_tracked": len(all_recs),
            "active_count": len(active_recs),
            "exited_count": len(exited_recs),
            "win_rate_pct": round(win_rate, 1),
            "avg_return_pct": round(avg_return, 2),
            "avg_mfe_pct": round(avg_mfe, 2),
            "avg_turtle_score": round(avg_score, 1),
            "best_pick": {
                "ticker": best_pick["ticker"],
                "direction": best_pick["direction"],
                "rec_date": best_pick["rec_date"],
                "return_pct": best_pick["return_pct"],
                "grade": best_pick["grade"],
            }
            if best_pick
            else None,
        },
        "pillar_weights": PILLAR_WEIGHTS,
        "date_wise_dates": date_wise_dates,
        "recommendations": all_recs,
        "universe_snapshot": universe_snapshot,
    }


def generate_dashboard_html(
    indicator_frames: Optional[Dict[str, pd.DataFrame]] = None,
    db_path: Path = DB_PATH,
    output_path: Path = DASHBOARD_HTML_PATH,
) -> Path:
    """Compile and write the standalone & server-compatible HTML dashboard."""
    payload = build_dashboard_payload(indicator_frames=indicator_frames, db_path=db_path)
    payload_json = json.dumps(payload, ensure_ascii=False)

    html = _DASHBOARD_TEMPLATE.replace("__DASHBOARD_PAYLOAD_JSON__", payload_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Generated professional web UI dashboard at %s", output_path)
    return output_path


_DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Turtle Trend Following — Institutional US Stock Screener & Tracker</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg-primary: #0b0e14;
      --bg-secondary: #111622;
      --bg-card: #161d2b;
      --bg-card-hover: #1c2536;
      --bg-elevated: #1f293d;
      --border: #263248;
      --border-light: #31415e;
      --text-primary: #f0f6fc;
      --text-secondary: #9fb0c7;
      --text-muted: #687a94;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-green-bg: rgba(16, 185, 129, 0.14);
      --accent-red: #f43f5e;
      --accent-red-bg: rgba(244, 63, 94, 0.14);
      --accent-amber: #f59e0b;
      --accent-amber-bg: rgba(245, 158, 11, 0.14);
      --accent-purple: #8b5cf6;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-primary);
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-size: 13px;
      line-height: 1.45;
      overflow-x: hidden;
    }

    /* Top Header Bar */
    .topbar {
      background: linear-gradient(180deg, #141b29 0%, #0f141f 100%);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 14px;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(12px);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .brand-logo {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: linear-gradient(135deg, #10b981 0%, #0284c7 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 18px;
      color: #fff;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.28);
    }

    .brand-title h1 {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(59, 130, 246, 0.16);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .brand-subtitle {
      font-size: 12px;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .control-group {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 5px 10px;
      border-radius: 8px;
    }

    .control-group label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      font-weight: 600;
    }

    select, input[type="text"], input[type="number"] {
      background: var(--bg-secondary);
      color: var(--text-primary);
      border: 1px solid var(--border-light);
      border-radius: 6px;
      padding: 5px 9px;
      font-size: 12px;
      font-family: var(--font-sans);
      outline: none;
    }

    select:focus, input:focus {
      border-color: var(--accent-blue);
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 7px 13px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border-light);
      background: var(--bg-card);
      color: var(--text-primary);
      text-decoration: none;
      transition: all 0.15s ease;
    }

    .btn:hover {
      background: var(--bg-elevated);
      border-color: var(--accent-blue);
    }

    .btn-primary {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      border-color: #3b82f6;
      color: #ffffff;
    }

    .btn-primary:hover {
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }

    .btn-success {
      background: linear-gradient(135deg, #059669 0%, #047857 100%);
      border-color: #10b981;
      color: #ffffff;
    }

    .btn-success:hover {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }

    /* Main Container */
    .container {
      max-width: 1780px;
      margin: 0 auto;
      padding: 18px 24px 48px;
    }

    /* KPI Strip */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 13px 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      overflow: hidden;
    }

    .kpi-card::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 3px;
      height: 100%;
      background: var(--accent-blue);
    }

    .kpi-card.kpi-green::before { background: var(--accent-green); }
    .kpi-card.kpi-amber::before { background: var(--accent-amber); }
    .kpi-card.kpi-purple::before { background: var(--accent-purple); }
    .kpi-card.kpi-cyan::before { background: var(--accent-cyan); }

    .kpi-label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
    }

    .kpi-value {
      font-size: 22px;
      font-weight: 800;
      font-family: var(--font-mono);
      margin: 5px 0 3px;
      color: var(--text-primary);
    }

    .kpi-sub {
      font-size: 11px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Navigation Tabs */
    .nav-tabs {
      display: flex;
      align-items: center;
      gap: 6px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 18px;
      overflow-x: auto;
      padding-bottom: 1px;
    }

    .tab-btn {
      background: transparent;
      color: var(--text-secondary);
      border: none;
      border-bottom: 2px solid transparent;
      padding: 10px 18px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      transition: all 0.15s ease;
    }

    .tab-btn:hover {
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.02);
    }

    .tab-btn.active {
      color: #60a5fa;
      border-bottom-color: #3b82f6;
      background: rgba(59, 130, 246, 0.08);
    }

    .tab-count {
      font-size: 11px;
      font-family: var(--font-mono);
      padding: 1px 7px;
      border-radius: 99px;
      background: var(--bg-elevated);
      color: var(--text-primary);
    }

    .tab-pane {
      display: none;
    }

    .tab-pane.active {
      display: block;
    }

    /* Section Panels */
    .panel {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 20px;
      overflow: hidden;
    }

    .panel-header {
      padding: 13px 18px;
      background: rgba(255, 255, 255, 0.015);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }

    .panel-title {
      font-size: 14px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .dir-pill {
      display: inline-flex;
      align-items: center;
      padding: 2px 9px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }

    .dir-long {
      background: var(--accent-green-bg);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .dir-short {
      background: var(--accent-red-bg);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.35);
    }

    /* Grade Badges */
    .grade-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 32px;
      padding: 2px 7px;
      border-radius: 6px;
      font-weight: 800;
      font-family: var(--font-mono);
      font-size: 12px;
    }

    .grade-Aplus { background: rgba(16, 185, 129, 0.22); color: #34d399; border: 1px solid #10b981; }
    .grade-A { background: rgba(16, 185, 129, 0.14); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.45); }
    .grade-Bplus { background: rgba(59, 130, 246, 0.18); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.45); }
    .grade-B { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.45); }
    .grade-C { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.4); }

    /* Status Badges */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
    }

    .status-active {
      background: rgba(6, 182, 212, 0.16);
      color: #22d3ee;
      border: 1px solid rgba(6, 182, 212, 0.4);
    }

    .status-exited {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .sys-badge {
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 5px;
      font-weight: 600;
      white-space: nowrap;
    }

    .sys-2 {
      background: rgba(139, 92, 246, 0.16);
      color: #c4b5fd;
      border: 1px solid rgba(139, 92, 246, 0.35);
    }

    .sys-1 {
      background: rgba(148, 163, 184, 0.12);
      color: #cbd5e1;
      border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Tables */
    .table-wrap {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }

    th {
      background: #121824;
      color: var(--text-secondary);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 10px 11px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }

    th:hover {
      color: var(--text-primary);
      background: #172030;
    }

    td {
      padding: 9px 11px;
      border-bottom: 1px solid rgba(38, 50, 72, 0.65);
      white-space: nowrap;
      font-size: 12.5px;
    }

    tr:hover td {
      background: var(--bg-card-hover);
    }

    /* Built-in Spreadsheet Grid Styling */
    .spreadsheet-grid th {
      border: 1px solid #2b3952;
      background: #141c2b;
      color: #dbeafe;
      font-size: 11.5px;
      padding: 10px 14px;
    }

    .spreadsheet-grid td {
      border: 1px solid rgba(49, 65, 94, 0.75);
      padding: 10px 14px;
      font-size: 13px;
    }

    .spreadsheet-grid tbody tr:nth-child(even) td {
      background: rgba(255, 255, 255, 0.015);
    }

    .spreadsheet-grid tbody tr:hover td {
      background: rgba(59, 130, 246, 0.12);
    }

    .mono {
      font-family: var(--font-mono);
    }

    .num-pos { color: #34d399; font-weight: 600; }
    .num-neg { color: #fb7185; font-weight: 600; }
    .num-neutral { color: var(--text-secondary); }

    .ticker-link {
      font-family: var(--font-mono);
      font-weight: 700;
      color: #60a5fa;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 6px;
      border-radius: 5px;
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.25);
    }

    .ticker-link:hover {
      background: rgba(59, 130, 246, 0.24);
      color: #93c5fd;
    }

    /* 5-Pillar Mini Bars */
    .pillar-bars {
      display: flex;
      align-items: center;
      gap: 3px;
      height: 16px;
    }

    .pillar-seg {
      width: 11px;
      height: 14px;
      border-radius: 2px;
      background: var(--bg-secondary);
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
    }

    .pillar-fill {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      background: var(--accent-blue);
    }

    /* Filter Bar */
    .filter-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      padding: 12px 18px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
    }

    /* Split Terminal Layout for Tab 3 */
    .terminal-grid {
      display: grid;
      grid-template-columns: 350px 1fr;
      gap: 16px;
      height: calc(100vh - 230px);
      min-height: 760px;
    }

    @media (max-width: 1100px) {
      .terminal-grid {
        grid-template-columns: 1fr;
        height: auto;
      }
    }

    .terminal-list {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .terminal-list-items {
      overflow-y: auto;
      flex: 1;
      divide-y: 1px solid var(--border);
    }

    .term-item {
      padding: 11px 14px;
      border-bottom: 1px solid rgba(38, 50, 72, 0.6);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: background 0.12s;
    }

    .term-item:hover {
      background: var(--bg-card-hover);
    }

    .term-item.active {
      background: rgba(59, 130, 246, 0.15);
      border-left: 3px solid var(--accent-blue);
    }

    .terminal-viewer {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .chart-iframe {
      width: 100%;
      flex: 1;
      border: none;
      background: #0E1117;
      min-height: 680px;
    }

    /* Modal Chart Drawer */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(5, 8, 14, 0.82);
      backdrop-filter: blur(6px);
      z-index: 200;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
    }

    .modal-backdrop.open {
      display: flex;
    }

    .modal-window {
      width: 96vw;
      max-width: 1650px;
      height: 92vh;
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.65);
    }

    .modal-header {
      padding: 12px 20px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .pillar-breakdown-strip {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      padding: 10px 20px;
      background: #101520;
      border-bottom: 1px solid var(--border);
    }

    .pillar-box {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 11px;
    }

    .pillar-box-header {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-secondary);
      margin-bottom: 4px;
      font-weight: 600;
    }

    .progress-track {
      width: 100%;
      height: 6px;
      background: var(--bg-secondary);
      border-radius: 99px;
      overflow: hidden;
    }

    .progress-bar {
      height: 100%;
      border-radius: 99px;
      background: linear-gradient(90deg, #3b82f6, #10b981);
    }

    /* Analytics Grid */
    .analytics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 18px;
      margin-bottom: 20px;
    }

    .chart-box {
      height: 360px;
      width: 100%;
    }

    /* Methodology Grid */
    .method-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 16px;
      padding: 18px;
    }

    .method-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }

    .method-card h3 {
      font-size: 14px;
      margin-bottom: 8px;
      color: #60a5fa;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .method-card p, .method-card li {
      font-size: 12.5px;
      color: var(--text-secondary);
      margin-bottom: 6px;
    }

    .method-card ul {
      padding-left: 18px;
    }

    .toast {
      position: fixed;
      bottom: 22px;
      right: 22px;
      background: var(--bg-elevated);
      border: 1px solid var(--accent-blue);
      color: var(--text-primary);
      padding: 12px 18px;
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      z-index: 300;
      display: none;
      font-weight: 600;
    }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header class="topbar">
    <div class="brand">
      <div class="brand-logo">TT</div>
      <div class="brand-title">
        <h1>
          Turtle Trend Following Screener
          <span class="brand-badge">S&amp;P 500 + Nasdaq 100</span>
        </h1>
        <div class="brand-subtitle" id="headerMetaSubtitle">
          5-Pillar Multi-Indicator Grading (Breakout/ATR • Momentum/MACD • Hull MA/SMA200 • RSI/Stochastic • RVOL/ADX)
        </div>
      </div>
    </div>

    <div class="toolbar">
      <div class="control-group">
        <label for="scanDateSelect">Daily Top 10 Date:</label>
        <select id="scanDateSelect"></select>
      </div>
      <button class="btn btn-primary" id="btnRunLiveScan" onclick="triggerLiveScan()" title="Sync latest EOD prices & run daily screener">
        <span>⟳</span> Sync EOD &amp; Run Scan
      </button>
    </div>
  </header>

  <main class="container">
    <!-- KPI Strip -->
    <section class="kpi-grid" id="kpiStrip"></section>

    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('tab-daily', this)">
        <span>🎯</span> Daily Top 10 Breakouts
        <span class="tab-count" id="badgeDailyCount">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('tab-sheet-long', this)">
        <span>📗</span> Long Trades Spreadsheet
        <span class="tab-count" id="badgeLongSheetCount">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('tab-sheet-short', this)">
        <span>📕</span> Short Trades Spreadsheet
        <span class="tab-count" id="badgeShortSheetCount">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('tab-tracker', this)">
        <span>📋</span> All Indicators Log
        <span class="tab-count" id="badgeTrackerCount">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('tab-terminal', this)">
        <span>📈</span> Interactive Chart Terminal
      </button>
      <button class="tab-btn" onclick="switchTab('tab-analytics', this)">
        <span>📊</span> Grade &amp; System Analytics
      </button>
      <button class="tab-btn" onclick="switchTab('tab-universe', this)">
        <span>🌐</span> Full Universe Watchlist
        <span class="tab-count" id="badgeUniverseCount">0</span>
      </button>
      <button class="tab-btn" onclick="switchTab('tab-methodology', this)">
        <span>📐</span> 5-Pillar Grading Rules
      </button>
    </nav>

    <!-- TAB 1: LONG TRADES SPREADSHEET (IN-SYSTEM SPREADSHEET) -->
    <section id="tab-sheet-long" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span class="dir-pill dir-long">▲ LONG TRADES SPREADSHEET</span>
            <span>Ranked in Decreasing Order of Turtle Score</span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">
            Click any <b>Security Name</b> to open its interactive daily chart (with recommended price &amp; vertical dotted recommendation date line)
          </div>
        </div>
        <div class="table-wrap">
          <table id="tableLongSheet" class="spreadsheet-grid">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 2: SHORT TRADES SPREADSHEET (IN-SYSTEM SPREADSHEET) -->
    <section id="tab-sheet-short" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span class="dir-pill dir-short">▼ SHORT TRADES SPREADSHEET</span>
            <span>Ranked in Decreasing Order of Turtle Score</span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">
            Click any <b>Security Name</b> to open its interactive daily chart (with recommended price &amp; vertical dotted recommendation date line)
          </div>
        </div>
        <div class="table-wrap">
          <table id="tableShortSheet" class="spreadsheet-grid">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: DAILY TOP 10 BREAKOUTS -->
    <section id="tab-daily" class="tab-pane active">
      <!-- Top 10 Longs Panel -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span class="dir-pill dir-long">▲ TOP 10 LONG</span>
            <span>Fresh Day-1 Donchian Breakout Candidates</span>
            <span style="color:var(--text-muted); font-size:12px; font-weight:500;" id="dailyLongSubtitle"></span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">
            Click any <b>Ticker</b> to open its interactive daily candlestick chart with Recommendation Date vertical dotted line &amp; entry level
          </div>
        </div>
        <div class="table-wrap">
          <table id="tableDailyLongs">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <!-- Top 10 Shorts Panel -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span class="dir-pill dir-short">▼ TOP 10 SHORT</span>
            <span>Fresh Day-1 Donchian Breakdown Candidates</span>
            <span style="color:var(--text-muted); font-size:12px; font-weight:500;" id="dailyShortSubtitle"></span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">
            System 2 (55d) takes priority over System 1 (20d); active tickers are deduplicated
          </div>
        </div>
        <div class="table-wrap">
          <table id="tableDailyShorts">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 4: MASTER RECOMMENDATION TRACKER -->
    <section id="tab-tracker" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span>📋 Master Recommendation Lifecycle Tracker</span>
            <span style="color:var(--text-muted); font-size:12px; font-weight:500;">
              Updates EOD daily while ACTIVE • Freezes Return % &amp; Chart once Close breaches 10d/20d Turtle Exit Channel
            </span>
          </div>
        </div>

        <div class="filter-bar">
          <div class="control-group">
            <label>Search:</label>
            <input type="text" id="filterTrackerSearch" placeholder="Ticker, Company, Sector..." oninput="renderTrackerTable()" />
          </div>
          <div class="control-group">
            <label>Status:</label>
            <select id="filterTrackerStatus" onchange="renderTrackerTable()">
              <option value="ALL">All Statuses</option>
              <option value="ACTIVE">ACTIVE (Live EOD)</option>
              <option value="EXITED">EXITED (Frozen at Turtle Exit)</option>
            </select>
          </div>
          <div class="control-group">
            <label>Direction:</label>
            <select id="filterTrackerDir" onchange="renderTrackerTable()">
              <option value="ALL">LONG &amp; SHORT</option>
              <option value="LONG">LONG Only</option>
              <option value="SHORT">SHORT Only</option>
            </select>
          </div>
          <div class="control-group">
            <label>Grade:</label>
            <select id="filterTrackerGrade" onchange="renderTrackerTable()">
              <option value="ALL">All Grades (A+ to C)</option>
              <option value="A+">Grade A+ (85–100)</option>
              <option value="A">Grade A (75–84.9)</option>
              <option value="B+">Grade B+ (65–74.9)</option>
              <option value="B">Grade B (55–64.9)</option>
              <option value="C">Grade C (&lt;55)</option>
            </select>
          </div>
          <div class="control-group">
            <label>System:</label>
            <select id="filterTrackerSystem" onchange="renderTrackerTable()">
              <option value="ALL">System 1 &amp; System 2</option>
              <option value="System 2 (55d)">System 2 (55d Breakout)</option>
              <option value="System 1 (20d)">System 1 (20d Breakout)</option>
            </select>
          </div>
          <div class="control-group">
            <label>Rec Date:</label>
            <select id="filterTrackerDate" onchange="renderTrackerTable()">
              <option value="ALL">All Dates</option>
            </select>
          </div>
        </div>

        <div class="table-wrap">
          <table id="tableTracker">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 3: INTERACTIVE CHART TERMINAL -->
    <section id="tab-terminal" class="tab-pane">
      <div class="terminal-grid">
        <div class="terminal-list">
          <div class="panel-header">
            <div class="panel-title">Tracked Recommendations</div>
          </div>
          <div style="padding:10px; border-bottom:1px solid var(--border); display:flex; gap:6px;">
            <input type="text" id="termSearch" placeholder="Filter ticker or date..." style="width:100%;" oninput="renderTerminalList()" />
          </div>
          <div class="terminal-list-items" id="terminalSidebarList"></div>
        </div>

        <div class="terminal-viewer">
          <div class="panel-header" id="terminalChartHeader">
            <div class="panel-title">Select a recommendation to inspect its daily chart</div>
            <div>
              <a class="btn" id="terminalOpenNewTab" href="#" target="_blank">↗ Open Full Chart in New Tab</a>
            </div>
          </div>
          <div class="pillar-breakdown-strip" id="terminalPillarStrip"></div>
          <iframe class="chart-iframe" id="terminalChartFrame" src="about:blank"></iframe>
        </div>
      </div>
    </section>

    <!-- TAB 4: GRADE & SYSTEM ANALYTICS -->
    <section id="tab-analytics" class="tab-pane">
      <div class="analytics-grid">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Win Rate &amp; Average Return by Letter Grade</div>
          </div>
          <div id="chartGradePerf" class="chart-box"></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Return % vs. Composite Turtle Score (All Tracked Picks)</div>
          </div>
          <div id="chartScoreScatter" class="chart-box"></div>
        </div>
      </div>

      <div class="analytics-grid">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Breakout Sector Distribution</div>
          </div>
          <div id="chartSectorDist" class="chart-box"></div>
        </div>
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">Cohort Performance Summary Table</div>
          </div>
          <div class="table-wrap">
            <table id="tableCohortPerf">
              <thead>
                <tr>
                  <th>Cohort / Segment</th>
                  <th>Total Picks</th>
                  <th>Active</th>
                  <th>Exited (Frozen)</th>
                  <th>Win Rate %</th>
                  <th>Avg Return %</th>
                  <th>Max Gain (Avg MFE %)</th>
                  <th>Avg Score</th>
                  <th>Avg Days Held</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 5: FULL UNIVERSE & NEAR-BREAKOUT PROXIMITY WATCHLIST -->
    <section id="tab-universe" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <span>🌐 Full S&amp;P 500 + Nasdaq 100 Universe &amp; Donchian Proximity Scanner</span>
          </div>
          <div style="font-size:12px; color:var(--text-secondary);">
            Monitor all 503+ liquid US stocks, their current indicator readings, and how close they are to triggering a 20d/55d breakout
          </div>
        </div>

        <div class="filter-bar">
          <div class="control-group">
            <label>Search:</label>
            <input type="text" id="filterUniSearch" placeholder="Ticker, Company, Sector..." oninput="renderUniverseTable()" />
          </div>
          <div class="control-group">
            <label>Channel State:</label>
            <select id="filterUniSignal" onchange="renderUniverseTable()">
              <option value="ALL">All Universe Stocks</option>
              <option value="FRESH">Fresh Day-1 Breakouts Only</option>
              <option value="NEAR_HIGH">Near 20d/55d High (Within 1.5%)</option>
              <option value="NEAR_LOW">Near 20d/55d Low (Within 1.5%)</option>
            </select>
          </div>
          <div class="control-group">
            <label>Sector:</label>
            <select id="filterUniSector" onchange="renderUniverseTable()">
              <option value="ALL">All GICS Sectors</option>
            </select>
          </div>
        </div>

        <div class="table-wrap">
          <table id="tableUniverse">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- TAB 6: METHODOLOGY & 5-PILLAR GRADING -->
    <section id="tab-methodology" class="tab-pane">
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">📐 Turtle Trend Following &amp; 5-Pillar Multi-Indicator Grading Architecture</div>
        </div>
        <div class="method-grid">
          <div class="method-card">
            <h3><span>Pillar 1: Breakout &amp; ATR Structure</span> <span class="brand-badge">25% Weight</span></h3>
            <ul>
              <li><b>ATR-Normalized Breakout Impulse (<code>Breakout_N</code>):</b> Measures <code>|Close - Donchian_Level| / N</code>. Rewards decisive <code>0.35N – 1.50N</code> closes beyond the channel while tapering above <code>2.5N</code> to avoid chasing overextended overnight earnings gaps.</li>
              <li><b>Base Volatility Compression (<code>N20 / SMA60(TR)</code>):</b> Rewards breakouts emerging from low-volatility accumulation bases.</li>
              <li><b>System 2 Priority Bonus:</b> 55-day Donchian breakouts receive priority and structural weighting over 20-day breakouts.</li>
            </ul>
          </div>
          <div class="method-card">
            <h3><span>Pillar 2: Multi-Horizon Momentum &amp; MACD</span> <span class="brand-badge">25% Weight</span></h3>
            <ul>
              <li><b>Volatility-Adjusted 60d Momentum:</b> <code>±ROC_60 / ATR%</code> measures intermediate risk-adjusted trend velocity.</li>
              <li><b>N-Normalized MACD (12, 26, 9) Histogram:</b> <code>±MACD_Hist / N</code> plus 3-day histogram acceleration confirms fresh momentum inflection on the breakout bar.</li>
            </ul>
          </div>
          <div class="method-card">
            <h3><span>Pillar 3: Hull MA (20) &amp; Macro Regime</span> <span class="brand-badge">20% Weight</span></h3>
            <ul>
              <li><b>Hull Moving Average (<code>HMA_20</code>):</b> Zero-lag weighted moving average curve <code>WMA(√20, 2·WMA(10) - WMA(20))</code>.</li>
              <li><b>5-Day HMA N-Slope:</b> <code>±(HMA_20[t] - HMA_20[t-5]) / N</code> confirms fast structural trend slope.</li>
              <li><b>200-Day SMA Alignment:</b> Rewards breakouts aligned with the long-term institutional trend regime.</li>
            </ul>
          </div>
          <div class="method-card">
            <h3><span>Pillar 4: Oscillator Power-Zone (RSI &amp; Stoch)</span> <span class="brand-badge">15% Weight</span></h3>
            <ul>
              <li><b>Trend-Following RSI(14) Curve:</b> Unlike mean-reversion screens, successful Day-1 Turtle breakouts occur in the <b>bullish power zone (RSI 56–76 Long / 24–44 Short)</b>. Scores peak at 67 (Long) / 33 (Short) and penalize extreme blow-off exhaustion (<code>&gt;82</code> Long / <code>&lt;18</code> Short).</li>
              <li><b>Slow Stochastic (14, 3, 3):</b> Combines directional <code>%K</code> positioning and positive <code>%K - %D</code> momentum spread.</li>
            </ul>
          </div>
          <div class="method-card">
            <h3><span>Pillar 5: Institutional Volume &amp; ADX</span> <span class="brand-badge">15% Weight</span></h3>
            <ul>
              <li><b>Relative Volume Surge (<code>RVOL</code>):</b> <code>Volume[t] / SMA_20(Volume[t-1])</code> verifies institutional participation on breakout day.</li>
              <li><b>Wilder's ADX(14) &amp; DI Spread:</b> Combines <code>ADX_14</code> trend strength with directional dominance <code>±(+DI_14 - -DI_14)</code>.</li>
            </ul>
          </div>
          <div class="method-card">
            <h3><span>Lifecycle Tracking &amp; Freeze Rules</span> <span class="brand-badge">Zero Look-Ahead</span></h3>
            <ul>
              <li><b>Fresh Day-1 Trigger:</b> Qualifies only on Day 1 when <code>Close</code> crosses the prior 20d/55d Donchian channel.</li>
              <li><b>Daily Chart Annotations:</b> Every chart displays a <b>vertical dotted line</b> on the recommendation date and a <b>horizontal dashed line</b> at the recommended price.</li>
              <li><b>Freeze on Turtle Exit:</b> Updates daily until <code>Close</code> breaches the System-matched opposite channel (<b>10-day Low/High</b> for System 1; <b>20-day Low/High</b> for System 2), at which point the row and chart freeze as <code>EXITED</code>.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  </main>

  <!-- Chart Modal Drawer -->
  <div class="modal-backdrop" id="chartModal" onclick="if(event.target===this) closeChartModal()">
    <div class="modal-window">
      <div class="modal-header">
        <div id="modalHeaderTitle" class="panel-title"></div>
        <div style="display:flex; align-items:center; gap:8px;">
          <a class="btn" id="modalOpenNewTab" href="#" target="_blank">↗ Open in Full Tab</a>
          <button class="btn" onclick="closeChartModal()">✕ Close (Esc)</button>
        </div>
      </div>
      <div class="pillar-breakdown-strip" id="modalPillarStrip"></div>
      <iframe class="chart-iframe" id="modalChartFrame" src="about:blank"></iframe>
    </div>
  </div>

  <div class="toast" id="toastMsg"></div>

  <script>
    const DATA = __DASHBOARD_PAYLOAD_JSON__;
    let currentScanDate = DATA.latest_scan_date || "";
    let trackerSortCol = "rec_date";
    let trackerSortAsc = false;
    let uniSortCol = "pct_to_h20";
    let uniSortAsc = false;
    let activeTerminalRecId = null;

    function showToast(msg, duration = 4000) {
      const el = document.getElementById("toastMsg");
      el.textContent = msg;
      el.style.display = "block";
      setTimeout(() => { el.style.display = "none"; }, duration);
    }

    function gradeBadgeHTML(grade) {
      const g = String(grade || "C");
      const cls = g === "A+" ? "grade-Aplus" : g === "A" ? "grade-A" : g === "B+" ? "grade-Bplus" : g === "B" ? "grade-B" : "grade-C";
      return `<span class="grade-badge ${cls}">${g}</span>`;
    }

    function dirBadgeHTML(dir) {
      return dir === "LONG"
        ? `<span class="dir-pill dir-long">▲ LONG</span>`
        : `<span class="dir-pill dir-short">▼ SHORT</span>`;
    }

    function sysBadgeHTML(sys) {
      const is2 = String(sys).includes("System 2");
      return `<span class="sys-badge ${is2 ? 'sys-2' : 'sys-1'}">${sys}</span>`;
    }

    function statusBadgeHTML(status, exitDate) {
      if (status === "ACTIVE") {
        return `<span class="status-badge status-active">● ACTIVE</span>`;
      }
      return `<span class="status-badge status-exited" title="Frozen on ${exitDate}">🔒 EXITED (${exitDate || ''})</span>`;
    }

    function fmtNum(val, decimals = 2, signed = false) {
      if (val === null || val === undefined || isNaN(val)) return "—";
      const n = Number(val);
      const s = n.toFixed(decimals);
      if (signed && n > 0) return `+${s}`;
      return s;
    }

    function fmtPctHTML(val) {
      if (val === null || val === undefined || isNaN(val)) return `<span class="mono num-neutral">—</span>`;
      const n = Number(val);
      const cls = n > 0 ? "num-pos" : n < 0 ? "num-neg" : "num-neutral";
      const sign = n > 0 ? "+" : "";
      return `<span class="mono ${cls}">${sign}${n.toFixed(2)}%</span>`;
    }

    function pillarMiniBarsHTML(rec) {
      const pillars = [
        { label: "Breakout & ATR (25%)", val: rec.p1_breakout_atr || 70, color: "#3b82f6" },
        { label: "Momentum & MACD (25%)", val: rec.p2_momentum_macd || 70, color: "#10b981" },
        { label: "Hull MA & SMA200 (20%)", val: rec.p3_hma_trend || 70, color: "#06b6d4" },
        { label: "RSI & Stoch Zone (15%)", val: rec.p4_oscillator_zone || 70, color: "#8b5cf6" },
        { label: "RVOL & ADX (15%)", val: rec.p5_volume_adx || 70, color: "#f59e0b" },
      ];
      const tooltip = pillars.map(p => `${p.label}: ${Number(p.val).toFixed(1)}`).join(" | ");
      const bars = pillars.map(p => {
        const h = Math.max(12, Math.min(100, Number(p.val)));
        return `<div class="pillar-seg"><div class="pillar-fill" style="height:${h}%; background:${p.color};"></div></div>`;
      }).join("");
      return `<div class="pillar-bars" title="${tooltip}">${bars}</div>`;
    }

    function pillarStripHTML(rec) {
      const items = [
        { name: "1. Breakout & ATR (25%)", val: rec.p1_breakout_atr || 0, sub: `Brk_N: ${fmtNum(rec.breakout_n)}N | ATR: ${fmtNum(rec.atr_pct)}%` },
        { name: "2. Momentum & MACD (25%)", val: rec.p2_momentum_macd || 0, sub: `60d ROC: ${fmtNum(rec.roc_60, 1, true)}% | MACD_N: ${fmtNum(rec.macd_hist_n, 3, true)}` },
        { name: "3. Hull MA & Trend (20%)", val: rec.p3_hma_trend || 0, sub: `HMA20 Slp: ${fmtNum(rec.hma_slope_n, 2, true)}N | SMA200: ${fmtNum(rec.dist_sma200_pct, 1, true)}%` },
        { name: "4. RSI & Stochastic (15%)", val: rec.p4_oscillator_zone || 0, sub: `RSI(14): ${fmtNum(rec.rsi_14, 1)} | Stoch %K: ${fmtNum(rec.stoch_k, 1)}` },
        { name: "5. Volume & ADX (15%)", val: rec.p5_volume_adx || 0, sub: `RVOL: ${fmtNum(rec.rvol, 2)}x | ADX(14): ${fmtNum(rec.adx_14, 1)}` },
      ];
      return items.map(it => `
        <div class="pillar-box">
          <div class="pillar-box-header">
            <span>${it.name}</span>
            <span class="mono" style="color:var(--text-primary);">${Number(it.val).toFixed(1)}/100</span>
          </div>
          <div class="progress-track">
            <div class="progress-bar" style="width:${Math.min(100, Math.max(5, Number(it.val)))}%;"></div>
          </div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:4px;" class="mono">${it.sub}</div>
        </div>
      `).join("");
    }

    function initDashboard() {
      document.getElementById("headerMetaSubtitle").textContent =
        `Latest EOD: ${DATA.latest_market_date} • Universe: ${DATA.universe_count} Stocks • Last Updated: ${DATA.generated_at}`;

      // Populate Scan Date selector
      const dateSel = document.getElementById("scanDateSelect");
      const trackerDateSel = document.getElementById("filterTrackerDate");
      dateSel.innerHTML = "";
      (DATA.scan_dates || []).forEach(dt => {
        const opt = document.createElement("option");
        opt.value = dt;
        opt.textContent = dt === DATA.latest_scan_date ? `${dt} (Latest)` : dt;
        dateSel.appendChild(opt);

        const opt2 = document.createElement("option");
        opt2.value = dt;
        opt2.textContent = dt;
        trackerDateSel.appendChild(opt2);
      });
      dateSel.addEventListener("change", (e) => {
        currentScanDate = e.target.value;
        renderDailyTables();
      });

      // Populate Universe Sector filter
      const sectors = Array.from(new Set((DATA.universe_snapshot || []).map(r => r.sector))).sort();
      const secSel = document.getElementById("filterUniSector");
      sectors.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        secSel.appendChild(opt);
      });

      renderKPIs();
      renderDailyTables();
      renderRecSpreadsheetTab();
      renderTrackerTable();
      renderTerminalList();
      renderAnalytics();
      renderUniverseTable();

      if (DATA.recommendations && DATA.recommendations.length > 0) {
        selectTerminalRec(DATA.recommendations[0].rec_id);
      }
    }

    function renderKPIs() {
      const k = DATA.kpis || {};
      const best = k.best_pick;
      const bestHTML = best
        ? `${best.ticker} (${best.direction}) ${fmtPctHTML(best.return_pct)}`
        : "—";

      document.getElementById("kpiStrip").innerHTML = `
        <div class="kpi-card kpi-cyan">
          <div class="kpi-label">Universe &amp; Latest EOD</div>
          <div class="kpi-value">${DATA.universe_count}</div>
          <div class="kpi-sub">S&amp;P 500 + Nasdaq 100 • EOD ${DATA.latest_market_date}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Total Tracked Picks</div>
          <div class="kpi-value">${k.total_tracked || 0}</div>
          <div class="kpi-sub">${k.active_count || 0} ACTIVE • ${k.exited_count || 0} EXITED (Frozen)</div>
        </div>
        <div class="kpi-card kpi-green">
          <div class="kpi-label">Win Rate &amp; Avg Return</div>
          <div class="kpi-value">${fmtNum(k.win_rate_pct, 1)}%</div>
          <div class="kpi-sub">Avg Return: ${fmtPctHTML(k.avg_return_pct)} • Avg MFE: +${fmtNum(k.avg_mfe_pct, 2)}%</div>
        </div>
        <div class="kpi-card kpi-purple">
          <div class="kpi-label">Avg Composite Score</div>
          <div class="kpi-value">${fmtNum(k.avg_turtle_score, 1)}</div>
          <div class="kpi-sub">5-Pillar Scale (0–100)</div>
        </div>
        <div class="kpi-card kpi-amber">
          <div class="kpi-label">Top Tracked Performer</div>
          <div class="kpi-value" style="font-size:17px;">${bestHTML}</div>
          <div class="kpi-sub">${best ? `Rec on ${best.rec_date} • Grade ${best.grade}` : 'No picks yet'}</div>
        </div>
      `;

      document.getElementById("badgeTrackerCount").textContent = k.total_tracked || 0;
      document.getElementById("badgeUniverseCount").textContent = (DATA.universe_snapshot || []).length;
    }

    function renderSideSpreadsheet(tableSelector, sideRecs, allDates) {
      // Sort strictly in decreasing order of Turtle Score
      const sorted = [...sideRecs].sort((a, b) => Number(b.turtle_score || 0) - Number(a.turtle_score || 0));

      // Determine active dates present for this side (or fallback to allDates)
      let dates = allDates.filter(d => sorted.some(r => r.date_wise_closes && r.date_wise_closes[d] !== undefined));
      if (!dates.length) {
        dates = allDates.length ? allDates : (DATA.latest_scan_date ? [DATA.latest_scan_date] : ["Latest"]);
      }

      const dateCount = Math.max(1, dates.length);
      const thead = document.querySelector(`${tableSelector} thead`);
      const tbody = document.querySelector(`${tableSelector} tbody`);

      const dateSubHeaders = dates.map(d => `<th style="text-align:right;">${d}</th>`).join("");
      thead.innerHTML = `
        <tr>
          <th rowspan="2" style="vertical-align:middle; text-align:left;">Security Name</th>
          <th colspan="${dateCount}" style="text-align:center; color:#60a5fa;">Date-Wise Closing Price</th>
          <th rowspan="2" style="vertical-align:middle; text-align:right;">Turtle Score</th>
          <th rowspan="2" style="vertical-align:middle; text-align:right;">Recommended Price</th>
          <th rowspan="2" style="vertical-align:middle; text-align:right;">Donchian Exit</th>
          <th rowspan="2" style="vertical-align:middle; text-align:right;">% Return Since Recommendation</th>
        </tr>
        <tr>
          ${dateSubHeaders}
        </tr>
      `;

      if (!sorted.length) {
        tbody.innerHTML = `<tr><td colspan="${5 + dateCount}" style="text-align:center; padding:24px; color:var(--text-muted);">No recommendations recorded for this side yet.</td></tr>`;
        return;
      }

      tbody.innerHTML = sorted.map((r) => {
        const comp = r.company_name && r.company_name !== r.ticker
          ? `${r.ticker} - ${r.company_name}`
          : `${r.ticker}`;
        const closesMap = r.date_wise_closes || {};
        const dateCells = dates.map(d => {
          const val = closesMap[d];
          return `<td class="mono" style="text-align:right;">${val !== undefined ? '$' + fmtNum(val) : '—'}</td>`;
        }).join("");

        return `
          <tr>
            <td>
              <a class="ticker-link" onclick="openChartModal('${r.rec_id}')" title="Click to open daily chart for ${comp}">
                📈 ${comp}
              </a>
            </td>
            ${dateCells}
            <td class="mono" style="text-align:right; font-weight:800; color:#60a5fa;">${fmtNum(r.turtle_score)}</td>
            <td class="mono" style="text-align:right;">$${fmtNum(r.rec_price)}</td>
            <td class="mono" style="text-align:right; color:#fbbf24;">$${fmtNum(r.current_exit_level)}</td>
            <td style="text-align:right;">${fmtPctHTML(r.return_pct)}</td>
          </tr>
        `;
      }).join("");
    }

    function renderRecSpreadsheetTab() {
      const recs = DATA.recommendations || [];
      const dates = (DATA.date_wise_dates && DATA.date_wise_dates.length)
        ? DATA.date_wise_dates
        : (DATA.latest_scan_date ? [DATA.latest_scan_date] : []);

      const longRecs = recs.filter(r => r.direction === "LONG");
      const shortRecs = recs.filter(r => r.direction === "SHORT");

      document.getElementById("badgeLongSheetCount").textContent = longRecs.length;
      document.getElementById("badgeShortSheetCount").textContent = shortRecs.length;

      renderSideSpreadsheet("#tableLongSheet", longRecs, dates);
      renderSideSpreadsheet("#tableShortSheet", shortRecs, dates);
    }

    function renderDailyTables() {
      const recs = (DATA.recommendations || []).filter(r => r.rec_date === currentScanDate);
      const longs = recs.filter(r => r.direction === "LONG").sort((a, b) => b.turtle_score - a.turtle_score);
      const shorts = recs.filter(r => r.direction === "SHORT").sort((a, b) => b.turtle_score - a.turtle_score);

      document.getElementById("badgeDailyCount").textContent = longs.length + shorts.length;
      document.getElementById("dailyLongSubtitle").textContent = `(${longs.length} qualified on ${currentScanDate})`;
      document.getElementById("dailyShortSubtitle").textContent = `(${shorts.length} qualified on ${currentScanDate})`;

      const headers = `
        <tr>
          <th>Rank</th>
          <th>Ticker (Chart)</th>
          <th>Company</th>
          <th>Sector</th>
          <th>System</th>
          <th>Grade</th>
          <th>Turtle Score</th>
          <th>5-Pillar Profile</th>
          <th>Rec Price</th>
          <th>Breakout Lvl</th>
          <th>Turtle Exit</th>
          <th>Latest/Exit</th>
          <th>Return %</th>
          <th>Status</th>
          <th>Brk_N</th>
          <th>ATR %</th>
          <th>HMA20 Slp</th>
          <th>RSI(14)</th>
          <th>MACD_N</th>
          <th>Stoch %K</th>
          <th>ADX(14)</th>
          <th>RVOL</th>
          <th>60d ROC%</th>
          <th>SMA200 %</th>
        </tr>
      `;

      document.querySelector("#tableDailyLongs thead").innerHTML = headers;
      document.querySelector("#tableDailyShorts thead").innerHTML = headers;

      const buildRows = (list) => {
        if (!list.length) {
          return `<tr><td colspan="24" style="text-align:center; padding:22px; color:var(--text-muted);">No fresh Day-1 breakouts qualified on ${currentScanDate} for this side.</td></tr>`;
        }
        return list.map((r, idx) => `
          <tr>
            <td class="mono" style="font-weight:700; color:var(--text-secondary);">#${idx + 1}</td>
            <td>
              <a class="ticker-link" onclick="openChartModal('${r.rec_id}')" title="Open Interactive Candlestick Chart for ${r.ticker}">
                📈 ${r.ticker}
              </a>
            </td>
            <td style="max-width:170px; overflow:hidden; text-overflow:ellipsis;">${r.company_name || r.ticker}</td>
            <td style="color:var(--text-secondary);">${r.sector || '—'}</td>
            <td>${sysBadgeHTML(r.system)}</td>
            <td>${gradeBadgeHTML(r.grade)}</td>
            <td class="mono" style="font-weight:800;">${fmtNum(r.turtle_score, 2)}</td>
            <td>${pillarMiniBarsHTML(r)}</td>
            <td class="mono">$${fmtNum(r.rec_price)}</td>
            <td class="mono">$${fmtNum(r.breakout_level)}</td>
            <td class="mono" style="color:#fbbf24;">$${fmtNum(r.current_exit_level)}</td>
            <td class="mono">$${fmtNum(r.latest_price)}</td>
            <td>${fmtPctHTML(r.return_pct)}</td>
            <td>${statusBadgeHTML(r.status, r.exit_date)}</td>
            <td class="mono">${fmtNum(r.breakout_n)}N</td>
            <td class="mono">${fmtNum(r.atr_pct)}%</td>
            <td class="mono ${r.hma_slope_n >= 0 ? 'num-pos' : 'num-neg'}">${fmtNum(r.hma_slope_n, 2, true)}</td>
            <td class="mono">${fmtNum(r.rsi_14, 1)}</td>
            <td class="mono ${r.macd_hist_n >= 0 ? 'num-pos' : 'num-neg'}">${fmtNum(r.macd_hist_n, 3, true)}</td>
            <td class="mono">${fmtNum(r.stoch_k, 1)}</td>
            <td class="mono">${fmtNum(r.adx_14, 1)}</td>
            <td class="mono" style="font-weight:700;">${fmtNum(r.rvol)}x</td>
            <td>${fmtPctHTML(r.roc_60)}</td>
            <td>${fmtPctHTML(r.dist_sma200_pct)}</td>
          </tr>
        `).join("");
      };

      document.querySelector("#tableDailyLongs tbody").innerHTML = buildRows(longs);
      document.querySelector("#tableDailyShorts tbody").innerHTML = buildRows(shorts);
    }

    function renderTrackerTable() {
      const q = (document.getElementById("filterTrackerSearch").value || "").trim().toLowerCase();
      const st = document.getElementById("filterTrackerStatus").value;
      const dir = document.getElementById("filterTrackerDir").value;
      const gr = document.getElementById("filterTrackerGrade").value;
      const sys = document.getElementById("filterTrackerSystem").value;
      const dt = document.getElementById("filterTrackerDate").value;

      let filtered = (DATA.recommendations || []).filter(r => {
        if (st !== "ALL" && r.status !== st) return false;
        if (dir !== "ALL" && r.direction !== dir) return false;
        if (gr !== "ALL" && r.grade !== gr) return false;
        if (sys !== "ALL" && r.system !== sys) return false;
        if (dt !== "ALL" && r.rec_date !== dt) return false;
        if (q) {
          const hay = `${r.ticker} ${r.company_name} ${r.sector} ${r.rec_date}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      filtered.sort((a, b) => {
        const va = a[trackerSortCol];
        const vb = b[trackerSortCol];
        if (typeof va === "number" && typeof vb === "number") {
          return trackerSortAsc ? va - vb : vb - va;
        }
        return trackerSortAsc
          ? String(va || "").localeCompare(String(vb || ""))
          : String(vb || "").localeCompare(String(va || ""));
      });

      const cols = [
        ["rec_date", "Rec Date"],
        ["ticker", "Ticker (Chart)"],
        ["company_name", "Company"],
        ["sector", "Sector"],
        ["direction", "Side"],
        ["system", "System"],
        ["upgrade_note", "Upgrade"],
        ["grade", "Grade"],
        ["turtle_score", "Score"],
        ["status", "Status"],
        ["rec_price", "Rec Price"],
        ["breakout_level", "Brk Lvl"],
        ["latest_date", "Latest Date"],
        ["latest_price", "Latest/Exit"],
        ["current_exit_level", "Exit Trail"],
        ["return_pct", "Return %"],
        ["max_favorable_pct", "Max Gain %"],
        ["max_adverse_pct", "Max DD %"],
        ["days_held", "Days"],
        ["atr_pct", "ATR %"],
        ["breakout_n", "Brk_N"],
        ["hma_slope_n", "HMA Slp"],
        ["rsi_14", "RSI(14)"],
        ["macd_hist_n", "MACD_N"],
        ["stoch_k", "Stoch %K"],
        ["adx_14", "ADX(14)"],
        ["rvol", "RVOL"],
        ["roc_60", "60d ROC%"],
        ["dist_sma200_pct", "SMA200%"],
      ];

      document.querySelector("#tableTracker thead").innerHTML = `
        <tr>
          ${cols.map(([key, label]) => {
            const arrow = trackerSortCol === key ? (trackerSortAsc ? " ▲" : " ▼") : "";
            return `<th onclick="sortTrackerBy('${key}')">${label}${arrow}</th>`;
          }).join("")}
        </tr>
      `;

      document.querySelector("#tableTracker tbody").innerHTML = filtered.map(r => `
        <tr>
          <td class="mono">${r.rec_date}</td>
          <td>
            <a class="ticker-link" onclick="openChartModal('${r.rec_id}')">📈 ${r.ticker}</a>
          </td>
          <td style="max-width:160px; overflow:hidden; text-overflow:ellipsis;">${r.company_name || r.ticker}</td>
          <td style="color:var(--text-secondary);">${r.sector || '—'}</td>
          <td>${dirBadgeHTML(r.direction)}</td>
          <td>${sysBadgeHTML(r.system)}</td>
          <td class="mono" style="color:#c4b5fd; font-size:11px;">${r.upgrade_note || '—'}</td>
          <td>${gradeBadgeHTML(r.grade)}</td>
          <td class="mono" style="font-weight:800;">${fmtNum(r.turtle_score)}</td>
          <td>${statusBadgeHTML(r.status, r.exit_date)}</td>
          <td class="mono">$${fmtNum(r.rec_price)}</td>
          <td class="mono">$${fmtNum(r.breakout_level)}</td>
          <td class="mono">${r.latest_date}</td>
          <td class="mono">$${fmtNum(r.latest_price)}</td>
          <td class="mono" style="color:#fbbf24;">$${fmtNum(r.current_exit_level)}</td>
          <td>${fmtPctHTML(r.return_pct)}</td>
          <td>${fmtPctHTML(r.max_favorable_pct)}</td>
          <td>${fmtPctHTML(r.max_adverse_pct)}</td>
          <td class="mono">${r.days_held}d</td>
          <td class="mono">${fmtNum(r.atr_pct)}%</td>
          <td class="mono">${fmtNum(r.breakout_n)}N</td>
          <td class="mono">${fmtNum(r.hma_slope_n, 2, true)}</td>
          <td class="mono">${fmtNum(r.rsi_14, 1)}</td>
          <td class="mono">${fmtNum(r.macd_hist_n, 3, true)}</td>
          <td class="mono">${fmtNum(r.stoch_k, 1)}</td>
          <td class="mono">${fmtNum(r.adx_14, 1)}</td>
          <td class="mono">${fmtNum(r.rvol)}x</td>
          <td>${fmtPctHTML(r.roc_60)}</td>
          <td>${fmtPctHTML(r.dist_sma200_pct)}</td>
        </tr>
      `).join("");
    }

    function sortTrackerBy(col) {
      if (trackerSortCol === col) {
        trackerSortAsc = !trackerSortAsc;
      } else {
        trackerSortCol = col;
        trackerSortAsc = false;
      }
      renderTrackerTable();
    }

    function renderTerminalList() {
      const q = (document.getElementById("termSearch").value || "").trim().toLowerCase();
      const list = (DATA.recommendations || []).filter(r => {
        if (!q) return true;
        return `${r.ticker} ${r.company_name} ${r.rec_date} ${r.direction} ${r.grade}`.toLowerCase().includes(q);
      });

      const container = document.getElementById("terminalSidebarList");
      container.innerHTML = list.map(r => `
        <div class="term-item ${r.rec_id === activeTerminalRecId ? 'active' : ''}" onclick="selectTerminalRec('${r.rec_id}')">
          <div>
            <div style="display:flex; align-items:center; gap:7px;">
              <span class="mono" style="font-weight:800; font-size:13.5px; color:#60a5fa;">${r.ticker}</span>
              ${dirBadgeHTML(r.direction)}
              ${gradeBadgeHTML(r.grade)}
            </div>
            <div style="font-size:11px; color:var(--text-secondary); margin-top:3px;">
              ${r.rec_date} • ${r.system} • Rec $${fmtNum(r.rec_price)}
            </div>
          </div>
          <div style="text-align:right;">
            <div>${fmtPctHTML(r.return_pct)}</div>
            <div style="font-size:10.5px; color:var(--text-muted); margin-top:2px;">${r.status} (${r.days_held}d)</div>
          </div>
        </div>
      `).join("");
    }

    function selectTerminalRec(recId) {
      activeTerminalRecId = recId;
      renderTerminalList();
      const rec = (DATA.recommendations || []).find(r => r.rec_id === recId);
      if (!rec) return;

      document.querySelector("#terminalChartHeader .panel-title").innerHTML = `
        <span class="mono" style="font-size:16px; color:#60a5fa;">${rec.ticker}</span>
        <span>${rec.company_name || ''}</span>
        ${dirBadgeHTML(rec.direction)}
        ${sysBadgeHTML(rec.system)}
        ${gradeBadgeHTML(rec.grade)}
        <span class="mono">Score: ${fmtNum(rec.turtle_score)}</span>
        <span>• Rec: <b>${rec.rec_date} @ $${fmtNum(rec.rec_price)}</b></span>
        <span>→ Latest: <b>$${fmtNum(rec.latest_price)}</b> (${fmtPctHTML(rec.return_pct)})</span>
        ${statusBadgeHTML(rec.status, rec.exit_date)}
      `;
      document.getElementById("terminalOpenNewTab").href = rec.chart_rel_url;
      document.getElementById("terminalPillarStrip").innerHTML = pillarStripHTML(rec);
      document.getElementById("terminalChartFrame").src = rec.chart_rel_url;
    }

    function openChartModal(recId) {
      const rec = (DATA.recommendations || []).find(r => r.rec_id === recId);
      if (!rec) return;
      document.getElementById("modalHeaderTitle").innerHTML = `
        <span class="mono" style="font-size:16px; color:#60a5fa;">${rec.ticker}</span>
        <span>${rec.company_name || ''} (${rec.sector || ''})</span>
        ${dirBadgeHTML(rec.direction)}
        ${sysBadgeHTML(rec.system)}
        ${gradeBadgeHTML(rec.grade)}
        <span class="mono">Score: ${fmtNum(rec.turtle_score)}/100</span>
        <span>• Rec Date: <b>${rec.rec_date} @ $${fmtNum(rec.rec_price)}</b></span>
        <span>→ Latest/Exit: <b>$${fmtNum(rec.latest_price)}</b> (${fmtPctHTML(rec.return_pct)})</span>
        ${statusBadgeHTML(rec.status, rec.exit_date)}
      `;
      document.getElementById("modalOpenNewTab").href = rec.chart_rel_url;
      document.getElementById("modalPillarStrip").innerHTML = pillarStripHTML(rec);
      document.getElementById("modalChartFrame").src = rec.chart_rel_url;
      document.getElementById("chartModal").classList.add("open");
    }

    function closeChartModal() {
      document.getElementById("chartModal").classList.remove("open");
      document.getElementById("modalChartFrame").src = "about:blank";
    }

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeChartModal();
    });

    function renderAnalytics() {
      const recs = DATA.recommendations || [];
      if (!recs.length) return;

      const grades = ["A+", "A", "B+", "B", "C"];
      const gStats = grades.map(g => {
        const sub = recs.filter(r => r.grade === g);
        const cnt = sub.length;
        const wins = sub.filter(r => r.return_pct > 0).length;
        const wr = cnt ? (wins / cnt * 100) : 0;
        const avgRet = cnt ? (sub.reduce((s, r) => s + r.return_pct, 0) / cnt) : 0;
        const avgMfe = cnt ? (sub.reduce((s, r) => s + r.max_favorable_pct, 0) / cnt) : 0;
        return { grade: g, count: cnt, winRate: wr, avgRet, avgMfe };
      }).filter(x => x.count > 0);

      Plotly.newPlot("chartGradePerf", [
        {
          x: gStats.map(x => `Grade ${x.grade} (n=${x.count})`),
          y: gStats.map(x => x.winRate),
          name: "Win Rate %",
          type: "bar",
          marker: { color: "#3b82f6" },
        },
        {
          x: gStats.map(x => `Grade ${x.grade} (n=${x.count})`),
          y: gStats.map(x => x.avgRet),
          name: "Avg Return %",
          type: "bar",
          marker: { color: "#10b981" },
        },
        {
          x: gStats.map(x => `Grade ${x.grade} (n=${x.count})`),
          y: gStats.map(x => x.avgMfe),
          name: "Avg Max Gain (MFE %)",
          type: "bar",
          marker: { color: "#f59e0b" },
        }
      ], {
        barmode: "group",
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#9fb0c7", size: 11 },
        margin: { l: 45, r: 20, t: 20, b: 40 },
        legend: { orientation: "h", y: 1.12 },
        xaxis: { gridcolor: "rgba(255,255,255,0.05)" },
        yaxis: { gridcolor: "rgba(255,255,255,0.05)", title: "Percentage (%)" }
      }, { responsive: true, displayModeBar: false });

      // Scatter: Turtle Score vs Return %
      const longs = recs.filter(r => r.direction === "LONG");
      const shorts = recs.filter(r => r.direction === "SHORT");
      Plotly.newPlot("chartScoreScatter", [
        {
          x: longs.map(r => r.turtle_score),
          y: longs.map(r => r.return_pct),
          text: longs.map(r => `${r.ticker} (${r.rec_date}) Grade ${r.grade}`),
          mode: "markers",
          name: "LONG Picks",
          marker: { color: "#10b981", size: 10, opacity: 0.85, line: { color: "#fff", width: 1 } }
        },
        {
          x: shorts.map(r => r.turtle_score),
          y: shorts.map(r => r.return_pct),
          text: shorts.map(r => `${r.ticker} (${r.rec_date}) Grade ${r.grade}`),
          mode: "markers",
          name: "SHORT Picks",
          marker: { color: "#f43f5e", size: 10, opacity: 0.85, line: { color: "#fff", width: 1 } }
        }
      ], {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#9fb0c7", size: 11 },
        margin: { l: 50, r: 20, t: 20, b: 45 },
        legend: { orientation: "h", y: 1.12 },
        xaxis: { title: "Composite Turtle Score (0–100)", gridcolor: "rgba(255,255,255,0.05)" },
        yaxis: { title: "Directional Return (%)", gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.25)" }
      }, { responsive: true, displayModeBar: false });

      // Sector breakdown
      const secCounts = {};
      recs.forEach(r => {
        const s = r.sector || "Unknown";
        secCounts[s] = (secCounts[s] || 0) + 1;
      });
      const secEntries = Object.entries(secCounts).sort((a, b) => b[1] - a[1]);
      Plotly.newPlot("chartSectorDist", [{
        x: secEntries.map(e => e[0]),
        y: secEntries.map(e => e[1]),
        type: "bar",
        marker: { color: "#06b6d4" }
      }], {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#9fb0c7", size: 11 },
        margin: { l: 45, r: 20, t: 20, b: 85 },
        xaxis: { gridcolor: "rgba(255,255,255,0.05)", tickangle: -25 },
        yaxis: { title: "Recommendations Count", gridcolor: "rgba(255,255,255,0.05)" }
      }, { responsive: true, displayModeBar: false });

      // Cohort Table
      const cohorts = [
        ["ALL TRACKED RECOMMENDATIONS", recs],
        ...grades.map(g => [`Grade ${g}`, recs.filter(r => r.grade === g)]),
        ["Direction: LONG", longs],
        ["Direction: SHORT", shorts],
        ["System 2 (55d Breakout)", recs.filter(r => String(r.system).includes("System 2"))],
        ["System 1 (20d Breakout)", recs.filter(r => String(r.system).includes("System 1"))],
      ].filter(([_, arr]) => arr.length > 0);

      document.querySelector("#tableCohortPerf tbody").innerHTML = cohorts.map(([title, arr]) => {
        const n = arr.length;
        const act = arr.filter(r => r.status === "ACTIVE").length;
        const ext = arr.filter(r => r.status === "EXITED").length;
        const wins = arr.filter(r => r.return_pct > 0).length;
        const wr = (wins / n) * 100;
        const avgR = arr.reduce((s, r) => s + r.return_pct, 0) / n;
        const avgM = arr.reduce((s, r) => s + r.max_favorable_pct, 0) / n;
        const avgS = arr.reduce((s, r) => s + r.turtle_score, 0) / n;
        const avgD = arr.reduce((s, r) => s + r.days_held, 0) / n;
        return `
          <tr>
            <td style="font-weight:700;">${title}</td>
            <td class="mono">${n}</td>
            <td class="mono" style="color:#22d3ee;">${act}</td>
            <td class="mono" style="color:#fbbf24;">${ext}</td>
            <td class="mono">${fmtNum(wr, 1)}%</td>
            <td>${fmtPctHTML(avgR)}</td>
            <td>${fmtPctHTML(avgM)}</td>
            <td class="mono">${fmtNum(avgS, 1)}</td>
            <td class="mono">${fmtNum(avgD, 1)}d</td>
          </tr>
        `;
      }).join("");
    }

    function renderUniverseTable() {
      const q = (document.getElementById("filterUniSearch").value || "").trim().toLowerCase();
      const sig = document.getElementById("filterUniSignal").value;
      const sec = document.getElementById("filterUniSector").value;

      let list = (DATA.universe_snapshot || []).filter(r => {
        if (sec !== "ALL" && r.sector !== sec) return false;
        if (sig === "FRESH" && !String(r.signal_tag).startsWith("Fresh")) return false;
        if (sig === "NEAR_HIGH" && !(r.pct_to_h20 >= -1.5 || r.pct_to_h55 >= -1.5)) return false;
        if (sig === "NEAR_LOW" && !(r.pct_to_l20 <= 1.5 || r.pct_to_l55 <= 1.5)) return false;
        if (q) {
          const hay = `${r.ticker} ${r.company_name} ${r.sector} ${r.signal_tag}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      list.sort((a, b) => {
        const va = a[uniSortCol];
        const vb = b[uniSortCol];
        if (typeof va === "number" && typeof vb === "number") {
          return uniSortAsc ? va - vb : vb - va;
        }
        return uniSortAsc
          ? String(va || "").localeCompare(String(vb || ""))
          : String(vb || "").localeCompare(String(va || ""));
      });

      const cols = [
        ["ticker", "Ticker"],
        ["company_name", "Company"],
        ["sector", "Sector"],
        ["signal_tag", "Donchian Status"],
        ["close", "Close"],
        ["high_20", "20d High"],
        ["pct_to_h20", "% vs 20d High"],
        ["high_55", "55d High"],
        ["pct_to_h55", "% vs 55d High"],
        ["low_20", "20d Low"],
        ["pct_to_l20", "% vs 20d Low"],
        ["atr_pct", "ATR %"],
        ["hma_slope_n", "HMA20 Slp"],
        ["rsi_14", "RSI(14)"],
        ["macd_hist_n", "MACD_N"],
        ["stoch_k", "Stoch %K"],
        ["adx_14", "ADX(14)"],
        ["rvol", "RVOL"],
        ["roc_60", "60d ROC%"],
        ["dist_sma200_pct", "SMA200%"],
      ];

      document.querySelector("#tableUniverse thead").innerHTML = `
        <tr>
          ${cols.map(([k, label]) => {
            const arrow = uniSortCol === k ? (uniSortAsc ? " ▲" : " ▼") : "";
            return `<th onclick="sortUniBy('${k}')">${label}${arrow}</th>`;
          }).join("")}
        </tr>
      `;

      document.querySelector("#tableUniverse tbody").innerHTML = list.map(r => {
        let sigBadge = `<span style="color:var(--text-muted);">${r.signal_tag}</span>`;
        if (r.signal_tag.includes("Fresh Long")) sigBadge = `<span class="dir-pill dir-long">${r.signal_tag}</span>`;
        else if (r.signal_tag.includes("Fresh Short")) sigBadge = `<span class="dir-pill dir-short">${r.signal_tag}</span>`;
        else if (r.signal_tag.includes("Cont. Long")) sigBadge = `<span style="color:#34d399; font-weight:600;">${r.signal_tag}</span>`;
        else if (r.signal_tag.includes("Cont. Short")) sigBadge = `<span style="color:#fb7185; font-weight:600;">${r.signal_tag}</span>`;

        return `
          <tr>
            <td class="mono" style="font-weight:700; color:#60a5fa;">${r.ticker}</td>
            <td style="max-width:180px; overflow:hidden; text-overflow:ellipsis;">${r.company_name}</td>
            <td style="color:var(--text-secondary);">${r.sector}</td>
            <td>${sigBadge}</td>
            <td class="mono">$${fmtNum(r.close)}</td>
            <td class="mono">$${fmtNum(r.high_20)}</td>
            <td>${fmtPctHTML(r.pct_to_h20)}</td>
            <td class="mono">$${fmtNum(r.high_55)}</td>
            <td>${fmtPctHTML(r.pct_to_h55)}</td>
            <td class="mono">$${fmtNum(r.low_20)}</td>
            <td>${fmtPctHTML(r.pct_to_l20)}</td>
            <td class="mono">${fmtNum(r.atr_pct)}%</td>
            <td class="mono ${r.hma_slope_n >= 0 ? 'num-pos' : 'num-neg'}">${fmtNum(r.hma_slope_n, 2, true)}</td>
            <td class="mono">${fmtNum(r.rsi_14, 1)}</td>
            <td class="mono ${r.macd_hist_n >= 0 ? 'num-pos' : 'num-neg'}">${fmtNum(r.macd_hist_n, 3, true)}</td>
            <td class="mono">${fmtNum(r.stoch_k, 1)}</td>
            <td class="mono">${fmtNum(r.adx_14, 1)}</td>
            <td class="mono">${fmtNum(r.rvol)}x</td>
            <td>${fmtPctHTML(r.roc_60)}</td>
            <td>${fmtPctHTML(r.dist_sma200_pct)}</td>
          </tr>
        `;
      }).join("");
    }

    function sortUniBy(col) {
      if (uniSortCol === col) {
        uniSortAsc = !uniSortAsc;
      } else {
        uniSortCol = col;
        uniSortAsc = false;
      }
      renderUniverseTable();
    }

    function switchTab(tabId, btnEl) {
      document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
      document.getElementById(tabId).classList.add("active");
      btnEl.classList.add("active");
      if (tabId === "tab-analytics") {
        setTimeout(() => window.dispatchEvent(new Event("resize")), 60);
      }
    }

    async function triggerLiveScan() {
      const btn = document.getElementById("btnRunLiveScan");
      btn.disabled = true;
      btn.innerHTML = "<span>⏳</span> Syncing &amp; Scanning...";
      showToast("Running incremental EOD sync and Turtle breakout scan...");
      try {
        const resp = await fetch("/api/run-scan", { method: "POST" });
        if (!resp.ok) throw new Error("Server endpoint not active");
        const res = await resp.json();
        showToast(`Scan complete for ${res.scan_date}! Reloading dashboard...`);
        setTimeout(() => window.location.reload(), 900);
      } catch (err) {
        showToast("Static mode active: Run 'python screener.py' or launch 'python app.py' for live server actions.", 5500);
      } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>⟳</span> Sync EOD &amp; Run Scan";
      }
    }

    window.addEventListener("DOMContentLoaded", initDashboard);
  </script>
</body>
</html>
"""
