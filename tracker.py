"""Lifecycle manager for active/exited recommendations and Excel (.xlsx) tracker exporter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from charting import generate_recommendation_chart, get_chart_filepath
from config import CHARTS_DIR, DAILY_CSV_DIR, DB_PATH, TRACKER_XLSX_PATH
from database import (
    get_active_recommendations,
    get_all_recommendations,
    upsert_recommendations,
)

logger = logging.getLogger(__name__)


def _calc_return_pct(direction: str, entry_price: float, current_price: float) -> float:
    if entry_price <= 0.0:
        return 0.0
    if direction == "LONG":
        return ((current_price - entry_price) / entry_price) * 100.0
    return ((entry_price - current_price) / entry_price) * 100.0


def update_active_recommendations(
    indicator_frames: Dict[str, pd.DataFrame],
    as_of_date: Optional[str] = None,
    generate_charts: bool = True,
    db_path: Path = DB_PATH,
    charts_dir: Path = CHARTS_DIR,
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str]]:
    """
    Step through all currently ACTIVE recommendations up to `as_of_date`.
    - Updates Latest Price, Return %, Max Gain %, Max Drawdown %, Days Held, and S1->S2 Upgrades.
    - Freezes a recommendation as EXITED on the exact date its Close breaches the System-matched
      Turtle Exit channel (10d Low/High for System 1; 20d Low/High for System 2).
    - Regenerates HTML charts for all updated active and newly exited picks.
    Returns (updated_records, active_long_tickers, active_short_tickers).
    """
    active_recs = get_active_recommendations(before_or_on_date=as_of_date, db_path=db_path)
    if not active_recs:
        return [], set(), set()

    updated_records: List[Dict[str, Any]] = []
    still_active_longs: Set[str] = set()
    still_active_shorts: Set[str] = set()

    for rec in active_recs:
        tkr = str(rec["ticker"])
        direction = str(rec["direction"])
        system = str(rec["system"])
        is_sys2 = "System 2" in system
        ind_df = indicator_frames.get(tkr)

        if ind_df is None or ind_df.empty:
            if direction == "LONG":
                still_active_longs.add(tkr)
            else:
                still_active_shorts.add(tkr)
            continue

        rec_date = str(rec["rec_date"])
        last_checked = str(rec.get("latest_date") or rec_date)
        rec_price = float(rec["rec_price"])

        # Slice trading days strictly after last_checked up to as_of_date
        all_dates = [str(d) for d in ind_df.index]
        if as_of_date:
            all_dates = [d for d in all_dates if d <= as_of_date]

        new_dates = [d for d in all_dates if d > last_checked]
        days_held = int([d for d in all_dates if d > rec_date].__len__())

        max_fav = float(rec.get("max_favorable_pct", 0.0))
        max_adv = float(rec.get("max_adverse_pct", 0.0))
        upgrade_note = str(rec.get("upgrade_note") or "")
        status = "ACTIVE"
        exit_date = ""
        exit_price = None

        for idx_step, dt_str in enumerate(new_dates):
            row = ind_df.loc[dt_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            close_val = float(row["close"])
            high_val = float(row["high"])
            low_val = float(row["low"])

            # Determine active Turtle exit channel level on day dt_str
            if direction == "LONG":
                exit_lvl = float(row["low_20"] if is_sys2 else row["low_10"])
                favorable_extreme = _calc_return_pct("LONG", rec_price, high_val)
                adverse_extreme = _calc_return_pct("LONG", rec_price, low_val)
                # Check S1 -> S2 Upgrade
                if not is_sys2 and not upgrade_note and bool(row.get("close", 0) > row.get("high_55", np.inf)):
                    upgrade_note = f"S1->S2 Upgrade ({dt_str})"
                breached_exit = (not np.isnan(exit_lvl)) and (close_val < exit_lvl)
            else:
                exit_lvl = float(row["high_20"] if is_sys2 else row["high_10"])
                favorable_extreme = _calc_return_pct("SHORT", rec_price, low_val)
                adverse_extreme = _calc_return_pct("SHORT", rec_price, high_val)
                if not is_sys2 and not upgrade_note and bool(row.get("close", 0) < row.get("low_55", -np.inf)):
                    upgrade_note = f"S1->S2 Upgrade ({dt_str})"
                breached_exit = (not np.isnan(exit_lvl)) and (close_val > exit_lvl)

            cur_ret = _calc_return_pct(direction, rec_price, close_val)
            max_fav = max(max_fav, favorable_extreme, cur_ret)
            max_adv = min(max_adv, adverse_extreme, cur_ret)

            rec["latest_date"] = dt_str
            rec["latest_price"] = round(close_val, 2)
            if not np.isnan(exit_lvl):
                rec["current_exit_level"] = round(exit_lvl, 2)
            rec["return_pct"] = round(cur_ret, 2)
            rec["max_favorable_pct"] = round(max_fav, 2)
            rec["max_adverse_pct"] = round(max_adv, 2)
            rec["upgrade_note"] = upgrade_note
            rec["days_held"] = len([d for d in all_dates if rec_date < d <= dt_str])

            if breached_exit:
                status = "EXITED"
                exit_date = dt_str
                exit_price = round(close_val, 2)
                rec["status"] = "EXITED"
                rec["exit_date"] = exit_date
                rec["exit_price"] = exit_price
                break

        if status == "ACTIVE":
            rec["days_held"] = days_held
            if direction == "LONG":
                still_active_longs.add(tkr)
            else:
                still_active_shorts.add(tkr)

        chart_file = get_chart_filepath(rec, charts_dir=charts_dir)
        rec["chart_path"] = str(chart_file)
        if generate_charts:
            generate_recommendation_chart(rec, ind_df, charts_dir=charts_dir)

        updated_records.append(rec)

    if updated_records:
        upsert_recommendations(updated_records, db_path=db_path)

    return updated_records, still_active_longs, still_active_shorts


def register_new_recommendations(
    top_longs: pd.DataFrame,
    top_shorts: pd.DataFrame,
    indicator_frames: Dict[str, pd.DataFrame],
    generate_charts: bool = True,
    db_path: Path = DB_PATH,
    charts_dir: Path = CHARTS_DIR,
) -> List[Dict[str, Any]]:
    """Persist today's new Top 10 Longs and Top 10 Shorts into SQLite and generate their daily charts."""
    new_records: List[Dict[str, Any]] = []
    combined = pd.concat([top_longs, top_shorts], ignore_index=True)
    if combined.empty:
        return new_records

    for _, row in combined.iterrows():
        rec = row.to_dict()
        rec["status"] = "ACTIVE"
        rec["latest_date"] = str(rec["rec_date"])
        rec["latest_price"] = float(rec["rec_price"])
        rec["current_exit_level"] = float(rec["initial_exit_level"])
        rec["return_pct"] = 0.0
        rec["max_favorable_pct"] = 0.0
        rec["max_adverse_pct"] = 0.0
        rec["days_held"] = 0
        rec["exit_date"] = ""
        rec["exit_price"] = None

        chart_file = get_chart_filepath(rec, charts_dir=charts_dir)
        rec["chart_path"] = str(chart_file)
        tkr = str(rec["ticker"])
        if generate_charts and tkr in indicator_frames:
            generate_recommendation_chart(rec, indicator_frames[tkr], charts_dir=charts_dir)

        new_records.append(rec)

    upsert_recommendations(new_records, db_path=db_path)
    return new_records


def save_daily_csv_reports(
    top_longs: pd.DataFrame,
    top_shorts: pd.DataFrame,
    scan_date: str,
    csv_dir: Path = DAILY_CSV_DIR,
) -> Tuple[Path, Path]:
    """Save dated CSV snapshots for today's Top 10 Longs and Top 10 Shorts."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    long_path = csv_dir / f"top10_longs_{scan_date}.csv"
    short_path = csv_dir / f"top10_shorts_{scan_date}.csv"
    top_longs.to_csv(long_path, index=False)
    top_shorts.to_csv(short_path, index=False)
    return long_path, short_path


def export_excel_tracker(
    top_longs: pd.DataFrame,
    top_shorts: pd.DataFrame,
    scan_date: str,
    xlsx_path: Path = TRACKER_XLSX_PATH,
    db_path: Path = DB_PATH,
) -> Path:
    """
    Create/update the multi-tab Excel workbook (`output/turtle_recommendations_tracker.xlsx`):
    - Tab 1 (`Tracker_Log`): Master tracking table of all recommendations (Active + Frozen Exited)
      with clickable hyperlinks on every Ticker opening its daily HTML candlestick chart.
    - Tab 2 (`Latest_Top10`): Clean dashboard of the latest scan's Top 10 Longs and Top 10 Shorts.
    - Tab 3 (`Grade_Performance`): Summary analytics by Grade, Direction, and System.
    """
    all_recs = get_all_recommendations(db_path=db_path)
    wb = Workbook()

    # Styles
    header_fill = PatternFill(start_color="1B263B", end_color="1B263B", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    link_font = Font(name="Calibri", size=11, bold=True, color="0066CC", underline="single")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=11)

    active_fill = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")
    exited_fill = PatternFill(start_color="ECEFF1", end_color="ECEFF1", fill_type="solid")
    pos_fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
    neg_fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")
    pos_font = Font(name="Calibri", size=11, bold=True, color="1B5E20")
    neg_font = Font(name="Calibri", size=11, bold=True, color="B71C1C")

    grade_fills = {
        "A+": PatternFill(start_color="A5D6A7", end_color="A5D6A7", fill_type="solid"),
        "A": PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid"),
        "B+": PatternFill(start_color="FFF59D", end_color="FFF59D", fill_type="solid"),
        "B": PatternFill(start_color="FFE082", end_color="FFE082", fill_type="solid"),
        "C": PatternFill(start_color="FFCCBC", end_color="FFCCBC", fill_type="solid"),
    }

    thin_border = Border(
        left=Side(style="thin", color="D0D7DE"),
        right=Side(style="thin", color="D0D7DE"),
        top=Side(style="thin", color="D0D7DE"),
        bottom=Side(style="thin", color="D0D7DE"),
    )

    # =========================================================================
    # TAB 1: Tracker_Log
    # =========================================================================
    ws_log = wb.active
    ws_log.title = "Tracker_Log"
    ws_log.views.sheetView[0].showGridLines = True

    log_columns = [
        ("Rec_Date", "rec_date"),
        ("Ticker (Click for Chart)", "ticker"),
        ("Company", "company_name"),
        ("Sector", "sector"),
        ("Direction", "direction"),
        ("System", "system"),
        ("Upgrade_Note", "upgrade_note"),
        ("Grade", "grade"),
        ("Turtle_Score", "turtle_score"),
        ("Status", "status"),
        ("Rec_Price", "rec_price"),
        ("Breakout_Lvl", "breakout_level"),
        ("Latest_Date", "latest_date"),
        ("Latest_or_Exit_Price", "latest_price"),
        ("Current_Exit_Lvl", "current_exit_level"),
        ("Return_%", "return_pct"),
        ("Max_Gain_%", "max_favorable_pct"),
        ("Max_DD_%", "max_adverse_pct"),
        ("Days_Held", "days_held"),
        ("Exit_Date", "exit_date"),
        ("ATR_%", "atr_pct"),
        ("Breakout_N", "breakout_n"),
        ("HMA_20", "hma_20"),
        ("HMA_Slope_N", "hma_slope_n"),
        ("RSI_14", "rsi_14"),
        ("MACD_Hist_N", "macd_hist_n"),
        ("Stoch_K", "stoch_k"),
        ("ADX_14", "adx_14"),
        ("RVOL", "rvol"),
        ("ROC_60_%", "roc_60"),
        ("Dist_SMA200_%", "dist_sma200_pct"),
    ]

    for col_idx, (col_title, _) in enumerate(log_columns, start=1):
        cell = ws_log.cell(row=1, column=col_idx, value=col_title)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    ws_log.freeze_panes = "C2"

    for row_idx, rec in enumerate(all_recs, start=2):
        for col_idx, (col_title, key) in enumerate(log_columns, start=1):
            val = rec.get(key, "")
            cell = ws_log.cell(row=row_idx, column=col_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border

            if key == "ticker":
                chart_str = str(rec.get("chart_path") or "")
                if chart_str:
                    try:
                        rel_link = os.path.relpath(chart_str, str(xlsx_path.parent)).replace("\\", "/")
                    except ValueError:
                        rel_link = Path(chart_str).resolve().as_uri()
                    cell.hyperlink = rel_link
                    cell.font = link_font
                cell.alignment = Alignment(horizontal="center")
            elif key == "grade":
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center")
                if str(val) in grade_fills:
                    cell.fill = grade_fills[str(val)]
            elif key == "status":
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center")
                cell.fill = active_fill if val == "ACTIVE" else exited_fill
            elif key == "return_pct":
                num_val = float(val or 0.0)
                cell.number_format = "+0.00;-0.00;0.00"
                cell.fill = pos_fill if num_val > 0 else (neg_fill if num_val < 0 else exited_fill)
                cell.font = pos_font if num_val > 0 else (neg_font if num_val < 0 else regular_font)
            elif isinstance(val, float):
                cell.number_format = "0.00"

    # =========================================================================
    # TAB 2: Latest_Top10
    # =========================================================================
    ws_latest = wb.create_sheet(title="Latest_Top10")
    ws_latest.views.sheetView[0].showGridLines = True

    latest_cols = [
        ("Rank", None),
        ("Ticker (Click for Chart)", "ticker"),
        ("Company", "company_name"),
        ("Sector", "sector"),
        ("System", "system"),
        ("Grade", "grade"),
        ("Turtle_Score", "turtle_score"),
        ("Close", "rec_price"),
        ("Breakout_Lvl", "breakout_level"),
        ("Exit_Lvl", "initial_exit_level"),
        ("Breakout_N", "breakout_n"),
        ("ATR_%", "atr_pct"),
        ("HMA20_Slope_N", "hma_slope_n"),
        ("RSI_14", "rsi_14"),
        ("MACD_Hist_N", "macd_hist_n"),
        ("Stoch_K", "stoch_k"),
        ("ADX_14", "adx_14"),
        ("RVOL", "rvol"),
        ("ROC_60_%", "roc_60"),
        ("Dist_SMA200_%", "dist_sma200_pct"),
    ]

    def _write_latest_section(start_row: int, title: str, df_side: pd.DataFrame, direction: str) -> int:
        banner = ws_latest.cell(row=start_row, column=1, value=f"{title} — Scan Date: {scan_date}")
        banner.font = Font(name="Calibri", size=13, bold=True, color="1B263B")
        start_row += 1

        for c_idx, (c_title, _) in enumerate(latest_cols, start=1):
            c_cell = ws_latest.cell(row=start_row, column=c_idx, value=c_title)
            c_cell.fill = header_fill
            c_cell.font = header_font
            c_cell.alignment = Alignment(horizontal="center")
            c_cell.border = thin_border
        start_row += 1

        if df_side.empty:
            empty_cell = ws_latest.cell(
                row=start_row,
                column=1,
                value=f"No fresh Day-1 {direction} breakouts qualified on {scan_date}.",
            )
            empty_cell.font = regular_font
            return start_row + 2

        for rank_idx, (_, s_row) in enumerate(df_side.iterrows(), start=1):
            rec_dict = s_row.to_dict()
            for c_idx, (_, key) in enumerate(latest_cols, start=1):
                val = rank_idx if key is None else rec_dict.get(key, "")
                cell = ws_latest.cell(row=start_row, column=c_idx, value=val)
                cell.font = regular_font
                cell.border = thin_border

                if key == "ticker":
                    chart_file = get_chart_filepath(rec_dict)
                    try:
                        rel_link = os.path.relpath(str(chart_file), str(xlsx_path.parent)).replace("\\", "/")
                    except ValueError:
                        rel_link = chart_file.resolve().as_uri()
                    cell.hyperlink = rel_link
                    cell.font = link_font
                    cell.alignment = Alignment(horizontal="center")
                elif key == "grade":
                    cell.font = bold_font
                    cell.alignment = Alignment(horizontal="center")
                    if str(val) in grade_fills:
                        cell.fill = grade_fills[str(val)]
                elif isinstance(val, float):
                    cell.number_format = "0.00"
            start_row += 1

        return start_row + 2

    next_r = _write_latest_section(1, "TOP 10 LONG BREAKOUT CANDIDATES", top_longs, "LONG")
    _write_latest_section(next_r, "TOP 10 SHORT BREAKOUT CANDIDATES", top_shorts, "SHORT")

    # =========================================================================
    # TAB 3: Grade_Performance
    # =========================================================================
    ws_perf = wb.create_sheet(title="Grade_Performance")
    ws_perf.views.sheetView[0].showGridLines = True

    perf_headers = [
        "Grade / Cohort",
        "Total_Picks",
        "Active_Count",
        "Exited_Count",
        "Win_Rate_%",
        "Avg_Return_%",
        "Median_Return_%",
        "Avg_Max_Gain_%",
        "Avg_Days_Held",
    ]
    for c_idx, h_text in enumerate(perf_headers, start=1):
        c_cell = ws_perf.cell(row=1, column=c_idx, value=h_text)
        c_cell.fill = header_fill
        c_cell.font = header_font
        c_cell.alignment = Alignment(horizontal="center")
        c_cell.border = thin_border

    if all_recs:
        df_all = pd.DataFrame(all_recs)
        cohorts: List[Tuple[str, pd.DataFrame]] = [("ALL RECOMMENDATIONS", df_all)]
        for g_label in ["A+", "A", "B+", "B", "C"]:
            sub_g = df_all[df_all["grade"] == g_label]
            if not sub_g.empty:
                cohorts.append((f"Grade {g_label}", sub_g))
        for d_label in ["LONG", "SHORT"]:
            sub_d = df_all[df_all["direction"] == d_label]
            if not sub_d.empty:
                cohorts.append((f"Direction: {d_label}", sub_d))
        for sys_label in ["System 2 (55d)", "System 1 (20d)"]:
            sub_s = df_all[df_all["system"] == sys_label]
            if not sub_s.empty:
                cohorts.append((f"System: {sys_label}", sub_s))

        for r_idx, (c_name, c_df) in enumerate(cohorts, start=2):
            total = len(c_df)
            act_cnt = int((c_df["status"] == "ACTIVE").sum())
            ext_cnt = int((c_df["status"] == "EXITED").sum())
            wins = int((c_df["return_pct"] > 0.0).sum())
            win_rate = (wins / total * 100.0) if total > 0 else 0.0
            avg_ret = float(c_df["return_pct"].mean()) if total > 0 else 0.0
            med_ret = float(c_df["return_pct"].median()) if total > 0 else 0.0
            avg_mfe = float(c_df["max_favorable_pct"].mean()) if total > 0 else 0.0
            avg_days = float(c_df["days_held"].mean()) if total > 0 else 0.0

            row_vals = [
                c_name,
                total,
                act_cnt,
                ext_cnt,
                round(win_rate, 1),
                round(avg_ret, 2),
                round(med_ret, 2),
                round(avg_mfe, 2),
                round(avg_days, 1),
            ]
            for c_idx, val in enumerate(row_vals, start=1):
                cell = ws_perf.cell(row=r_idx, column=c_idx, value=val)
                cell.font = bold_font if c_idx == 1 else regular_font
                cell.border = thin_border
                if isinstance(val, float):
                    cell.number_format = "0.00"

    # Auto-fit column widths across all sheets
    for ws in (ws_log, ws_latest, ws_perf):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 34)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(xlsx_path))
    return xlsx_path
