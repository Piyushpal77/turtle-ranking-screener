"""Interactive daily HTML candlestick & indicator chart generator using Plotly."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_LOOKBACK_BARS_BEFORE_REC, CHARTS_DIR

logger = logging.getLogger(__name__)


def get_chart_filepath(rec: Dict[str, Any], charts_dir: Path = CHARTS_DIR) -> Path:
    """Return deterministic HTML chart path for a recommendation."""
    charts_dir.mkdir(parents=True, exist_ok=True)
    rec_id = rec["rec_id"]
    return charts_dir / f"{rec_id}.html"


def generate_recommendation_chart(
    rec: Dict[str, Any],
    ind_df: pd.DataFrame,
    charts_dir: Path = CHARTS_DIR,
) -> Path:
    """
    Generate or update the standalone interactive HTML daily chart for a recommendation.
    - Renders a dedicated non-overlapping HTML summary header above the chart canvas.
    - Shows 90 trading days prior to `rec_date` up to `latest_date` (or `exit_date` if EXITED).
    - Plots candlesticks, Donchian Breakout channel, Turtle Exit trail, Hull MA (20), SMA (200).
    - Plots a vertical dotted line on `rec_date` and a horizontal line at `rec_price`.
    - Marks the exit point if the recommendation has breached its Turtle Exit channel and frozen.
    """
    chart_path = get_chart_filepath(rec, charts_dir=charts_dir)
    if ind_df is None or ind_df.empty:
        return chart_path

    rec_date = str(rec["rec_date"])
    status = str(rec.get("status", "ACTIVE"))
    end_date = str(rec.get("exit_date") or rec.get("latest_date") or ind_df.index[-1])

    # Determine slice indices
    dates = list(ind_df.index.astype(str))
    if rec_date in dates:
        rec_idx = dates.index(rec_date)
    else:
        after = [i for i, d in enumerate(dates) if d >= rec_date]
        rec_idx = after[0] if after else max(0, len(dates) - 1)
        rec_date = dates[rec_idx]

    end_candidates = [i for i, d in enumerate(dates) if d <= end_date]
    end_idx = end_candidates[-1] if end_candidates else len(dates) - 1
    start_idx = max(0, rec_idx - CHART_LOOKBACK_BARS_BEFORE_REC)

    sub = ind_df.iloc[start_idx : end_idx + 1].copy()
    if sub.empty:
        return chart_path

    x_dates = list(sub.index.astype(str))
    direction = str(rec["direction"])
    system = str(rec["system"])
    is_sys2 = "System 2" in system
    ticker = str(rec["ticker"])
    company = str(rec.get("company_name", ticker))
    grade = str(rec["grade"])
    score = float(rec["turtle_score"])
    rec_price = float(rec["rec_price"])
    latest_price = float(
        rec.get("exit_price") if status == "EXITED" and rec.get("exit_price") else rec["latest_price"]
    )
    ret_pct = float(rec.get("return_pct", 0.0))
    days_held = int(rec.get("days_held", 0))
    cur_exit_lvl = float(rec.get("current_exit_level", rec.get("initial_exit_level", 0.0)))

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.032,
        row_heights=[0.54, 0.15, 0.15, 0.16],
        subplot_titles=(
            f"{ticker} ({company}) — Daily Price, Donchian Channels, Hull MA (20) & SMA (200)",
            "Volume & 20-Day SMA (RVOL Baseline)",
            "MACD (12, 26, 9)",
            "RSI (14) & Slow Stochastic (14, 3, 3)",
        ),
    )

    # 1. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=x_dates,
            open=sub["open"],
            high=sub["high"],
            low=sub["low"],
            close=sub["close"],
            name="OHLC",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        ),
        row=1,
        col=1,
    )

    # 2. Donchian Entry Channel & Turtle Exit Channel
    upper_col = "high_55" if is_sys2 else "high_20"
    lower_col = "low_55" if is_sys2 else "low_20"
    chan_label = "55d Donchian" if is_sys2 else "20d Donchian"

    if direction == "LONG":
        exit_col = "low_20" if is_sys2 else "low_10"
        exit_label = "20d Donchian Exit Low" if is_sys2 else "10d Donchian Exit Low"
    else:
        exit_col = "high_20" if is_sys2 else "high_10"
        exit_label = "20d Donchian Exit High" if is_sys2 else "10d Donchian Exit High"

    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub[upper_col],
            mode="lines",
            name=f"{chan_label} High",
            line=dict(color="rgba(100, 181, 246, 0.55)", width=1.2, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub[lower_col],
            mode="lines",
            name=f"{chan_label} Low",
            line=dict(color="rgba(100, 181, 246, 0.55)", width=1.2, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(100, 181, 246, 0.06)",
        ),
        row=1,
        col=1,
    )

    # Turtle Exit Channel Trail
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub[exit_col],
            mode="lines",
            name=exit_label,
            line=dict(color="#FFA726", width=1.8, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # 3. Hull Moving Average (HMA 20) & SMA 200
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["hma_20"],
            mode="lines",
            name="Hull MA (20)",
            line=dict(color="#00E5FF", width=2.0),
        ),
        row=1,
        col=1,
    )
    if "sma_200" in sub.columns and sub["sma_200"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=sub["sma_200"],
                mode="lines",
                name="SMA (200)",
                line=dict(color="#FFD54F", width=1.6),
            ),
            row=1,
            col=1,
        )

    # 4. Horizontal Recommended Price Line (from rec_date to end_date)
    post_rec_dates = [d for d in x_dates if d >= rec_date]
    accent_color = "#00E676" if direction == "LONG" else "#FF5252"
    if post_rec_dates:
        fig.add_trace(
            go.Scatter(
                x=post_rec_dates,
                y=[rec_price] * len(post_rec_dates),
                mode="lines",
                name=f"Recommended Price (${rec_price:.2f})",
                line=dict(color=accent_color, width=2.2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Position label to the left if rec_date is near the right edge of the chart
        is_near_right_edge = (len(x_dates) - 1 - x_dates.index(rec_date)) < 12
        text_pos = "middle left" if is_near_right_edge else "middle right"
        label_text = (
            f"REC {direction} @ ${rec_price:.2f} ({grade})  "
            if is_near_right_edge
            else f"  REC {direction} @ ${rec_price:.2f} ({grade})"
        )

        fig.add_trace(
            go.Scatter(
                x=[rec_date],
                y=[rec_price],
                mode="markers+text",
                name="Recommendation Entry",
                marker=dict(
                    symbol="triangle-up" if direction == "LONG" else "triangle-down",
                    size=13,
                    color=accent_color,
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                text=[label_text],
                textposition=text_pos,
                textfont=dict(color=accent_color, size=11.5, family="Arial Black"),
                cliponaxis=False,
            ),
            row=1,
            col=1,
        )

    # 5. Vertical Dotted Line on Recommendation Date across all panels
    for r_idx in (1, 2, 3, 4):
        fig.add_vline(
            x=rec_date,
            line_width=2.0,
            line_dash="dot",
            line_color=accent_color,
            row=r_idx,
            col=1,
        )

    # Place Rec Date callout badge inside top of price chart area (never overlapping subplot title)
    fig.add_annotation(
        x=rec_date,
        y=0.98,
        xref="x",
        yref="y domain",
        xanchor="right",
        yanchor="top",
        xshift=-6,
        text=f"Rec Date: {rec_date} (${rec_price:.2f})",
        showarrow=False,
        font=dict(color="#0B0E14", size=11, family="Arial"),
        bgcolor=accent_color,
        bordercolor=accent_color,
        borderpad=4,
    )

    # 6. If EXITED, add Vertical Dotted Line & Marker on Exit Date
    exit_date_str = str(rec.get("exit_date") or "")
    if status == "EXITED" and exit_date_str and exit_date_str in x_dates:
        exit_price_val = float(rec.get("exit_price") or latest_price)
        for r_idx in (1, 2, 3, 4):
            fig.add_vline(
                x=exit_date_str,
                line_width=1.8,
                line_dash="dot",
                line_color="#FFB300",
                row=r_idx,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=[exit_date_str],
                y=[exit_price_val],
                mode="markers+text",
                name=f"Donchian Exit (${exit_price_val:.2f})",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="#FFB300",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                text=[f"EXITED @ ${exit_price_val:.2f} ({ret_pct:+.2f}%)  "],
                textposition="middle left",
                textfont=dict(color="#FFB300", size=11, family="Arial Black"),
                cliponaxis=False,
            ),
            row=1,
            col=1,
        )

    # Row 2: Volume + 20-Day Volume SMA
    vol_colors = [
        "#26A69A" if c >= o else "#EF5350"
        for o, c in zip(sub["open"], sub["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=sub["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.85,
        ),
        row=2,
        col=1,
    )
    if "vol_sma_20" in sub.columns:
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=sub["vol_sma_20"],
                mode="lines",
                name="Vol SMA (20)",
                line=dict(color="#FFD54F", width=1.5),
            ),
            row=2,
            col=1,
        )

    # Row 3: MACD (12, 26, 9)
    macd_colors = ["#26A69A" if h >= 0 else "#EF5350" for h in sub["macd_hist"].fillna(0.0)]
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=sub["macd_hist"],
            name="MACD Hist",
            marker_color=macd_colors,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["macd_line"],
            mode="lines",
            name="MACD",
            line=dict(color="#29B6F6", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["macd_signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#FFA726", width=1.5),
        ),
        row=3,
        col=1,
    )

    # Row 4: RSI (14) & Stochastic (%K / %D)
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["rsi_14"],
            mode="lines",
            name="RSI (14)",
            line=dict(color="#AB47BC", width=1.8),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["stoch_k"],
            mode="lines",
            name="Stoch %K",
            line=dict(color="#26C6DA", width=1.3, dash="dot"),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=sub["stoch_d"],
            mode="lines",
            name="Stoch %D",
            line=dict(color="#FF7043", width=1.2, dash="dot"),
        ),
        row=4,
        col=1,
    )
    for lvl in (20, 50, 80):
        fig.add_hline(
            y=lvl,
            line_width=1.0,
            line_dash="dot",
            line_color="rgba(255,255,255,0.25)",
            row=4,
            col=1,
        )

    # Keep Plotly layout free of title overlap by placing legend cleanly in its own dedicated top band
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        height=980,
        margin=dict(l=55, r=35, t=75, b=35),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            font=dict(size=11, color="#C9D1D9"),
            bgcolor="rgba(22, 27, 34, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    for r_idx in (1, 2, 3, 4):
        fig.update_xaxes(
            type="category",
            nticks=20,
            gridcolor="rgba(255,255,255,0.06)",
            row=r_idx,
            col=1,
        )
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=r_idx, col=1)

    status_badge = (
        f"ACTIVE (Updated through {end_date})"
        if status == "ACTIVE"
        else f"EXITED & FROZEN on {exit_date_str}"
    )
    ret_color = "#10B981" if ret_pct >= 0 else "#F43F5E"
    dir_bg = "rgba(16, 185, 129, 0.16)" if direction == "LONG" else "rgba(244, 63, 94, 0.16)"
    dir_fg = "#34D399" if direction == "LONG" else "#FB7185"

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{ticker} ({company}) — Turtle Daily Chart</title>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background-color: #0E1117;
      color: #F0F6FC;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    .chart-header {{
      background: #161B22;
      border-bottom: 1px solid #263248;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 14px;
    }}
    .ticker-badge {{
      font-size: 17px;
      font-weight: 800;
      color: #60A5FA;
      font-family: Consolas, monospace;
    }}
    .company-name {{
      color: #9FB0C7;
      font-weight: 600;
    }}
    .pill {{
      padding: 3px 9px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
    }}
    .metrics-strip {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 13px;
      color: #9FB0C7;
    }}
    .metrics-strip b {{
      color: #F0F6FC;
      font-family: Consolas, monospace;
    }}
  </style>
</head>
<body>
  <div class="chart-header">
    <div class="header-left">
      <span class="ticker-badge">{ticker}</span>
      <span class="company-name">({company})</span>
      <span class="pill" style="background:{dir_bg}; color:{dir_fg};">{direction} • {system}</span>
      <span class="pill" style="background:rgba(59,130,246,0.18); color:#93C5FD;">Grade {grade} ({score:.1f}/100)</span>
    </div>
    <div class="metrics-strip">
      <span>Rec Date: <b>{rec_date}</b></span>
      <span>Recommended Price: <b>${rec_price:.2f}</b></span>
      <span>Latest/Exit Close: <b>${latest_price:.2f}</b></span>
      <span>Return: <b style="color:{ret_color};">{ret_pct:+.2f}%</b> ({days_held}d)</span>
      <span>Donchian Exit: <b style="color:#FFA726;">${cur_exit_lvl:.2f}</b></span>
      <span>Status: <b>{status_badge}</b></span>
    </div>
  </div>
  {plot_div}
</body>
</html>
"""
    chart_path.write_text(full_html, encoding="utf-8")
    return chart_path
