"""Multi-factor 5-Pillar Turtle Grading & Top-10 Ranking Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    MIN_TRADING_DAYS_REQUIRED,
    PILLAR_WEIGHTS,
    TOP_N,
    assign_letter_grade,
)


def _clip_scale(val: float, low: float, high: float, out_min: float = 0.0, out_max: float = 100.0) -> float:
    """Linearly map `val` from [low, high] into [out_min, out_max] with clamping."""
    if np.isnan(val):
        return (out_min + out_max) / 2.0
    if high == low:
        return out_max
    ratio = (val - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    return out_min + ratio * (out_max - out_min)


def score_breakout_n(breakout_n: float) -> float:
    """
    Score ATR-normalized breakout distance (|Close - Channel| / N).
    Peaks in the 0.35N - 1.50N decisive breakout zone; tapers above 2.0N to penalize blow-off gaps.
    """
    if np.isnan(breakout_n) or breakout_n <= 0.0:
        return 40.0
    if breakout_n <= 0.35:
        return _clip_scale(breakout_n, 0.0, 0.35, 55.0, 85.0)
    if breakout_n <= 1.50:
        return _clip_scale(breakout_n, 0.35, 1.50, 85.0, 100.0)
    if breakout_n <= 2.50:
        return _clip_scale(breakout_n, 1.50, 2.50, 100.0, 75.0)
    return _clip_scale(breakout_n, 2.50, 4.00, 75.0, 35.0)


def score_vol_compression(vol_comp: float) -> float:
    """Lower N20/N60 ratio means tighter volatility base prior to breakout."""
    if np.isnan(vol_comp):
        return 65.0
    if vol_comp <= 0.75:
        return 100.0
    if vol_comp <= 1.05:
        return _clip_scale(vol_comp, 0.75, 1.05, 100.0, 75.0)
    return _clip_scale(vol_comp, 1.05, 1.50, 75.0, 35.0)


def score_rsi_power_zone(dir_rsi: float) -> float:
    """
    Trend-following RSI curve (using directional RSI: RSI_14 for Long, 100 - RSI_14 for Short).
    Rewards the 56-75 trend thrust zone (peak at 67) and penalizes >80 blow-off exhaustion.
    """
    if np.isnan(dir_rsi):
        return 50.0
    if dir_rsi < 48.0:
        return _clip_scale(dir_rsi, 30.0, 48.0, 15.0, 45.0)
    if dir_rsi <= 67.0:
        return _clip_scale(dir_rsi, 48.0, 67.0, 45.0, 100.0)
    if dir_rsi <= 76.0:
        return _clip_scale(dir_rsi, 67.0, 76.0, 100.0, 88.0)
    return _clip_scale(dir_rsi, 76.0, 90.0, 88.0, 30.0)


def score_stoch_thrust(dir_k: float, dir_spread: float) -> float:
    """
    Trend-following Stochastic score combining directional %K sweet-spot (60-88)
    and positive %K - %D momentum spread.
    """
    if np.isnan(dir_k):
        k_score = 50.0
    elif dir_k < 50.0:
        k_score = _clip_scale(dir_k, 20.0, 50.0, 20.0, 55.0)
    elif dir_k <= 86.0:
        k_score = _clip_scale(dir_k, 50.0, 86.0, 55.0, 100.0)
    else:
        k_score = _clip_scale(dir_k, 86.0, 98.0, 100.0, 65.0)

    spread_score = _clip_scale(dir_spread, -8.0, 12.0, 25.0, 100.0)
    return 0.65 * k_score + 0.35 * spread_score


def score_sma200_regime(dir_dist_sma200: float) -> float:
    """
    Score directional distance from 200-day SMA.
    Healthy trend alignment (+2% to +35%) scores highest; extreme extension (>60%) tapers slightly.
    """
    if np.isnan(dir_dist_sma200):
        return 55.0
    if dir_dist_sma200 < 0.0:
        return _clip_scale(dir_dist_sma200, -25.0, 0.0, 20.0, 55.0)
    if dir_dist_sma200 <= 30.0:
        return _clip_scale(dir_dist_sma200, 0.0, 30.0, 55.0, 100.0)
    return _clip_scale(dir_dist_sma200, 30.0, 75.0, 100.0, 70.0)


def compute_candidate_pillars(row: Dict[str, Any], direction: str) -> Dict[str, float]:
    """Compute the 5 calibrated pillar domain scores (0-100) for a qualified candidate."""
    sign = 1.0 if direction == "LONG" else -1.0

    # Pillar 1: Breakout & ATR Structure (25%)
    s_brk = score_breakout_n(float(row["breakout_n"]))
    s_vcomp = score_vol_compression(float(row["vol_compression"]))
    s_sys = 100.0 if "System 2" in str(row["system"]) else 68.0
    p1_breakout_atr = 0.45 * s_brk + 0.30 * s_vcomp + 0.25 * s_sys

    # Pillar 2: Multi-Horizon Momentum & MACD (25%)
    dir_vol_mom = sign * float(row["vol_adj_mom"])
    dir_macd_hist = sign * float(row["macd_hist_n"])
    dir_macd_accel = sign * float(row["macd_accel_n"])
    s_mom = _clip_scale(dir_vol_mom, -2.0, 8.0, 20.0, 100.0)
    s_macd_h = _clip_scale(dir_macd_hist, -0.25, 0.65, 20.0, 100.0)
    s_macd_a = _clip_scale(dir_macd_accel, -0.20, 0.45, 25.0, 100.0)
    p2_momentum_macd = 0.50 * s_mom + 0.30 * s_macd_h + 0.20 * s_macd_a

    # Pillar 3: Hull MA (20) & Macro Trend (20%)
    dir_hma_slope = sign * float(row["hma_slope_n"])
    dir_price_hma = sign * float(row["price_to_hma_n"])
    dir_sma200 = sign * float(row["dist_sma200_pct"])
    s_hma_slope = _clip_scale(dir_hma_slope, -0.3, 1.5, 15.0, 100.0)
    if dir_price_hma <= 2.2:
        s_price_hma = _clip_scale(dir_price_hma, -0.5, 2.2, 25.0, 100.0)
    else:
        s_price_hma = _clip_scale(dir_price_hma, 2.2, 4.5, 100.0, 60.0)
    s_sma200 = score_sma200_regime(dir_sma200)
    p3_hma_trend = 0.45 * s_hma_slope + 0.25 * s_price_hma + 0.30 * s_sma200

    # Pillar 4: Oscillator Power-Zone (RSI 14 + Stochastic 14,3,3) (15%)
    raw_rsi = float(row["rsi_14"])
    dir_rsi = raw_rsi if direction == "LONG" else (100.0 - raw_rsi)
    raw_k = float(row["stoch_k"])
    dir_k = raw_k if direction == "LONG" else (100.0 - raw_k)
    dir_stoch_spread = sign * float(row["stoch_spread"])
    s_rsi = score_rsi_power_zone(dir_rsi)
    s_stoch = score_stoch_thrust(dir_k, dir_stoch_spread)
    p4_oscillator_zone = 0.55 * s_rsi + 0.45 * s_stoch

    # Pillar 5: Institutional Volume & ADX Conviction (15%)
    rvol = float(row["rvol"])
    adx = float(row["adx_14"])
    dir_di_spread = sign * float(row["di_spread"])
    s_rvol = _clip_scale(rvol, 0.70, 2.40, 25.0, 100.0)
    s_adx = _clip_scale(adx, 14.0, 42.0, 30.0, 100.0)
    s_di = _clip_scale(dir_di_spread, -5.0, 25.0, 20.0, 100.0)
    p5_volume_adx = 0.55 * s_rvol + 0.25 * s_adx + 0.20 * s_di

    return {
        "p1_breakout_atr": round(p1_breakout_atr, 2),
        "p2_momentum_macd": round(p2_momentum_macd, 2),
        "p3_hma_trend": round(p3_hma_trend, 2),
        "p4_oscillator_zone": round(p4_oscillator_zone, 2),
        "p5_volume_adx": round(p5_volume_adx, 2),
    }


def _score_and_rank_side(candidates: List[Dict[str, Any]], direction: str, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Score all qualified candidates on one side (LONG or SHORT) using a blend of:
    - 75% Calibrated 5-Pillar Domain Score (absolute quality consistency across days)
    - 25% Cross-Sectional Percentile Rank across today's qualified peers (relative tie-breaking)
    """
    if not candidates:
        return pd.DataFrame()

    for cand in candidates:
        cand.update(compute_candidate_pillars(cand, direction))

    df = pd.DataFrame(candidates)
    pillar_cols = [
        ("p1_breakout_atr", PILLAR_WEIGHTS["breakout_atr"]),
        ("p2_momentum_macd", PILLAR_WEIGHTS["momentum_macd"]),
        ("p3_hma_trend", PILLAR_WEIGHTS["hma_trend"]),
        ("p4_oscillator_zone", PILLAR_WEIGHTS["oscillator_zone"]),
        ("p5_volume_adx", PILLAR_WEIGHTS["volume_adx"]),
    ]

    domain_composite = pd.Series(0.0, index=df.index)
    pct_composite = pd.Series(0.0, index=df.index)

    n_cands = len(df)
    for col, weight in pillar_cols:
        domain_composite += weight * df[col]
        if n_cands > 1:
            pct_rank = df[col].rank(pct=True, method="average") * 100.0
        else:
            pct_rank = df[col]
        pct_composite += weight * pct_rank

    # Blend 75% calibrated domain score + 25% cross-sectional percentile rank
    df["turtle_score"] = (0.75 * domain_composite + 0.25 * pct_composite).round(2)
    df["grade"] = df["turtle_score"].apply(assign_letter_grade)

    df = df.sort_values(by=["turtle_score", "rvol", "breakout_n"], ascending=[False, False, False])
    return df.head(top_n).reset_index(drop=True)


def screen_and_rank_universe(
    indicator_frames: Dict[str, pd.DataFrame],
    ticker_meta: Dict[str, Dict[str, str]],
    scan_date: Optional[str] = None,
    active_long_tickers: Optional[Set[str]] = None,
    active_short_tickers: Optional[Set[str]] = None,
    top_n: int = TOP_N,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Evaluate all tickers on `scan_date` (or their latest available trading date),
    filter for fresh Day-1 Turtle breakouts, exclude already-ACTIVE tickers in the same direction,
    and return (top_longs_df, top_shorts_df, effective_scan_date).
    """
    active_longs = active_long_tickers or set()
    active_shorts = active_short_tickers or set()

    # Determine effective scan date if not provided
    if scan_date is None:
        latest_dates = [
            str(df.index[-1])
            for df in indicator_frames.values()
            if df is not None and len(df) >= MIN_TRADING_DAYS_REQUIRED
        ]
        if not latest_dates:
            return pd.DataFrame(), pd.DataFrame(), ""
        # Use the mode (most common latest trading date across the universe)
        effective_date = pd.Series(latest_dates).mode().iloc[0]
    else:
        effective_date = scan_date

    long_candidates: List[Dict[str, Any]] = []
    short_candidates: List[Dict[str, Any]] = []

    for tkr, df in indicator_frames.items():
        if df is None or len(df) < MIN_TRADING_DAYS_REQUIRED:
            continue
        if effective_date not in df.index:
            continue

        row = df.loc[effective_date]
        # Handle potential duplicate index edge-case
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        close_val = float(row["close"])
        atr_n = float(row["atr_n"])
        if np.isnan(close_val) or np.isnan(atr_n) or atr_n <= 0.0:
            continue

        meta = ticker_meta.get(tkr, {"company_name": tkr, "sector": "Unknown"})

        # Check Fresh Day-1 Long Qualification (System 2 55d takes priority over System 1 20d)
        if tkr not in active_longs:
            is_fresh_l2 = bool(row.get("fresh_long_s2", False))
            is_fresh_l1 = bool(row.get("fresh_long_s1", False))
            if is_fresh_l2 or is_fresh_l1:
                if is_fresh_l2:
                    system = "System 2 (55d)"
                    brk_lvl = float(row["high_55"])
                    exit_lvl = float(row["low_20"])
                else:
                    system = "System 1 (20d)"
                    brk_lvl = float(row["high_20"])
                    exit_lvl = float(row["low_10"])

                if not np.isnan(brk_lvl) and close_val > brk_lvl:
                    breakout_n = (close_val - brk_lvl) / atr_n
                    long_candidates.append(
                        _build_candidate_dict(
                            tkr=tkr,
                            meta=meta,
                            scan_date=effective_date,
                            direction="LONG",
                            system=system,
                            brk_lvl=brk_lvl,
                            exit_lvl=exit_lvl,
                            breakout_n=breakout_n,
                            row=row,
                        )
                    )

        # Check Fresh Day-1 Short Qualification (System 2 55d takes priority over System 1 20d)
        if tkr not in active_shorts:
            is_fresh_s2 = bool(row.get("fresh_short_s2", False))
            is_fresh_s1 = bool(row.get("fresh_short_s1", False))
            if is_fresh_s2 or is_fresh_s1:
                if is_fresh_s2:
                    system = "System 2 (55d)"
                    brk_lvl = float(row["low_55"])
                    exit_lvl = float(row["high_20"])
                else:
                    system = "System 1 (20d)"
                    brk_lvl = float(row["low_20"])
                    exit_lvl = float(row["high_10"])

                if not np.isnan(brk_lvl) and close_val < brk_lvl:
                    breakout_n = (brk_lvl - close_val) / atr_n
                    short_candidates.append(
                        _build_candidate_dict(
                            tkr=tkr,
                            meta=meta,
                            scan_date=effective_date,
                            direction="SHORT",
                            system=system,
                            brk_lvl=brk_lvl,
                            exit_lvl=exit_lvl,
                            breakout_n=breakout_n,
                            row=row,
                        )
                    )

    top_longs = _score_and_rank_side(long_candidates, "LONG", top_n=top_n)
    top_shorts = _score_and_rank_side(short_candidates, "SHORT", top_n=top_n)
    return top_longs, top_shorts, effective_date


def _build_candidate_dict(
    tkr: str,
    meta: Dict[str, str],
    scan_date: str,
    direction: str,
    system: str,
    brk_lvl: float,
    exit_lvl: float,
    breakout_n: float,
    row: pd.Series,
) -> Dict[str, Any]:
    """Assemble standardized indicator snapshot dictionary for a breakout candidate."""
    def _safe(key: str, default: float = 0.0) -> float:
        val = float(row.get(key, default))
        return default if np.isnan(val) else val

    return {
        "rec_id": f"{tkr}_{scan_date}_{direction}",
        "rec_date": scan_date,
        "ticker": tkr,
        "company_name": meta.get("company_name", tkr),
        "sector": meta.get("sector", "Unknown"),
        "direction": direction,
        "system": system,
        "upgrade_note": "",
        "rec_price": round(_safe("close"), 2),
        "breakout_level": round(brk_lvl, 2),
        "initial_exit_level": round(exit_lvl, 2),
        "atr_n": round(_safe("atr_n"), 4),
        "atr_pct": round(_safe("atr_pct"), 2),
        "vol_compression": round(_safe("vol_compression", 1.0), 3),
        "breakout_n": round(breakout_n, 2),
        "hma_20": round(_safe("hma_20"), 2),
        "hma_slope_n": round(_safe("hma_slope_n"), 2),
        "price_to_hma_n": round(_safe("price_to_hma_n"), 2),
        "rsi_14": round(_safe("rsi_14", 50.0), 1),
        "macd_hist_n": round(_safe("macd_hist_n"), 3),
        "macd_accel_n": round(_safe("macd_accel_n"), 3),
        "stoch_k": round(_safe("stoch_k", 50.0), 1),
        "stoch_d": round(_safe("stoch_d", 50.0), 1),
        "stoch_spread": round(_safe("stoch_spread", 0.0), 1),
        "adx_14": round(_safe("adx_14", 20.0), 1),
        "di_spread": round(_safe("di_spread", 0.0), 1),
        "rvol": round(_safe("rvol", 1.0), 2),
        "roc_60": round(_safe("roc_60", 0.0), 2),
        "vol_adj_mom": round(_safe("vol_adj_mom", 0.0), 2),
        "dist_sma200_pct": round(_safe("dist_sma200_pct", 0.0), 2),
    }
