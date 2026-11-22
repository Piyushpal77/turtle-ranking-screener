"""Technical indicator engine for Turtle Trend Following & Multi-Factor Grading."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ADX_WINDOW,
    ATR_BASE_WINDOW,
    ATR_WINDOW,
    HMA_SLOPE_LAG,
    HMA_WINDOW,
    MACD_ACCEL_LAG,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    ROC_WINDOW,
    RSI_WINDOW,
    RVOL_WINDOW,
    SMA_REGIME_WINDOW,
    STOCH_K_WINDOW,
    STOCH_SMOOTH_D,
    STOCH_SMOOTH_K,
    SYSTEM1_ENTRY_WINDOW,
    SYSTEM1_EXIT_WINDOW,
    SYSTEM2_ENTRY_WINDOW,
)


def weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Compute linearly Weighted Moving Average (WMA) over `period` bars."""
    if period <= 1:
        return series.copy()
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    return series.rolling(period, min_periods=period).apply(
        lambda x: float(np.dot(x, weights) / weight_sum), raw=True
    )


def hull_moving_average(series: pd.Series, period: int = HMA_WINDOW) -> pd.Series:
    """
    Compute Hull Moving Average (HMA):
    HMA(n) = WMA(sqrt(n), 2 * WMA(n/2, price) - WMA(n, price))
    """
    half_length = max(1, int(period // 2))
    sqrt_length = max(1, int(math.isqrt(period)))
    wma_half = weighted_moving_average(series, half_length)
    wma_full = weighted_moving_average(series, period)
    raw_hma = 2.0 * wma_half - wma_full
    return weighted_moving_average(raw_hma, sqrt_length)


def compute_rsi(close: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    """Compute Wilder's Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = STOCH_K_WINDOW,
    smooth_k: int = STOCH_SMOOTH_K,
    smooth_d: int = STOCH_SMOOTH_D,
) -> tuple[pd.Series, pd.Series]:
    """Compute Slow Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    fast_k = 100.0 * (close - lowest_low) / denom
    slow_k = fast_k.rolling(smooth_k, min_periods=1).mean().fillna(50.0)
    slow_d = slow_k.rolling(smooth_d, min_periods=1).mean().fillna(50.0)
    return slow_k, slow_d


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    tr: pd.Series,
    period: int = ADX_WINDOW,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Wilder's ADX(14), +DI(14), and -DI(14)."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )

    atr_adx = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_adx
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_adx

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx.fillna(20.0), plus_di.fillna(20.0), minus_di.fillna(20.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full Turtle & Multi-Indicator suite on a single ticker's OHLCV DataFrame.
    Guarantees zero look-ahead bias on Donchian entry/exit channels and volume baseline.
    """
    out = df.copy()
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)
    prev_close = close.shift(1)

    # 1. Wilder's True Range & ATR ('N')
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["tr"] = tr
    out["atr_n"] = tr.ewm(alpha=1.0 / ATR_WINDOW, min_periods=ATR_WINDOW, adjust=False).mean()
    out["atr_pct"] = (out["atr_n"] / close.replace(0.0, np.nan)) * 100.0
    base_tr = tr.rolling(ATR_BASE_WINDOW, min_periods=ATR_WINDOW).mean().replace(0.0, np.nan)
    out["vol_compression"] = (out["atr_n"] / base_tr).fillna(1.0)

    # 2. Shifted Donchian Channels (Strictly prior K sessions, excluding row t)
    shifted_high = high.shift(1)
    shifted_low = low.shift(1)

    out["high_20"] = shifted_high.rolling(SYSTEM1_ENTRY_WINDOW, min_periods=SYSTEM1_ENTRY_WINDOW).max()
    out["low_20"] = shifted_low.rolling(SYSTEM1_ENTRY_WINDOW, min_periods=SYSTEM1_ENTRY_WINDOW).min()
    out["high_55"] = shifted_high.rolling(SYSTEM2_ENTRY_WINDOW, min_periods=SYSTEM2_ENTRY_WINDOW).max()
    out["low_55"] = shifted_low.rolling(SYSTEM2_ENTRY_WINDOW, min_periods=SYSTEM2_ENTRY_WINDOW).min()

    # System 1 10-day Exit Channels (System 2 uses low_20 / high_20 as its 20-day exit)
    out["low_10"] = shifted_low.rolling(SYSTEM1_EXIT_WINDOW, min_periods=SYSTEM1_EXIT_WINDOW).min()
    out["high_10"] = shifted_high.rolling(SYSTEM1_EXIT_WINDOW, min_periods=SYSTEM1_EXIT_WINDOW).max()

    # 3. Fresh Day-1 Breakout Qualifications (Closing Basis)
    # A stock triggers a fresh breakout on Day t if Close[t] > High_K[t] AND Close[t-1] <= High_K[t-1]
    out["is_long_s2_raw"] = close > out["high_55"]
    out["is_long_s1_raw"] = close > out["high_20"]
    out["is_short_s2_raw"] = close < out["low_55"]
    out["is_short_s1_raw"] = close < out["low_20"]

    out["fresh_long_s2"] = out["is_long_s2_raw"] & (~out["is_long_s2_raw"].shift(1, fill_value=False))
    out["fresh_long_s1"] = out["is_long_s1_raw"] & (~out["is_long_s1_raw"].shift(1, fill_value=False))
    out["fresh_short_s2"] = out["is_short_s2_raw"] & (~out["is_short_s2_raw"].shift(1, fill_value=False))
    out["fresh_short_s1"] = out["is_short_s1_raw"] & (~out["is_short_s1_raw"].shift(1, fill_value=False))

    # 4. Hull Moving Average (HMA 20)
    out["hma_20"] = hull_moving_average(close, HMA_WINDOW)
    safe_n = out["atr_n"].replace(0.0, np.nan)
    out["hma_slope_n"] = (out["hma_20"] - out["hma_20"].shift(HMA_SLOPE_LAG)) / safe_n
    out["price_to_hma_n"] = (close - out["hma_20"]) / safe_n

    # 5. MACD (12, 26, 9)
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd_line"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd_line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    out["macd_hist_n"] = out["macd_hist"] / safe_n
    out["macd_accel_n"] = (out["macd_hist"] - out["macd_hist"].shift(MACD_ACCEL_LAG)) / safe_n

    # 6. Wilder's RSI (14) & Slow Stochastic (14, 3, 3)
    out["rsi_14"] = compute_rsi(close, RSI_WINDOW)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    out["stoch_spread"] = stoch_k - stoch_d

    # 7. Wilder's ADX (14) & Directional Indicators
    adx, plus_di, minus_di = compute_adx(high, low, tr, ADX_WINDOW)
    out["adx_14"] = adx
    out["plus_di_14"] = plus_di
    out["minus_di_14"] = minus_di
    out["di_spread"] = plus_di - minus_di

    # 8. Relative Volume (RVOL), 60-Day Momentum (ROC_60), and 200-Day SMA
    vol_sma_20 = volume.shift(1).rolling(RVOL_WINDOW, min_periods=10).mean().replace(0.0, np.nan)
    out["vol_sma_20"] = vol_sma_20
    out["rvol"] = (volume / vol_sma_20).fillna(1.0)

    out["roc_60"] = ((close - close.shift(ROC_WINDOW)) / close.shift(ROC_WINDOW).replace(0.0, np.nan)) * 100.0
    out["vol_adj_mom"] = out["roc_60"] / out["atr_pct"].replace(0.0, np.nan)

    out["sma_200"] = close.rolling(SMA_REGIME_WINDOW, min_periods=50).mean()
    out["dist_sma200_pct"] = ((close - out["sma_200"]) / out["sma_200"].replace(0.0, np.nan)) * 100.0

    return out
