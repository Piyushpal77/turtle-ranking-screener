"""Configuration parameters for the Turtle Trend Following US Stock Ranking Screener."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DAILY_CSV_DIR = OUTPUT_DIR / "daily_csv"
CHARTS_DIR = OUTPUT_DIR / "charts"

DB_PATH = DATA_DIR / "screener.db"
UNIVERSE_CACHE_PATH = DATA_DIR / "universe_cache.json"
CUSTOM_TICKERS_PATH = BASE_DIR / "custom_tickers.txt"
TRACKER_XLSX_PATH = OUTPUT_DIR / "turtle_recommendations_tracker.xlsx"

# Ensure directories exist
for directory in (DATA_DIR, OUTPUT_DIR, DAILY_CSV_DIR, CHARTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Screener ranking size
TOP_N = 10

# Historical data settings
INITIAL_LOOKBACK_PERIOD = "2y"  # Full warmup for 200d SMA, 60d ROC, 55d Donchian
INCREMENTAL_OVERLAP_DAYS = 7    # Calendar days overlap to verify split/dividend consistency
SPLIT_TOLERANCE_PCT = 0.001     # 0.1% divergence on overlap triggers full history refresh
MIN_TRADING_DAYS_REQUIRED = 65  # Minimum history required to compute 55d Donchian & 60d ROC

# Universe cache expiry
UNIVERSE_CACHE_HOURS = 24

# Donchian & Turtle parameters
SYSTEM1_ENTRY_WINDOW = 20
SYSTEM2_ENTRY_WINDOW = 55
SYSTEM1_EXIT_WINDOW = 10
SYSTEM2_EXIT_WINDOW = 20

# Technical indicator windows
ATR_WINDOW = 20
ATR_BASE_WINDOW = 60
HMA_WINDOW = 20
HMA_SLOPE_LAG = 5
RSI_WINDOW = 14
STOCH_K_WINDOW = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_ACCEL_LAG = 3
ADX_WINDOW = 14
RVOL_WINDOW = 20
ROC_WINDOW = 60
SMA_REGIME_WINDOW = 200

# 5-Pillar Composite Weights (must sum to 1.0)
PILLAR_WEIGHTS = {
    "breakout_atr": 0.25,     # Breakout_N impulse, base vol compression (N20/N60), System 2 bonus
    "momentum_macd": 0.25,    # Vol-adjusted 60d ROC + N-normalized MACD histogram & acceleration
    "hma_trend": 0.20,        # Hull MA (20) 5-day N-slope, Price-to-HMA alignment, SMA200 regime
    "oscillator_zone": 0.15,  # Trend-following RSI(14) power-zone curve + Stochastic(14,3,3) thrust
    "volume_adx": 0.15,       # Relative Volume (RVOL) surge + ADX(14) & directional DI spread
}

# Letter Grade Thresholds (0-100 scale)
GRADE_THRESHOLDS = [
    (85.0, "A+"),
    (75.0, "A"),
    (65.0, "B+"),
    (55.0, "B"),
    (0.0, "C"),
]

# Charting parameters
CHART_LOOKBACK_BARS_BEFORE_REC = 90


def assign_letter_grade(score: float) -> str:
    """Map a 0-100 composite Turtle Score to a letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "C"
