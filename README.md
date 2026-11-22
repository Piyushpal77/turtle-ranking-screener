# Turtle Trend Following US Stock Ranking Screener

A modular Python stock screener and in-system tracking dashboard inspired by the **Turtle Trading System**. Scans the **S&P 500** and **Nasdaq 100** (~515+ liquid US equities) daily for fresh **Day-1 Donchian Channel breakouts**, grades every candidate on a **0–100 Multi-Factor Turtle Score (`A+` to `C`)**, stores daily OHLCV bars in a **local SQLite database (`data/screener.db`)** for fast incremental EOD updates, and generates a **self-contained Web UI (`output/dashboard.html`)** featuring built-in **Long Trades Spreadsheet** and **Short Trades Spreadsheet** tabs where every recommended stock is **hyperlinked to its interactive daily HTML candlestick chart**.

---

## Key Features

* **Fresh Day-1 Closing Breakouts (Zero Look-Ahead Bias):**
  * **System 1 (20-Day):** Daily `Close` crosses above the prior 20-day High (Long) or below the prior 20-day Low (Short).
  * **System 2 (55-Day):** Daily `Close` crosses above the prior 55-day High (Long) or below the prior 55-day Low (Short).
  * Tickers that already have an `ACTIVE` (un-exited) recommendation in the same direction are skipped to reserve Top 10 slots for fresh breakouts (while logging `S1 -> S2 Upgrade` on the existing active tracker row).
* **5-Pillar Multi-Indicator Grading (`0–100` Score & `A+` to `C` Grade):**
  1. **Breakout & ATR Structure (25%):** ATR-normalized breakout distance (`Breakout_N`, penalizing overextended $>2.5N$ blow-off gaps), base volatility compression ($N_{20}/N_{60}$), and System 2 (55d) priority bonus.
  2. **Multi-Horizon Momentum & MACD (25%):** Volatility-adjusted 60-day momentum ($\text{ROC}_{60} / \text{ATR}\%$) + $N$-normalized **MACD(12, 26, 9) Histogram** level and 3-day acceleration.
  3. **Hull Moving Average & Macro Trend (20%):** 20-period **Hull Moving Average ($\text{HMA}_{20}$)** $N$-normalized 5-day slope, price-to-HMA alignment, and 200-day SMA regime alignment.
  4. **Oscillator Power-Zone (15%):** Trend-following **RSI(14)** thrust curve (peaks in the $56\text{–}76$ Long / $24\text{–}44$ Short power zone while penalizing $>80$ / $<20$ exhaustion) + **Slow Stochastic(14, 3, 3)** $\%K$ thrust and $\%K - \%D$ spread.
  5. **Institutional Volume & ADX Conviction (15%):** Relative Volume (**RVOL**) surge + Wilder's **ADX(14)** and $\pm(\text{+DI} - \text{-DI})$ directional spread.
* **SQLite Incremental EOD Database (`data/screener.db`):**
  * Downloads 2 years of adjusted daily OHLCV bars on first run, then downloads only incremental EOD bars each day with a 7-day overlap check that automatically detects overnight stock splits/dividends and refreshes affected tickers.
* **Built-In Web UI Spreadsheets & Hyperlinked Interactive HTML Charts (`output/dashboard.html`):**
  * Dedicated **`Long Trades Spreadsheet`** and **`Short Trades Spreadsheet`** tabs built directly into the UI, ranked in decreasing order of **Turtle Score** with headings: `Security Name`, `Date-Wise Closing Price`, `Turtle Score`, `Recommended Price`, `Donchian Exit`, and `% Return Since Recommendation`.
  * Clicking any `Security Name` opens its interactive Plotly daily chart (`output/charts/<TICKER>_<REC_DATE>_<DIRECTION>.html`) featuring candlesticks, Donchian channels, Donchian Exit channel trail, Hull MA (20), SMA (200), a **vertical dotted line on the recommendation date**, a **horizontal line at the recommended price**, Volume, MACD, and RSI/Stochastic sub-charts.
  * **Automatic Donchian Exit Freeze:** Each daily run updates `ACTIVE` recommendations until `Close` breaches the System-matched Donchian Exit (`10-day` opposite channel for System 1; `20-day` opposite channel for System 2), at which point the recommendation's return and chart freeze as `EXITED`.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Daily EOD Screener

```bash
python screener.py
```

### 3. Open the Web UI Dashboard & Spreadsheets

Open `output/dashboard.html` directly in your browser, or start the live local UI server:

```bash
python app.py
```
