# Turtle Trend Following US Stock Ranking Screener — Implementation Plan

## 1. Objective & Core Specifications

Build a modular, production-ready Python US stock screener and tracking system based on the **Turtle Trading System** to identify, grade, and track the **Top 10 Long** and **Top 10 Short** breakout candidates daily.

* **Scope:** Pure market screening, multi-indicator grading, SQL EOD data persistence, Excel recommendation tracking, and interactive daily chart generation (no position sizing, unit limits, or order execution).
* **Universe:** Deduplicated constituents of the **S&P 500** and **Nasdaq 100** (~515–520 unique liquid US equities) plus optional custom symbols via `custom_tickers.txt`.
* **Breakout Qualification (Fresh Day-1 Closing Basis):**
  * **Long Candidates:** Today's `Close > prior 20-day High` (**System 1**) or `Close > prior 55-day High` (**System 2**), **AND** yesterday's close was *not* already breaking that respective channel (`Close[t-1] <= High_K[t-1]`).
  * **Short Candidates:** Today's `Close < prior 20-day Low` (**System 1**) or `Close < prior 55-day Low` (**System 2**), **AND** yesterday's close was *not* already breaking that respective channel (`Close[t-1] >= Low_K[t-1]`).
  * **Active Deduplication:** Tickers that already have an `ACTIVE` (un-exited) recommendation in the same direction in the tracker are excluded from taking a new Top 10 slot. If an active `System 1 (20d)` recommendation later breaks its 55-day channel while still active, its existing tracker row is annotated with an `S1 -> S2 Upgrade` note.
* **Exit & Freeze Rule (Classic System-Matched Donchian Exit on Close):**
  * **System 1 (20d Breakout):** Remains `ACTIVE` until `Close < prior 10-day Low` (Long) or `Close > prior 10-day High` (Short).
  * **System 2 (55d Breakout):** Remains `ACTIVE` until `Close < prior 20-day Low` (Long) or `Close > prior 20-day High` (Short).
  * Once a recommendation breaches its Turtle Exit channel on a closing basis, its status flips to `EXITED`, and both its tracker metrics (Exit Date, Exit Price, Final Return %, Holding Days) and its interactive HTML chart are **frozen** at the exit date.

---

## 2. Project Directory Structure

```text
turtle-ranking-screener/
├── turtle_screener_implementation_plan.md  # Architectural & implementation blueprint
├── requirements.txt                        # Python dependencies
├── config.py                               # Weights, grading thresholds, lookbacks, and paths
├── custom_tickers.txt                      # Optional user-specified tickers to include in universe
├── database.py                             # SQLite schema, incremental EOD sync & split detection
├── universe.py                             # S&P 500 & Nasdaq 100 scraper + ticker normalizer
├── indicators.py                           # Donchian, Wilder ATR (N), Hull MA, MACD, RSI, Stoch, ADX, RVOL
├── ranker.py                               # 5-pillar scoring engine, letter grading (A+ to C) & Top 10 ranker
├── tracker.py                              # Lifecycle manager (Active vs Exited) & Excel (.xlsx) workbook updater
├── charting.py                             # Interactive Plotly candlestick & indicator HTML chart generator
├── screener.py                             # CLI entrypoint orchestrating sync, scan, tracking & charting
├── data/
│   ├── screener.db                         # Local SQLite database (tickers, daily_prices, recommendations)
│   └── universe_cache.json                 # 24-hour cached constituent metadata
└── output/
    ├── turtle_recommendations_tracker.xlsx # Master live-updating Excel workbook with chart hyperlinks
    ├── daily_csv/                          # Dated CSV snapshots (top10_longs_YYYY-MM-DD.csv, etc.)
    └── charts/                             # Interactive HTML daily charts (<TICKER>_<REC_DATE>.html)
```

---

## 3. SQLite Database & Incremental EOD Sync (`database.py`)

To avoid re-downloading full price histories every day, all market data and recommendation state are persisted in `data/screener.db`:

1. **Tables:**
   * `tickers`: `ticker TEXT PRIMARY KEY, company_name TEXT, sector TEXT, source_index TEXT, updated_at TEXT`
   * `daily_prices`: `ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, PRIMARY KEY (ticker, date)`
   * `recommendations`:
     * Identification: `rec_id TEXT PRIMARY KEY` (`<TICKER>_<REC_DATE>_<DIRECTION>`), `rec_date TEXT`, `ticker TEXT`, `direction TEXT` (`LONG`/`SHORT`), `system TEXT` (`System 1 (20d)` / `System 2 (55d)`), `upgrade_note TEXT`
     * Entry Snapshot: `grade TEXT`, `turtle_score REAL`, `rec_price REAL`, `breakout_level REAL`, `initial_exit_level REAL`, `atr_n REAL`, `atr_pct REAL`, `breakout_n REAL`, `hma_20 REAL`, `rsi_14 REAL`, `macd_hist REAL`, `stoch_k REAL`, `adx_14 REAL`, `rvol REAL`, `roc_60 REAL`, `dist_sma200_pct REAL`
     * Live / Frozen Tracking: `status TEXT` (`ACTIVE` / `EXITED`), `latest_date TEXT`, `latest_price REAL`, `current_exit_level REAL`, `return_pct REAL`, `max_favorable_pct REAL`, `max_adverse_pct REAL`, `days_held INTEGER`, `exit_date TEXT`, `exit_price REAL`, `chart_path TEXT`
2. **Smart Incremental EOD Sync + Split/Dividend Protection:**
   * **Cold Start:** On first run (or for any new constituent), download `2y` of daily auto-adjusted OHLCV bars (`yf.download(..., auto_adjust=True)`) to ensure full warmup for 200-day SMA, 60-day ATR/ROC, and 55-day Donchian channels.
   * **Daily Incremental Sync:** Query `MAX(date)` per ticker from `daily_prices`. Fetch only the window from `max_date - 5 trading days` through today.
   * **Split/Dividend Reconciliation:** Compare the overlapping 5 days' newly downloaded `close` prices against the stored `close` prices in SQLite. If any overlapping day diverges by `> 0.05%` (indicating an overnight stock split or dividend adjustment), automatically re-fetch and replace the full 2-year history for that specific ticker. Otherwise, `INSERT OR REPLACE` only the new EOD bars.

---

## 4. Technical Indicator Suite (`indicators.py`)

All channel and reference indicators are strictly shifted where required to guarantee **zero look-ahead bias**:

1. **Donchian Breakout & Exit Channels (Shifted by 1 Day):**
   * `High_20 = High.shift(1).rolling(20).max()` and `Low_20 = Low.shift(1).rolling(20).min()`
   * `High_55 = High.shift(1).rolling(55).max()` and `Low_55 = Low.shift(1).rolling(55).min()`
   * `Low_10 = Low.shift(1).rolling(10).min()` (System 1 Long Exit) and `High_10 = High.shift(1).rolling(10).max()` (System 1 Short Exit)
2. **Wilder's Volatility ($N$, $\text{ATR}\%$, and Volatility Compression):**
   * True Range: $TR_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$
   * Wilder 20-day ATR: $N_t = \text{EWM}(TR, \alpha = 1/20)$
   * $\text{ATR}\% = (N_t / \text{Close}_t) \times 100$
   * Volatility Compression Ratio: $\text{Vol\_Comp} = N_{20} / \text{SMA}_{60}(TR)$ (lower ratio prior to breakout signals tighter base accumulation).
3. **Hull Moving Average ($\text{HMA}_{20}$):**
   * $$\text{HMA}_{20} = \text{WMA}_{\lfloor\sqrt{20}\rfloor}\Big(2 \cdot \text{WMA}_{10}(\text{Close}) - \text{WMA}_{20}(\text{Close})\Big)$$
   * **HMA $N$-Normalized 5-Day Slope:** $\text{HMA\_Slope}_N = \frac{\text{HMA}_{20, t} - \text{HMA}_{20, t-5}}{N_t}$
   * **Price-to-HMA Alignment:** $\frac{\text{Close}_t - \text{HMA}_{20, t}}{N_t}$
4. **MACD ($12, 26, 9$):**
   * $\text{MACD Line} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})$
   * $\text{Signal Line} = \text{EMA}_{9}(\text{MACD Line})$
   * $\text{MACD\_Hist} = \text{MACD Line} - \text{Signal Line}$
   * **$N$-Normalized Histogram & 3-Day Acceleration:** $\text{MACD\_Hist}_N = \frac{\text{MACD\_Hist}_t}{N_t}$ and $\Delta \text{MACD\_Hist}_N = \frac{\text{MACD\_Hist}_t - \text{MACD\_Hist}_{t-3}}{N_t}$
5. **Wilder's RSI ($14$):**
   * Standard 14-period Wilder RSI on `Close`.
   * **Trend-Following RSI Thrust Score (`RSI_Quality`, $0\text{–}100$):**
     * In breakout trading, healthy Day-1 breakouts occur in the **power zone** ($\text{RSI} \in [55, 76]$ for Longs; $[24, 45]$ for Shorts), whereas extreme values ($\text{RSI} > 82$ Long or $< 18$ Short) signal short-term blow-off exhaustion.
     * Piece-wise bell function peaking at $\text{RSI} = 67$ for Longs (and $\text{RSI} = 33$ for Shorts), smoothly penalizing weak momentum ($<50$ Long) and overextended blow-offs ($>80$ Long).
6. **Stochastic Oscillator ($14, 3, 3$):**
   * $\text{Fast } \%K = 100 \times \frac{\text{Close}_t - \text{Low}_{14}}{\text{High}_{14} - \text{Low}_{14}}$, smoothed by 3-day SMA for $\text{Slow } \%K$ and 3-day SMA for $\%D$.
   * **Stochastic Thrust Score (`Stoch_Quality`):** Combines directional $\%K - \%D$ bull/bear spread and sweet-spot positioning ($\%K \in [60, 90]$ for Longs; $[10, 40]$ for Shorts).
7. **Momentum, Volume & Trend Strength (`ROC_60`, `RVOL`, `ADX_14`, `SMA_200`):**
   * **60-Day Rate of Change (`ROC_60`) & Volatility-Adjusted Momentum:** $\text{Vol\_Adj\_Mom} = \frac{\pm \text{ROC}_{60}}{\text{ATR}\%}$
   * **Relative Volume (`RVOL`):** $\text{Volume}_t / \text{SMA}_{20}(\text{Volume}_{t-1})$
   * **Wilder's `ADX_14`, `+DI_14`, `-DI_14`:** Measures trend strength and directional spread $\pm(\text{+DI}_{14} - \text{-DI}_{14})$.
   * **200-Day SMA Distance (`Dist_SMA200_Pct`):** $\frac{\text{Close}_t - \text{SMA}_{200}}{\text{SMA}_{200}} \times 100$

---

## 5. Research-Backed 5-Pillar Grading & Ranking Engine (`ranker.py`)

Every qualified Day-1 breakout candidate is evaluated across **5 orthogonal pillars** (combining cross-sectional percentile ranks and bounded quality curves into a `0–100` **Turtle Score**):

| Pillar | Weight | Component Indicators | Why It Improves Turtle Breakouts |
| :--- | :---: | :--- | :--- |
| **1. Breakout & ATR Structure** | **25%** | `Breakout_N` (sweet-spot curve peaking at $0.4N\text{–}1.5N$, tapering $>2.5N$ to penalize runaway gaps) + Base Volatility Compression ($N_{20}/N_{60}$) + System 2 (55d) bonus | Rewards decisive breakouts emerging from tight volatility bases while filtering out overextended gap traps. |
| **2. Multi-Horizon Momentum** | **25%** | Volatility-Adjusted 60d Momentum (`ROC_60 / ATR%`) + `MACD_Hist_N` level & 3-day histogram acceleration | Confirms both intermediate trend velocity (60d) and immediate inflection acceleration (MACD histogram expansion). |
| **3. Hull MA & Macro Trend** | **20%** | `HMA_20` $N$-normalized 5-day slope + Price-to-HMA alignment + directional `SMA_200` regime alignment | Uses zero-lag Hull Moving Average slope to verify fast structural trend support in harmony with the 200-day macro trend. |
| **4. Oscillator Power-Zone (RSI + Stochastic)** | **15%** | `RSI_14` Trend Thrust Score (peaks at 67 Long / 33 Short, penalizes $>82$ / $<18$ exhaustion) + `Stochastic (14,3,3)` $\%K$ thrust & $\%K - \%D$ spread | Ensures strong momentum confirmation without buying/shorting into extreme overbought/oversold exhaustion. |
| **5. Volume & ADX Conviction** | **15%** | Relative Volume (`RVOL`) surge + `ADX_14` level & directional $\pm(\text{+DI} - \text{-DI})$ spread | Confirms institutional volume participation and genuine directional trend regime. |

### Letter Grade Assignment
* **`A+`** (`85.0 – 100.0`): Elite multi-factor institutional breakout
* **`A`** (`75.0 – 84.9`): Strong confirmed breakout
* **`B+`** (`65.0 – 74.9`): Solid trend-aligned breakout
* **`B`** (`55.0 – 64.9`): Moderate breakout with partial oscillator/volume confirmation
* **`C`** (`< 55.0`): Low-conviction breakout

The top 10 Longs and top 10 Shorts (sorted descending by `Turtle_Score`) are selected for the day's recommendations.

---

## 6. Live-Updating Excel Tracker (`tracker.py`)

On each daily run, `tracker.py` updates all currently `ACTIVE` recommendations in SQLite before logging new picks, then exports a formatted multi-tab Excel workbook (`output/turtle_recommendations_tracker.xlsx` via `openpyxl`):

1. **Lifecycle Update of Existing `ACTIVE` Picks:**
   * For every `ACTIVE` row in `recommendations`, evaluate all trading days after `latest_date` up to the newest EOD date in `daily_prices`.
   * Step day-by-day so that if a stock breached its Turtle Exit channel (`Low_10`/`High_10` for System 1; `Low_20`/`High_20` for System 2) on an intermediate day, it is frozen on that exact breach date (`status = 'EXITED'`, `exit_date`, `exit_price = close`).
   * If still `ACTIVE`, update `latest_date`, `latest_price`, `current_exit_level`, `return_pct`, `max_favorable_pct`, `max_adverse_pct`, `days_held`, and check for `S1 -> S2 Upgrade`.
2. **Excel Workbook Tabs:**
   * **`Tracker_Log` (Master Historical & Active Tracking Sheet):**
     * Columns: `Rec_Date`, `Ticker` (**Hyperlinked to interactive daily HTML chart**), `Direction`, `System`, `Upgrade_Note`, `Grade`, `Turtle_Score`, `Status` (`ACTIVE` / `EXITED`), `Rec_Price`, `Breakout_Lvl`, `Latest_Date`, `Latest_or_Exit_Price`, `Current_Exit_Lvl`, `Return_%`, `Max_Gain_%`, `Max_DD_%`, `Days_Held`, `Exit_Date`, `ATR_%`, `Breakout_N`, `HMA20_Slope`, `RSI_14`, `MACD_Hist_N`, `Stoch_K`, `ADX_14`, `RVOL`, `ROC_60_%`, `Dist_SMA200_%`.
     * Conditional formatting for `Grade` (`A+`/`A` green, `B+`/`B` amber), `Status` (`ACTIVE` blue badge, `EXITED` gray), and `Return_%` (green positive / red negative).
   * **`Latest_Top10` (Today's Dashboard):**
     * Side-by-side or stacked clean tables for **Today's Top 10 Longs** and **Today's Top 10 Shorts** with clickable chart hyperlinks on every ticker.
   * **`Grade_Performance` (Analytics Summary):**
     * Aggregated win rate, average return, median return, average holding days, and active count broken down by `Grade` (`A+`, `A`, `B+`, `B`, `C`), `Direction` (`LONG` vs `SHORT`), and `System` (`System 1` vs `System 2`).

---

## 7. Interactive Daily HTML Chart Generator (`charting.py`)

For every new recommendation—and on every EOD run for all currently `ACTIVE` recommendations (plus one final update when a recommendation transitions to `EXITED`)—generate a standalone interactive Plotly HTML chart at `output/charts/<TICKER>_<REC_DATE>_<DIRECTION>.html`:

* **Time Window:** 90 trading days prior to `Rec_Date` through `latest_date` (or `exit_date` once frozen).
* **Main Price Panel (Row 1 - 55% height):**
  * Daily OHLC Candlesticks.
  * **Donchian Breakout Channel:** 20-day or 55-day Upper/Lower bands (shaded light channel) + **Turtle Exit Channel** (10-day or 20-day opposite exit trail in dashed orange).
  * **Overlays:** `HMA_20` (fast cyan/magenta trend curve) and `SMA_200` (macro gold line).
  * **Vertical Dotted Line on `Rec_Date`:** High-visibility vertical dotted line marking the exact day of recommendation, with a callout annotation banner (`"RECOMMENDED LONG (Grade A+, Score 88.4) on YYYY-MM-DD @ $XXX.XX"`).
  * **Horizontal Recommended Price Line:** Horizontal dashed reference line at `Rec_Price` extending from `Rec_Date` to the current/exit bar.
  * **Exit Marker (if `EXITED`):** Vertical dotted exit line & marker on `Exit_Date` showing the Turtle Exit breach price and final locked-in return.
* **Sub-Panels:**
  * **Row 2 (15% height):** Volume bars colored by up/down day + 20-day Volume SMA (`RVOL = 1.0x` baseline).
  * **Row 3 (15% height):** `MACD (12, 26, 9)` lines and color-coded Histogram.
  * **Row 4 (15% height):** `RSI (14)` and `Stochastic %K/%D` with horizontal reference bands at `20 / 50 / 80`.
* **Header HUD Summary Card:** Displays Ticker, Company Name, Sector, Recommendation Date, System, Grade, Composite Score, Recommended Price, Latest/Exit Price, Return %, Current Turtle Exit Level, and Status (`ACTIVE` vs `EXITED - FROZEN`).

---

## 8. CLI & Execution Workflow (`screener.py`)

Support clean CLI arguments for daily operations as well as historical backfill/testing:

* `python screener.py` — Runs the standard daily pipeline:
  1. Refresh universe cache (`universe.py`).
  2. Perform incremental EOD sync into `data/screener.db` with split/dividend verification (`database.py`).
  3. Compute technical indicators across the universe (`indicators.py`).
  4. Update all existing `ACTIVE` recommendations in SQLite, freezing any that breached their 10d/20d Turtle Exit channel today (`tracker.py`).
  5. Qualify today's fresh Day-1 breakouts (excluding already-active tickers), compute the 5-pillar `Turtle_Score` and `Grade`, and rank the **Top 10 Longs** and **Top 10 Shorts** (`ranker.py`).
  6. Insert today's Top 10 Longs & Shorts into SQLite, generate/update interactive HTML charts for all active & newly exited picks (`charting.py`), and write `output/turtle_recommendations_tracker.xlsx` + dated CSV files (`tracker.py`).
* `python screener.py --backfill-days 20` — Replays the daily screener chronologically over the last $N$ trading days from the SQLite database so you can immediately populate and inspect realistic historical recommendations, active progressions, frozen exits, and hyperlinked charts.
