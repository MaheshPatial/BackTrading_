import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


# ---------------------------
# Data loading and transforms
# pip install pandas numpy yfinance matplotlib
# ---------------------------

def load_daily(symbol: str, start: str = "1990-01-01", csv_path: Optional[str] = None,
               auto_adjust: bool = True) -> pd.DataFrame:
    """Load daily OHLCV for a symbol.

    Priority: CSV if provided; otherwise yfinance if available.
    CSV must have columns: Date, Open, High, Low, Close[, Volume].
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        cols = {c.lower(): c for c in df.columns}
        # Normalize case
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                lc = col.lower()
                if lc in cols:
                    df[col] = df[cols[lc]]
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if yf is None:
        raise RuntimeError(
            "yfinance not available. Provide --csv PATH to a local OHLCV file.")

    data = yf.download(symbol, start=start, interval="1d", auto_adjust=auto_adjust,
                       progress=False, threads=True)
    if data is None or len(data) == 0:
        raise RuntimeError(f"No data returned for {symbol}")

    # yfinance returns MultiIndex columns sometimes; normalize
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.rename(columns={
        "Adj Close": "Close",
    })
    # Ensure required columns
    for col in ["Open", "High", "Low", "Close"]:
        if col not in data.columns:
            raise RuntimeError(f"Missing column {col} in downloaded data")
    if "Volume" not in data.columns:
        data["Volume"] = np.nan
    data.index = pd.to_datetime(data.index)
    return data[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def to_weekly(daily: pd.DataFrame, week_ending: str = "W-FRI") -> pd.DataFrame:
    """Aggregate daily OHLCV to weekly bars ending on the given weekday.

    week_ending: pandas offset alias like 'W-FRI' or 'W-SUN'.
    """
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    w = daily.resample(week_ending).agg(ohlc).dropna(subset=["Open", "High", "Low", "Close"])  # type: ignore
    return w


def _weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Linear weighted moving average (highest weight to most recent)."""
    weights = np.arange(1, window + 1)
    def _wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window, min_periods=window).apply(_wma, raw=True)


def add_ma200w(weekly: pd.DataFrame, ma_type: Literal["sma", "wma"] = "sma") -> pd.DataFrame:
    w = weekly.copy()
    if ma_type == "wma":
        w["WMA200W"] = _weighted_moving_average(w["Close"], 200)
        w["MA200W"] = w["WMA200W"]
    else:
        w["SMA200W"] = w["Close"].rolling(200, min_periods=200).mean()
        w["MA200W"] = w["SMA200W"]
    w["MA200W_prev"] = w["MA200W"].shift(1)
    w["rising"] = w["MA200W"] > w["MA200W_prev"]
    # Also add a 20-week SMA for trailing exit
    w["SMA20W"] = w["Close"].rolling(20, min_periods=20).mean()
    return w


# ---------------------------
# Live proximity helper (optional)
# ---------------------------

def _yf_last_price(symbol: str) -> float:
    """Fetch a recent last price from yfinance (best-effort).

    Tries fast_info, then intraday history, then daily close.
    Note: yfinance can be delayed; for true real-time use paid feeds.
    """
    if yf is None:
        raise RuntimeError("yfinance not available for live check")
    t = yf.Ticker(symbol)
    # 1) fast_info
    price = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            for k in ("last_price", "lastPrice", "regular_market_price", "regularMarketPrice"):
                v = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    price = float(v)
                    break
    except Exception:
        pass
    # 2) recent intraday
    if price is None:
        try:
            h = t.history(period="7d", interval="1m")
            if h is None or h.empty:
                h = t.history(period="5d", interval="5m")
            if h is not None and not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
    # 3) daily fallback
    if price is None:
        try:
            d = t.history(period="5d", interval="1d")
            if d is not None and not d.empty:
                price = float(d["Close"].dropna().iloc[-1])
        except Exception:
            pass
    if price is None:
        raise RuntimeError("Unable to fetch last price from yfinance")
    return price


# ---------------------------
# Signals and backtest logic
# ---------------------------

def compute_events(weekly: pd.DataFrame, thresholds: List[int] = [5, 4, 3, 2, 1]) -> pd.DataFrame:
    """Compute threshold approach events and touch events.

    Event logic (per threshold L):
    - WMA must be rising this week
    - From above: last week's close > last week's WMA
    - This week's low <= WMA * (1 + L/100)
    - Last week's close > last week's WMA * (1 + L/100) (i.e., entered the band this week)

    Touch event: this week's low <= this week's WMA.
    """
    w = weekly.copy()
    w["from_above"] = w["Close"].shift(1) > w["MA200W_prev"]
    for L in thresholds:
        cur_thr = w["MA200W"] * (1 + L / 100.0)
        prev_thr = w["MA200W_prev"] * (1 + L / 100.0)
        name = f"evt_{L}pct"
        w[name] = (
            w["rising"]
            & w["from_above"]
            & (w["Low"] <= cur_thr)
            & (w["Close"].shift(1) > prev_thr)
        )
    w["evt_touch"] = (
        w["rising"] & w["from_above"] & (w["Low"] <= w["MA200W"]))
    # The overall start of a setup is first entry into <=5% band
    w["setup_start"] = w.get("evt_5pct", pd.Series(False, index=w.index))
    return w


@dataclass
class Instance:
    symbol: str
    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    outcome: str  # 'stop', 'profit', 'trail', 'open'
    weeks: Optional[int]
    avg_entry: Optional[float]
    exit_price: Optional[float]
    return_pct: Optional[float]
    alloc_filled: float


def backtest_trades(symbol: str, w: pd.DataFrame, exit_mode: Literal["none", "profit", "trail"] = "none",
                    profit_target_pct: float = 0.15) -> pd.DataFrame:
    """Simulate scale-in entries and exits.

    Entries:
      - 3%: +7.5%
      - 2%: +10%
      - 1%: +12.5%
      - touch (<= MA): +70%

    Exit:
      - stop: weekly close < MA200W
      - profit: close >= avg_entry * (1 + profit_target_pct)
      - trail: close < SMA20W

    Trade starts when 3% event first triggers; 5%/4% are alerts only.
    Additional scale-ins can fill in the same week if threshold(s) are crossed.
    """
    alloc_schedule = [
        (3, 0.075),
        (2, 0.10),
        (1, 0.125),
    ]

    instances: List[Instance] = []
    in_trade = False
    start_ts: Optional[pd.Timestamp] = None
    start_idx: Optional[int] = None
    alloc_filled = 0.0
    avg_entry = 0.0
    filled_levels = set()

    def apply_fill(price: float, weight: float):
        nonlocal avg_entry, alloc_filled
        if weight <= 0:
            return
        if alloc_filled == 0:
            avg_entry = price
            alloc_filled = weight
        else:
            avg_entry = (avg_entry * alloc_filled + price * weight) / (alloc_filled + weight)
            alloc_filled += weight

    for i, (ts, row) in enumerate(w.iterrows()):
        ma = row["MA200W"]
        close = row["Close"]
        low = row["Low"]

        # Determine which threshold levels were reached this week
        level_reached = {
            5: low <= ma * 1.05,
            4: low <= ma * 1.04,
            3: low <= ma * 1.03,
            2: low <= ma * 1.02,
            1: low <= ma * 1.01,
            0: low <= ma,
        }

        if not in_trade:
            # Start only when 3% is reached from above and rising
            if bool(row.get("rising", False)) and bool(row.get("from_above", False)) and level_reached[3]:
                in_trade = True
                start_ts = ts
                start_idx = i
                alloc_filled = 0.0
                avg_entry = 0.0
                filled_levels = set()
                # Fill levels this week in order: 3,2,1, touch
                for L, wt in alloc_schedule:
                    if level_reached[L]:
                        price = ma * (1 + L / 100.0)
                        apply_fill(price, wt)
                        filled_levels.add(L)
                if level_reached[0]:  # touch
                    apply_fill(ma, max(0.0, 1.0 - alloc_filled))
                    filled_levels.add(0)
        else:
            # In trade: add any additional fills this week not yet filled
            for L, wt in alloc_schedule:
                if L not in filled_levels and level_reached[L]:
                    price = ma * (1 + L / 100.0)
                    apply_fill(price, wt)
                    filled_levels.add(L)
            if 0 not in filled_levels and level_reached[0]:
                apply_fill(ma, max(0.0, 1.0 - alloc_filled))
                filled_levels.add(0)

            # Evaluate exits (weekly close-based)
            exit_price: Optional[float] = None
            outcome: Optional[str] = None

            # Stop takes precedence if it happens this week
            if close < ma:
                exit_price = float(close)
                outcome = "stop"
            else:
                if exit_mode == "profit" and alloc_filled > 0:
                    target = avg_entry * (1.0 + profit_target_pct)
                    if close >= target:
                        exit_price = float(close)
                        outcome = "profit"
                elif exit_mode == "trail":
                    sma20 = row.get("SMA20W", np.nan)
                    if not np.isnan(sma20) and close < float(sma20):
                        exit_price = float(close)
                        outcome = "trail"

            if outcome is not None:
                weeks = None
                if start_idx is not None:
                    weeks = i - start_idx
                ret = None
                if alloc_filled > 0 and exit_price is not None:
                    ret = (exit_price - avg_entry) / avg_entry
                instances.append(Instance(
                    symbol=symbol,
                    start=start_ts,
                    end=ts,
                    outcome=outcome,
                    weeks=weeks,
                    avg_entry=avg_entry if alloc_filled > 0 else None,
                    exit_price=exit_price,
                    return_pct=ret,
                    alloc_filled=alloc_filled,
                ))
                in_trade = False
                start_ts = None
                start_idx = None
                alloc_filled = 0.0
                avg_entry = 0.0
                filled_levels = set()

    # Open trade at end
    if in_trade:
        instances.append(Instance(
            symbol=symbol,
            start=start_ts,
            end=None,
            outcome="open",
            weeks=None,
            avg_entry=avg_entry if alloc_filled > 0 else None,
            exit_price=None,
            return_pct=None,
            alloc_filled=alloc_filled,
        ))

    df = pd.DataFrame([{
        "symbol": x.symbol,
        "start": x.start,
        "end": x.end,
        "outcome": x.outcome,
        "weeks": x.weeks,
        "avg_entry": x.avg_entry,
        "exit_price": x.exit_price,
        "return_pct": x.return_pct,
        "alloc_filled": x.alloc_filled,
    } for x in instances])
    return df


def summarize_instances(instances: pd.DataFrame) -> dict:
    total = int(len(instances))
    wins = int(instances[instances["outcome"].isin(["profit", "trail"])].shape[0]) if total else 0
    stops = int(instances[instances["outcome"] == "stop"].shape[0]) if total else 0
    opens = int(instances[instances["outcome"] == "open"].shape[0]) if total else 0
    avg_ret = float(instances["return_pct"].dropna().mean()) if total else 0.0
    return {
        "total_trades": total,
        "wins": wins,
        "stops": stops,
        "open": opens,
        "win_rate": (wins / total) if total else 0.0,
        "avg_return": avg_ret,
    }


def maybe_plot(symbol: str, w: pd.DataFrame, instances: pd.DataFrame, out_dir: str, ma_type: str) -> Optional[str]:
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(w.index, w["Close"], label="Close", color="black", linewidth=1)
    label = "200W SMA" if ma_type == "sma" else "200W WMA"
    ax.plot(w.index, w["MA200W"], label=label, color="purple", linewidth=1.5)

    # Mark starts and stops
    starts = instances["start"].dropna().values if len(instances) else []
    ends = instances["end"].dropna().values if len(instances) else []
    if len(starts):
        ax.scatter(starts, w.loc[starts, "Close"], marker="^", color="green", label="Entry start", zorder=3)
    if len(ends):
        ax.scatter(ends, w.loc[ends, "Close"], marker="v", color="red", label="Exit", zorder=3)

    ax.set_title(f"{symbol} — Weekly Close and 200W SMA")
    ax.legend()
    ax.grid(True, alpha=0.2)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"plot_{symbol}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(symbol: str, start: str, csv: Optional[str], out_dir: str, week_ending: str = "W-FRI",
        plot: bool = True, ma_type: Literal["sma", "wma"] = "sma",
        exit_mode: Literal["none", "profit", "trail"] = "none", profit_target_pct: float = 0.15):
    daily = load_daily(symbol, start=start, csv_path=csv)
    weekly = to_weekly(daily, week_ending=week_ending)
    weekly = add_ma200w(weekly, ma_type=ma_type)
    weekly = compute_events(weekly)

    # Drop periods before we have 200W data
    weekly = weekly.dropna(subset=["MA200W", "MA200W_prev"]).copy()
    if weekly.empty:
        raise RuntimeError("Not enough data to compute 200-week SMA.")

    instances = backtest_trades(symbol, weekly, exit_mode=exit_mode, profit_target_pct=profit_target_pct)
    summary = summarize_instances(instances)

    # Outputs
    os.makedirs(out_dir, exist_ok=True)
    events_path = os.path.join(out_dir, f"events_{symbol}.csv")
    weekly_out = weekly[[
        "Open", "High", "Low", "Close", "Volume", "MA200W", "rising", "from_above",
        "evt_5pct", "evt_4pct", "evt_3pct", "evt_2pct", "evt_1pct", "evt_touch", "setup_start"
    ]].copy()
    weekly_out.to_csv(events_path, index=True)

    inst_path = os.path.join(out_dir, f"instances_{symbol}.csv")
    instances.to_csv(inst_path, index=False)

    plot_path = maybe_plot(symbol, weekly, instances, out_dir, ma_type=ma_type) if plot else None

    # Console summary
    print(f"Symbol: {symbol}")
    print(f"Total trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']}")
    print(f"Stops: {summary['stops']}")
    print(f"Open: {summary['open']}")
    print(f"Win rate: {summary['win_rate']:.2%}")
    print(f"Avg return: {summary['avg_return']:.2%}")
    print(f"Events CSV: {events_path}")
    print(f"Instances CSV: {inst_path}")
    if plot_path:
        print(f"Plot: {plot_path}")


def live_check(symbol: str, start: str, week_ending: str = "W-FRI", ma_type: Literal["sma", "wma"] = "sma") -> None:
    """Print a quick proximity status vs 200W MA using latest price.

    - Computes weekly MA from history (daily -> weekly aggregation)
    - Fetches latest price (best-effort via yfinance)
    - Reports distance and which thresholds (5/4/3/2/1% or touch) are currently crossed
    - Applies rising/from_above conditions based on last completed week
    """
    daily = load_daily(symbol, start=start, csv_path=None)
    weekly = to_weekly(daily, week_ending=week_ending)
    weekly = add_ma200w(weekly, ma_type=ma_type)
    weekly = weekly.dropna(subset=["MA200W", "MA200W_prev"]).copy()
    if len(weekly) < 2:
        raise RuntimeError("Not enough weekly data for live check.")

    ma = float(weekly["MA200W"].iloc[-1])
    prev_ma = float(weekly["MA200W_prev"].iloc[-1])
    rising = ma > prev_ma
    # From_above uses last completed week
    last_close_prev = float(weekly["Close"].iloc[-2]) if len(weekly) >= 2 else np.nan
    prev_ma_prev = float(weekly["MA200W_prev"].iloc[-1]) if len(weekly) >= 1 else np.nan
    from_above = bool(last_close_prev > prev_ma_prev)

    last_price = _yf_last_price(symbol)
    dist_pct = (last_price - ma) / ma * 100.0

    levels = []
    for L in [5, 4, 3, 2, 1]:
        thr = ma * (1.0 + L / 100.0)
        if last_price <= thr:
            levels.append(f"{L}%")
    touch = last_price <= ma

    print(f"Symbol: {symbol}")
    print(f"Latest price: {last_price:.2f}")
    print(f"MA200W ({ma_type}): {ma:.2f}")
    print(f"Rising MA: {rising}")
    print(f"From above (last week): {from_above}")
    print(f"Distance to MA: {dist_pct:.2f}%")
    if levels:
        print(f"Within levels: {', '.join(levels)}")
    if touch:
        print("Touch: price <= MA200W")


def main():
    parser = argparse.ArgumentParser(description="200W MA approach/backtest demo")
    parser.add_argument("--symbol", default="MSFT", help="Ticker symbol (e.g., MSFT)")
    parser.add_argument("--start", default="1990-01-01", help="Start date for history")
    parser.add_argument("--csv", default=None, help="Optional path to CSV with OHLCV")
    parser.add_argument("--out", default="outputs", help="Output directory for CSV/plots")
    parser.add_argument("--week-ending", default="W-FRI", help="Week ending day (e.g. W-FRI, W-SUN)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot output")
    parser.add_argument("--ma-type", choices=["sma", "wma"], default="sma", help="Type of 200W MA to use")
    parser.add_argument("--exit-mode", choices=["none", "profit", "trail"], default="none",
                        help="Exit rule: none, profit target, or 20W trailing")
    parser.add_argument("--profit-pct", type=float, default=0.15, help="Profit target (e.g., 0.15 = 15%)")
    parser.add_argument("--live-check", action="store_true", help="Print current proximity vs 200W MA for the symbol")
    args = parser.parse_args()

    if args.live_check:
        live_check(
            symbol=args.symbol,
            start=args.start,
            week_ending=args.week_ending,
            ma_type=args.ma_type,
        )
    else:
        run(
            symbol=args.symbol,
            start=args.start,
            csv=args.csv,
            out_dir=args.out,
            week_ending=args.week_ending,
            plot=not args.no_plot,
            ma_type=args.ma_type,
            exit_mode=args.exit_mode,
            profit_target_pct=args.profit_pct,
        )


if __name__ == "__main__":
    main()
