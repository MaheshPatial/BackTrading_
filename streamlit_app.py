import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:  # Fallback if Plotly isn't available
    go = None

# Try to import the backtesting runner for on-demand generation
try:
    from backtesting import run_in_memory  # type: ignore
except Exception:
    run_in_memory = None


def load_events_from_files(outputs_dir: str) -> pd.DataFrame:
    """Load events from CSV files (fallback for pre-generated data)."""
    files = sorted(glob.glob(os.path.join(outputs_dir, "events_*.csv")))
    frames: List[pd.DataFrame] = []
    for fp in files:
        sym = Path(fp).stem.replace("events_", "")
        df = pd.read_csv(fp)
        # Normalize date column
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Date"})
        elif "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Date"]).copy()
    return out


def load_instances_from_files(outputs_dir: str) -> pd.DataFrame:
    """Load instances from CSV files (fallback for pre-generated data)."""
    files = sorted(glob.glob(os.path.join(outputs_dir, "instances_*.csv")))
    frames: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=["start", "end"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["start"] = pd.to_datetime(out["start"]).dt.tz_localize(None)
    if "end" in out.columns:
        out["end"] = pd.to_datetime(out["end"]).dt.tz_localize(None)
    return out


def symbol_list(events: pd.DataFrame, instances: pd.DataFrame) -> List[str]:
    syms = set()
    if not events.empty:
        syms.update(events["symbol"].unique().tolist())
    if not instances.empty and "symbol" in instances.columns:
        syms.update(instances["symbol"].unique().tolist())
    return sorted(syms)


def filter_date_range(df: pd.DataFrame, start, end, date_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    return df[mask]


def make_chart(evs: pd.DataFrame, inst: pd.DataFrame, sym: str):
    if go is None:
        st.line_chart(evs.set_index("Date")["Close"])  # minimal fallback
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evs["Date"], y=evs["Close"], name="Close", mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=evs["Date"], y=evs["MA200W"], name="MA200W", mode="lines", line=dict(color="purple")))

    # Entry/Exit markers
    if not inst.empty:
        evs_idx = evs.set_index("Date")["Close"]
        starts = inst["start"].dropna().unique().tolist()
        ends = inst["end"].dropna().unique().tolist()
        if starts:
            y = [evs_idx.get(t, np.nan) for t in starts]
            fig.add_trace(go.Scatter(x=starts, y=y, mode="markers", name="Entry start",
                                     marker=dict(symbol="triangle-up", color="green", size=10)))
        if ends:
            y = [evs_idx.get(t, np.nan) for t in ends]
            fig.add_trace(go.Scatter(x=ends, y=y, mode="markers", name="Exit",
                                     marker=dict(symbol="triangle-down", color="red", size=10)))

    fig.update_layout(title=f"{sym} — Weekly Close & MA200W", legend=dict(orientation="h"))
    fig.update_yaxes(tickformat=",.2f")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="200W MA Backtest Dashboard", layout="wide")
    st.title("200W MA Backtest Dashboard")

    # Initialize session state for generated data
    if "generated_events" not in st.session_state:
        st.session_state.generated_events = pd.DataFrame()
    if "generated_instances" not in st.session_state:
        st.session_state.generated_instances = pd.DataFrame()
    if "generation_status" not in st.session_state:
        st.session_state.generation_status = ""

    # Sidebar controls
    st.sidebar.markdown("## ⚡ Generate Data")
    st.sidebar.caption("Generate backtest data on-demand. Data is stored in your browser session.")
    
    DEFAULT_TICKERS = ["AAPL", "AMZN", "NVDA", "AMD", "MSFT", "GOOGL", "META"]
    quick_syms = st.sidebar.multiselect("Select tickers", DEFAULT_TICKERS, default=["NVDA"])
    custom_str = st.sidebar.text_input("Custom tickers (comma-separated)")
    custom_syms = [s.strip().upper() for s in custom_str.split(",") if s.strip()] if custom_str else []
    gen_syms = sorted(set(quick_syms + custom_syms))

    col_a, col_b = st.sidebar.columns(2)
    week_ending = col_a.selectbox("Week ending", ["W-FRI", "W-SUN"], index=0)
    ma_type = col_b.selectbox("MA type", ["sma", "wma"], index=0)
    exit_mode = col_a.selectbox("Exit mode", ["none", "profit", "trail"], index=0)
    profit_pct = float(col_b.number_input("Profit target", min_value=0.0, max_value=5.0, value=0.15, step=0.05))
    start_date = st.sidebar.text_input("History start date", value="1990-01-01")

    # Generate button
    if st.sidebar.button("🚀 Generate Data", type="primary", use_container_width=True):
        if run_in_memory is None:
            st.sidebar.error("❌ Cannot import backtesting module. Ensure backtesting.py is present.")
        elif not gen_syms:
            st.sidebar.warning("⚠️ Select at least one ticker.")
        else:
            all_events = []
            all_instances = []
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            for i, sym in enumerate(gen_syms):
                status_text.text(f"Processing {sym}...")
                try:
                    events_df, instances_df = run_in_memory(
                        symbol=sym,
                        start=start_date,
                        csv=None,
                        week_ending=week_ending,
                        ma_type=ma_type,
                        exit_mode=exit_mode,
                        profit_target_pct=profit_pct,
                    )
                    all_events.append(events_df)
                    all_instances.append(instances_df)
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing {sym}: {e}")
                progress_bar.progress((i + 1) / len(gen_syms))
            
            # Combine all results
            if all_events:
                st.session_state.generated_events = pd.concat(all_events, ignore_index=True)
            if all_instances:
                st.session_state.generated_instances = pd.concat(all_instances, ignore_index=True)
            
            status_text.text("✅ Generation complete!")
            st.session_state.generation_status = f"Generated data for: {', '.join(gen_syms)}"
            st.rerun()

    # Show generation status
    if st.session_state.generation_status:
        st.sidebar.success(st.session_state.generation_status)

    st.sidebar.markdown("---")

    # Use generated data (in-memory) as primary source
    events = st.session_state.generated_events.copy()
    instances = st.session_state.generated_instances.copy()

    # If no generated data, show instructions
    if events.empty or instances.empty:
        st.info("👆 **Get started**: Select tickers in the sidebar and click **Generate Data** to see your backtest results.")
        st.markdown("""
        ### How it works:
        1. **Select tickers** from the dropdown or enter custom ones
        2. **Configure parameters** (MA type, exit mode, etc.)
        3. Click **🚀 Generate Data**
        4. View charts, trades, and statistics below
        
        > **Note**: Data is generated in real-time and stored in your browser session. 
        > If you refresh the page, you'll need to regenerate.
        """)
        return

    # Normalize types
    events["Date"] = pd.to_datetime(events["Date"]).dt.tz_localize(None)
    instances["start"] = pd.to_datetime(instances["start"]).dt.tz_localize(None)
    if "end" in instances.columns:
        instances["end"] = pd.to_datetime(instances["end"]).dt.tz_localize(None)

    # Symbol filter
    symbols = symbol_list(events, instances)
    default_pick = symbols[:1] if symbols else []
    sel_symbols = st.sidebar.multiselect("Filter symbols", options=symbols, default=default_pick)

    # Date range based on events
    min_d = events["Date"].min().date()
    max_d = events["Date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    outcome_opts = sorted(instances["outcome"].dropna().unique().tolist()) if "outcome" in instances.columns else []
    sel_outcomes = st.sidebar.multiselect("Outcomes", options=outcome_opts, default=outcome_opts)

    # Overview KPIs for selection
    inst_sel = instances[instances["symbol"].isin(sel_symbols)] if "symbol" in instances.columns and sel_symbols else instances.copy()
    if sel_outcomes:
        inst_sel = inst_sel[inst_sel["outcome"].isin(sel_outcomes)]
    inst_sel = inst_sel.copy()
    total = int(len(inst_sel))
    wins = int(inst_sel[inst_sel["outcome"].isin(["profit", "trail"])].shape[0]) if total else 0
    stops = int(inst_sel[inst_sel["outcome"] == "stop"].shape[0]) if total else 0
    opens = int(inst_sel[inst_sel["outcome"] == "open"].shape[0]) if total else 0
    avg_ret = float(inst_sel["return_pct"].dropna().mean()) if total else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", f"{total}")
    c2.metric("Win rate", f"{(wins/total):.2%}" if total else "—")
    c3.metric("Stops", f"{stops}")
    c4.metric("Avg return", f"{avg_ret:.2%}" if total else "—")

    st.subheader("Trades Table")
    st.dataframe(inst_sel.sort_values("start"), use_container_width=True, height=280)

    # Per-symbol chart and events
    display_symbols = sel_symbols if sel_symbols else symbols
    for sym in display_symbols:
        st.markdown(f"### {sym}")
        evs_sym = events[events["symbol"] == sym].copy()
        evs_sym = filter_date_range(evs_sym, start_d, end_d, "Date")
        inst_sym = instances[instances["symbol"] == sym].copy()
        if sel_outcomes:
            inst_sym = inst_sym[inst_sym["outcome"].isin(sel_outcomes)]
        inst_sym = inst_sym[(inst_sym["start"] >= pd.to_datetime(start_d)) & (inst_sym["start"] <= pd.to_datetime(end_d))]

        make_chart(evs_sym, inst_sym, sym)

        with st.expander("Show weekly events data"):
            show_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "MA200W", "rising", "from_above",
                                      "evt_5pct", "evt_4pct", "evt_3pct", "evt_2pct", "evt_1pct", "evt_touch", "setup_start", "symbol"] if c in evs_sym.columns]
            st.dataframe(evs_sym[show_cols].sort_values("Date"), use_container_width=True, height=240)


# Sidebar help sections
with st.sidebar.expander("What are MA and WMA?"):
    st.markdown(
        "- **MA (moving average)**: the average of the last N prices. Here we use a 200-week simple moving average (SMA) by default.\n"
        "- **WMA (weighted moving average)**: same window but recent prices get higher weights, making it respond slightly faster.\n"
        "- In this project, '200W MA' refers to the moving average computed from weekly candles over 200 weeks."
    )

with st.sidebar.expander("What is scale-in (3/2/1% + touch)?"):
    st.markdown(
        "The position is built in four steps as price approaches the 200W MA from above on a rising trend:\n"
        "- At 3% above MA: buy 7.5% of the intended position.\n"
        "- At 2% above MA: buy 10%.\n"
        "- At 1% above MA: buy 12.5%.\n"
        "- On a touch (price ≤ MA): buy the remaining to reach 100%.\n"
        "This avoids missing bounces that occur before a full touch and concentrates size near the MA."
    )


if __name__ == "__main__":
    main()
