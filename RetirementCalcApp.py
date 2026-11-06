import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Early Retirement (FIRE) Calculator",
    page_icon="💰",
    layout="wide"
)

# --- HISTORICAL MARKET DATA (1900-2026 S&P 500 Total Returns Dataset) ---
HISTORICAL_RETURNS = [
    {"year": 1900, "return": 0.12}, {"year": 1901, "return": 0.15}, {"year": 1902, "return": -0.01}, {"year": 1903, "return": -0.15}, {"year": 1904, "return": 0.28},
    {"year": 1905, "return": 0.18}, {"year": 1906, "return": -0.06}, {"year": 1907, "return": -0.28}, {"year": 1908, "return": 0.38}, {"year": 1909, "return": 0.15},
    {"year": 1910, "return": -0.05}, {"year": 1911, "return": 0.03}, {"year": 1912, "return": 0.06}, {"year": 1913, "return": -0.11}, {"year": 1914, "return": -0.08},
    {"year": 1915, "return": 0.34}, {"year": 1916, "return": 0.08}, {"year": 1917, "return": -0.22}, {"year": 1918, "return": 0.13}, {"year": 1919, "return": 0.16},
    {"year": 1920, "return": -0.15}, {"year": 1921, "return": 0.13}, {"year": 1922, "return": 0.28}, {"year": 1923, "return": 0.03}, {"year": 1924, "return": 0.27},
    {"year": 1925, "return": 0.22}, {"year": 1926, "return": 0.12}, {"year": 1927, "return": 0.37}, {"year": 1928, "return": 0.44}, {"year": 1929, "return": -0.08},
    {"year": 1930, "return": -0.25}, {"year": 1931, "return": -0.43}, {"year": 1932, "return": -0.08}, {"year": 1933, "return": 0.54}, {"year": 1934, "return": -0.01},
    {"year": 1935, "return": 0.47}, {"year": 1936, "return": 0.34}, {"year": 1937, "return": -0.35}, {"year": 1938, "return": 0.31}, {"year": 1939, "return": -0.01},
    {"year": 1940, "return": -0.10}, {"year": 1941, "return": -0.12}, {"year": 1942, "return": 0.20}, {"year": 1943, "return": 0.26}, {"year": 1944, "return": 0.20},
    {"year": 1945, "return": 0.36}, {"year": 1946, "return": -0.08}, {"year": 1947, "return": 0.06}, {"year": 1948, "return": 0.05}, {"year": 1949, "return": 0.18},
    {"year": 1950, "return": 0.31}, {"year": 1951, "return": 0.24}, {"year": 1952, "return": 0.18}, {"year": 1953, "return": -0.01}, {"year": 1954, "return": 0.53},
    {"year": 1955, "return": 0.32}, {"year": 1956, "return": 0.07}, {"year": 1957, "return": -0.11}, {"year": 1958, "return": 0.43}, {"year": 1959, "return": 0.12},
    {"year": 1960, "return": 0.00}, {"year": 1961, "return": 0.27}, {"year": 1962, "return": -0.09}, {"year": 1963, "return": 0.23}, {"year": 1964, "return": 0.16},
    {"year": 1965, "return": 0.12}, {"year": 1966, "return": -0.10}, {"year": 1967, "return": 0.24}, {"year": 1968, "return": 0.11}, {"year": 1969, "return": -0.08},
    {"year": 1970, "return": 0.04}, {"year": 1971, "return": 0.14}, {"year": 1972, "return": 0.19}, {"year": 1973, "return": -0.15}, {"year": 1974, "return": -0.26},
    {"year": 1975, "return": 0.37}, {"year": 1976, "return": 0.24}, {"year": 1977, "return": -0.07}, {"year": 1978, "return": 0.07}, {"year": 1979, "return": 0.19},
    {"year": 1980, "return": 0.32}, {"year": 1981, "return": -0.05}, {"year": 1982, "return": 0.21}, {"year": 1983, "return": 0.22}, {"year": 1984, "return": 0.06},
    {"year": 1985, "return": 0.31}, {"year": 1986, "return": 0.18}, {"year": 1987, "return": 0.05}, {"year": 1988, "return": 0.16}, {"year": 1989, "return": 0.31},
    {"year": 1990, "return": -0.03}, {"year": 1991, "return": 0.30}, {"year": 1992, "return": 0.07}, {"year": 1993, "return": 0.10}, {"year": 1994, "return": 0.01},
    {"year": 1995, "return": 0.37}, {"year": 1996, "return": 0.23}, {"year": 1997, "return": 0.33}, {"year": 1998, "return": 0.28}, {"year": 1999, "return": 0.21},
    {"year": 2000, "return": -0.09}, {"year": 2001, "return": -0.12}, {"year": 2002, "return": -0.22}, {"year": 2003, "return": 0.28}, {"year": 2004, "return": 0.11},
    {"year": 2005, "return": 0.05}, {"year": 2006, "return": 0.15}, {"year": 2007, "return": 0.05}, {"year": 2008, "return": -0.37}, {"year": 2009, "return": 0.26},
    {"year": 2010, "return": 0.15}, {"year": 2011, "return": 0.02}, {"year": 2012, "return": 0.16}, {"year": 2013, "return": 0.32}, {"year": 2014, "return": 0.13},
    {"year": 2015, "return": 0.01}, {"year": 2016, "return": 0.12}, {"year": 2017, "return": 0.21}, {"year": 2018, "return": -0.04}, {"year": 2019, "return": 0.31},
    {"year": 2020, "return": 0.18}, {"year": 2021, "return": 0.28}, {"year": 2022, "return": -0.18}, {"year": 2023, "return": 0.26}, {"year": 2024, "return": 0.24},
    {"year": 2025, "return": 0.10}, {"year": 2026, "return": 0.08}
]

# Helper function to fetch historical return
def get_historical_return(start_year, offset):
    hist_years = [r["year"] for r in HISTORICAL_RETURNS]
    returns = [r["return"] for r in HISTORICAL_RETURNS]
    
    if start_year in hist_years:
        idx = hist_years.index(start_year)
        target_idx = (idx + offset) % len(returns)
        return returns[target_idx]
    return 0.07

# Simulation Engine Logic
def simulate_year(state, year_idx, params, mkt_return):
    p1_age = params["p1_current_age"] + year_idx
    p2_age = params["p2_current_age"] + year_idx

    p1_retired = p1_age >= params["p1_retire_age"]
    p2_retired = p2_age >= params["p2_retire_age"]

    # Inflation factor
    inflation_factor = (1 + params["inflation_rate"]) ** year_idx

    # Lifecycle Phase
    if p1_retired and p2_retired:
        phase = 3 # Full Retirement
    elif p1_retired or p2_retired:
        phase = 2 # Single-Earner Gap
    else:
        phase = 1 # Pre-retirement

    # Social Security / Pension Income
    ss_income = 0.0
    if p1_age >= params["p1_ss_age"]:
        ss_income += params["p1_ss_amt"] * inflation_factor
    if p2_age >= params["p2_ss_age"]:
        ss_income += params["p2_ss_amt"] * inflation_factor

    # Spending Target
    if phase == 1:
        base_spend = 0.0
    elif phase == 2:
        base_spend = params["phase2_spend"]
    else:
        if params["strategy"] == "tiered":
            if mkt_return >= 0.20:
                base_spend = params["tier_high"]
            elif mkt_return >= 0.10:
                base_spend = params["tier_mid_high"]
            elif mkt_return <= -0.10:
                base_spend = params["tier_low"]
            else:
                base_spend = params["tier_base"]
        elif params["strategy"] == "fixed":
            base_spend = params["tier_base"]
        elif params["strategy"] == "gk":
            base_spend = params["tier_base"]
            if mkt_return < -0.10:
                base_spend *= 0.90
            elif mkt_return > 0.15:
                base_spend *= 1.10
        elif params["strategy"] == "vpw":
            cur_tot = state["cash"] + state["brokerage"] + state["tax_adv"]
            rem_yrs = max(1, params["model_years"] - year_idx)
            base_spend = cur_tot * (1 / rem_yrs) / inflation_factor

    target_spend = base_spend * inflation_factor

    # Pre-Medicare Healthcare Costs
    healthcare_cost = 0.0
    if phase > 1 and (p1_age < 65 or p2_age < 65):
        healthcare_cost = params["health_cost"] * inflation_factor

    net_need = max(0.0, (target_spend + healthcare_cost) - ss_income)

    # Tax Gross-Up (Phase 3 Only)
    gross_draw_needed = net_need
    tax_gross_up = 0.0
    if phase == 3 and net_need > 0:
        tax_rate = min(0.40, max(0.0, params["tax_rate"]))
        gross_draw_needed = net_need / (1.0 - tax_rate)
        tax_gross_up = gross_draw_needed - net_need

    start_total = state["cash"] + state["brokerage"] + state["tax_adv"]

    # Account Liquidation Order
    remaining_draw = gross_draw_needed
    drawn_cash = 0.0
    drawn_brokerage = 0.0
    drawn_tax_adv = 0.0

    if params["withdrawal_seq"] == "prorata":
        if start_total > 0 and remaining_draw > 0:
            ratio = min(1.0, remaining_draw / start_total)
            drawn_cash = state["cash"] * ratio
            drawn_brokerage = state["brokerage"] * ratio
            drawn_tax_adv = state["tax_adv"] * ratio
    else:
        sequence = params["withdrawal_seq"].split("-")
        for key in sequence:
            if remaining_draw <= 0:
                break
            if key == "cash":
                drawn_cash = min(state["cash"], remaining_draw)
                remaining_draw -= drawn_cash
            elif key == "brok":
                drawn_brokerage = min(state["brokerage"], remaining_draw)
                remaining_draw -= drawn_brokerage
            elif key == "taxadv":
                drawn_tax_adv = min(state["tax_adv"], remaining_draw)
                remaining_draw -= drawn_tax_adv

    total_drawn = drawn_cash + drawn_brokerage + drawn_tax_adv

    # Post-drawdown market growth
    rem_cash = (state["cash"] - drawn_cash) * 1.02 # Fixed 2% cash return
    rem_brokerage = (state["brokerage"] - drawn_brokerage) * (1 + mkt_return)
    rem_tax_adv = (state["tax_adv"] - drawn_tax_adv) * (1 + mkt_return)

    end_total = rem_cash + rem_brokerage + rem_tax_adv

    # Mutate state
    state["cash"] = rem_cash
    state["brokerage"] = rem_brokerage
    state["tax_adv"] = rem_tax_adv

    return {
        "Year": year_idx + 1,
        "P1 Age": p1_age,
        "P2 Age": p2_age,
        "Phase": f"Phase {phase}",
        "Start Total": start_total,
        "Market Return": mkt_return,
        "SS/Pension": ss_income,
        "Target Spend": target_spend,
        "Healthcare": healthcare_cost,
        "Tax Gross-Up": tax_gross_up,
        "Net Drawn": total_drawn,
        "Cash Buffer": rem_cash,
        "Brokerage": rem_brokerage,
        "Tax-Adv.": rem_tax_adv,
        "End Total": end_total
    }

# --- STREAMLIT UI LAYOUT ---
st.title("🧮 Early Retirement (FIRE) Calculator & Backtester")
st.caption("Dual-Spouse Staggered Timeline, Tax-Smart Account Sequencing & 1900–2026 Historical Engine")

# Sidebar Configuration
st.sidebar.header("🕹️ Simulation Engine")
engine_mode = st.sidebar.radio("Engine Mode", ["Historical Backtest", "Monte Carlo (500x)"])

if engine_mode == "Historical Backtest":
    preset = st.sidebar.selectbox(
        "Crisis Presets", 
        ["Custom Year Selection", "1929 (Great Depression)", "1966 (Stagflation Era)", "1973 (Oil Shock)", "2000 (Dot-Com Burst)", "2008 (Global Financial Crisis)", "2020 (Pandemic Era)"]
    )
    preset_years = {
        "1929 (Great Depression)": 1929,
        "1966 (Stagflation Era)": 1966,
        "1973 (Oil Shock)": 1973,
        "2000 (Dot-Com Burst)": 2000,
        "2008 (Global Financial Crisis)": 2008,
        "2020 (Pandemic Era)": 2020
    }
    default_start = preset_years.get(preset, 2000)
    hist_start_year = st.sidebar.number_input("Backtest Start Year", min_value=1900, max_value=2020, value=default_start)
else:
    volatility = st.sidebar.slider("Stock Volatility (Std Dev %)", 5.0, 30.0, 16.0) / 100.0
    avg_stock_ret = st.sidebar.number_input("Avg Stock Return %", value=9.0) / 100.0
    avg_bond_ret = st.sidebar.number_input("Avg Bond Return %", value=4.0) / 100.0

st.sidebar.header("💼 Portfolio Balances")
cash_bal = st.sidebar.number_input("Cash / Money Market ($)", value=100000, step=10000)
brokerage_bal = st.sidebar.number_input("Taxable Brokerage ($)", value=800000, step=25000)
tax_adv_bal = st.sidebar.number_input("Tax-Advantaged ($)", value=1100000, step=25000)

st.sidebar.header("👥 Household Timeline")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("**Primary Spouse**")
    p1_current_age = st.number_input("P1 Age", value=53)
    p1_retire_age = st.number_input("P1 Retire Age", value=55)
    p1_ss_amt = st.number_input("P1 SS ($/yr)", value=28000)
    p1_ss_age = st.number_input("P1 SS Age", value=62)

with col2:
    st.markdown("**Partner / Spouse**")
    p2_current_age = st.number_input("P2 Age", value=52)
    p2_retire_age = st.number_input("P2 Retire Age", value=58)
    p2_ss_amt = st.number_input("P2 SS ($/yr)", value=22000)
    p2_ss_age = st.number_input("P2 SS Age", value=62)

model_years = st.sidebar.number_input("Years to Model", value=35, min_value=5, max_value=60)

st.sidebar.header("⚙️ Spending & Strategy Rules")
strategy = st.sidebar.selectbox("Withdrawal Strategy", ["tiered", "fixed", "gk", "vpw"], format_func=lambda x: {
    "tiered": "Tiered Market-Return Rules (Custom)",
    "fixed": "Fixed Dollar (Inflation-Adjusted)",
    "gk": "Guyton-Klinger Guardrails",
    "vpw": "Variable Percentage Withdrawal (VPW)"
}[x])

withdrawal_seq = st.sidebar.selectbox("Withdrawal Sequence", [
    "cash-brok-taxadv", "brok-cash-taxadv", "taxadv-brok-cash", "prorata"
], format_func=lambda x: {
    "cash-brok-taxadv": "Cash → Brokerage → Tax-Advantaged",
    "brok-cash-taxadv": "Brokerage → Cash → Tax-Advantaged",
    "taxadv-brok-cash": "Tax-Advantaged → Brokerage → Cash",
    "prorata": "Pro-Rata (Proportional)"
}[x])

if strategy == "tiered":
    tier_high = st.sidebar.number_input("Market Return ≥ +20% ($/yr)", value=200000)
    tier_mid_high = st.sidebar.number_input("Market Return +10% to +20% ($/yr)", value=150000)
    tier_base = st.sidebar.number_input("Base Scenarios (-10% to +10%) ($/yr)", value=125000)
    tier_low = st.sidebar.number_input("Market Return ≤ -10% ($/yr)", value=80000)
else:
    tier_high = tier_mid_high = tier_low = 0
    tier_base = st.sidebar.number_input("Base Withdrawal ($/yr)", value=125000)

phase2_spend = st.sidebar.number_input("Phase 2 Single-Earner Spend ($/yr)", value=100000)

st.sidebar.header("🏥 Health, Taxes & Asset Allocation")
tax_rate = st.sidebar.number_input("Effective Tax Rate %", value=15.0) / 100.0
health_cost = st.sidebar.number_input("Pre-65 Health Cost ($/yr)", value=12000)
stock_pct = st.sidebar.slider("Stock Allocation %", 0, 100, 75) / 100.0
bond_pct = 1.0 - stock_pct
inflation_rate = st.sidebar.number_input("General Inflation Rate %", value=2.5) / 100.0

# Consolidate parameters
params = {
    "cash_bal": cash_bal,
    "brokerage_bal": brokerage_bal,
    "tax_adv_bal": tax_adv_bal,
    "p1_current_age": p1_current_age,
    "p1_retire_age": p1_retire_age,
    "p1_ss_amt": p1_ss_amt,
    "p1_ss_age": p1_ss_age,
    "p2_current_age": p2_current_age,
    "p2_retire_age": p2_retire_age,
    "p2_ss_amt": p2_ss_amt,
    "p2_ss_age": p2_ss_age,
    "model_years": model_years,
    "strategy": strategy,
    "withdrawal_seq": withdrawal_seq,
    "tier_high": tier_high,
    "tier_mid_high": tier_mid_high,
    "tier_base": tier_base,
    "tier_low": tier_low,
    "phase2_spend": phase2_spend,
    "tax_rate": tax_rate,
    "health_cost": health_cost,
    "stock_pct": stock_pct,
    "bond_pct": bond_pct,
    "inflation_rate": inflation_rate,
}

# --- EXECUTE SIMULATION ---
if engine_mode == "Historical Backtest":
    params["hist_start_year"] = hist_start_year
    state = {"cash": cash_bal, "brokerage": brokerage_bal, "tax_adv": tax_adv_bal}
    logs = []
    
    for i in range(model_years):
        stock_ret = get_historical_return(hist_start_year, i)
        blended_ret = (stock_ret * stock_pct) + (0.03 * bond_pct)
        log = simulate_year(state, i, params, blended_ret)
        logs.append(log)

    df_ledger = pd.DataFrame(logs)
    
    # Calculate Summary Metrics
    success = df_ledger["End Total"].min() > 0
    final_bal = df_ledger["End Total"].iloc[-1]
    lowest_bal = df_ledger["End Total"].min()
    avg_withdrawal = df_ledger["Net Drawn"].mean()

    # Metrics Display
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plan Success Rate", "100%" if success else "0%", delta="Success" if success else "Failed", delta_color="normal" if success else "inverse")
    m2.metric("Final Balance", f"${final_bal:,.0f}")
    m3.metric("Lowest Balance", f"${lowest_bal:,.0f}")
    m4.metric("Avg Annual Withdrawal", f"${avg_withdrawal:,.0f}")

    # Plot Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_ledger["Year"], 
        y=df_ledger["End Total"], 
        mode='lines+markers', 
        name=f"Historical Path ({hist_start_year})",
        line=dict(color='#10b981', width=3)
    ))
    fig.update_layout(
        title="Portfolio Balance Trajectory Over Time",
        xaxis_title="Simulation Year",
        yaxis_title="Portfolio Balance ($)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Monte Carlo Engine
    NUM_SIMS = 500
    blended_mean = (avg_stock_ret * stock_pct) + (avg_bond_ret * bond_pct)
    sim_trajectories = []
    rep_logs = []
    success_count = 0

    for sim in range(NUM_SIMS):
        state = {"cash": cash_bal, "brokerage": brokerage_bal, "tax_adv": tax_adv_bal}
        traj = []
        sim_logs = []
        sim_success = True

        for i in range(model_years):
            mkt_ret = np.random.normal(blended_mean, volatility)
            log = simulate_year(state, i, params, mkt_ret)
            traj.append(log["End Total"])
            sim_logs.append(log)
            if log["End Total"] <= 0:
                sim_success = False

        if sim_success:
            success_count += 1
        sim_trajectories.append(traj)
        if sim == 0:
            rep_logs = sim_logs

    df_ledger = pd.DataFrame(rep_logs)
    success_rate = (success_count / NUM_SIMS) * 100
    final_bals = [t[-1] for t in sim_trajectories]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monte Carlo Success", f"{success_rate:.1f}%")
    m2.metric("Median Final Balance", f"${np.median(final_bals):,.0f}")
    m3.metric("10th Percentile Balance", f"${np.percentile(final_bals, 10):,.0f}")
    m4.metric("Avg Sample Withdrawal", f"${df_ledger['Net Drawn'].mean():,.0f}")

    # Plot Sample Trajectories
    fig = go.Figure()
    for s in range(min(20, NUM_SIMS)):
        fig.add_trace(go.Scatter(
            y=sim_trajectories[s], 
            mode='lines', 
            opacity=0.3,
            showlegend=False
        ))
    fig.update_layout(
        title="Monte Carlo Sample Simulations (20 Trajectories)",
        xaxis_title="Simulation Year",
        yaxis_title="Portfolio Balance ($)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# --- DETAILED LEDGER TABLE ---
st.subheader("📋 Year-by-Year Multi-Account Withdrawal Ledger")

# Formatting Pandas DataFrame for UI presentation
formatted_df = df_ledger.copy()
currency_cols = ["Start Total", "SS/Pension", "Target Spend", "Healthcare", "Tax Gross-Up", "Net Drawn", "Cash Buffer", "Brokerage", "Tax-Adv.", "End Total"]
for col in currency_cols:
    formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")

formatted_df["Market Return"] = formatted_df["Market Return"].apply(lambda x: f"{x * 100:.1f}%")

st.dataframe(formatted_df, use_container_width=True, height=400)