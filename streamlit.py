import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as C
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis, norm
import streamlit.components.v1 as components

plt.style.use("ggplot")
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("BTC Dashboard")

DAYS_TO_FETCH = 365

def momentum_bar(value, title, min_val=-1, max_val=1, width="100%"):
    pct = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.0
    pct = max(0.0, min(1.0, pct))
    pct_pct = pct * 100.0
    if value > 0.1:
        color = "#2ecc71"
        label = "Positive"
    elif value < -0.1:
        color = "#e74c3c"
        label = "Negative"
    else:
        color = "#f39c12"
        label = "Neutral"
    html = f"""
    <div style="font-family: Inter, Arial, sans-serif; width:{width}; padding:6px 4px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-weight:600; font-size:15px; color:#fff; margin-bottom:6px;">{title}</div>
        <div style="font-size:13px; color:#d1d5db; margin-left:8px;">{label}</div>
      </div>
      <div style="
          width:100%;
          height:12px;
          background:#111827;
          border-radius:8px;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
          overflow:hidden;
          margin-top:6px;">
        <div style="
            width:{pct_pct:.2f}%;
            height:100%;
            background: linear-gradient(90deg, {color}, {color});
            border-radius:8px;
            transition: width 600ms ease;">
        </div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:6px; color:#9CA3AF; font-size:12px;">
        <div>Value: {value:.4f}</div>
        <div>{pct_pct:.1f}%</div>
      </div>
    </div>
    """
    components.html(html, height=95, scrolling=False)

@st.cache_data(ttl=300)
def fetch_crypto_data(symbol, days=DAYS_TO_FETCH):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception as e:
        st.error(f"Failed to fetch {symbol} data: {e}")
        return pd.DataFrame()
    if 'prices' not in data:
        st.warning(f"No price data returned for {symbol}.")
        return pd.DataFrame()
    prices = np.array(data['prices'])
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if st.button("Refresh BTC Price Data"):
    st.cache_data.clear()

df = fetch_crypto_data("bitcoin")
if df.empty:
    st.stop()

    df = fetch_crypto_data("bitcoin")
if df.empty:
    st.stop()

st.sidebar.subheader("BTC Snapshot (Today)")

if not df.empty:
    latest_price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2] if len(df) > 1 else latest_price
    today_move = (latest_price - prev_close) / prev_close * 100

    st.sidebar.markdown(f"**Asset:** BTC")
    st.sidebar.markdown(f"**Current Price:** ${latest_price:,.2f}")
    st.sidebar.markdown(f"**Today's Move:** {today_move:+.2f}%")
else:
    st.sidebar.write("No BTC data available")

df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close']/df['close'].shift(1))
df['cum_return'] = (1 + df['return']).cumprod() - 1
df['drawdown'] = df['close']/df['close'].cummax() - 1
df['vol_30d'] = df['log_return'].rolling(30).std() * np.sqrt(365)
df['price_std'] = df['close'].rolling(30).std()
df['DMA365'] = df['close'].rolling(365, min_periods=1).mean()
df['Momentum_Deviation'] = df['close'] - df['DMA365']
df['Volatility_7'] = df['return'].rolling(7).std()
df['MA_20'] = df['close'].rolling(20).mean()
df['Proximity52W'] = (df['close'] - df['close'].rolling(365).min()) / (df['close'].rolling(365).max() - df['close'].rolling(365).min())
df['CrossMomentum'] = df['close']/df['MA_20'] - 1
df['RelMomentum'] = df['close'].pct_change(28)/df['close'].pct_change(84)

mean_vol = df['Volatility_7'].mean()
std_vol = df['Volatility_7'].std()
def get_regime(v):
    if v < mean_vol - 0.5*std_vol:
        return 'LOW'
    elif v > mean_vol + 0.5*std_vol:
        return 'HIGH'
    else:
        return 'MEDIUM'
df['Volatility_Regime'] = df['Volatility_7'].apply(get_regime)

X_price = np.arange(len(df)).reshape(-1,1)
y_price = df['close'].values.reshape(-1,1)
price_model = LinearRegression().fit(X_price, y_price)
price_trend = price_model.predict(X_price)

daily_returns = df['close'].pct_change().dropna().values
probs = np.ones_like(daily_returns) / len(daily_returns)

def expected_log_growth(f):
    if np.any(1 + f * daily_returns <= 0):
        return -np.inf
    return -np.sum(probs * np.log(1 + f * daily_returns))
result = minimize_scalar(expected_log_growth, bounds=(-2, 5), method='bounded')
f_star_monthly = result.x if result.success else np.nan

skewness = skew(df['return'].dropna()) if len(df['return'].dropna())>0 else np.nan
kurt_val = kurtosis(df['return'].dropna()) if len(df['return'].dropna())>0 else np.nan
max_dd = df['drawdown'].min()
annualized_vol = df['log_return'].std() * np.sqrt(365)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Stats & Momentum",
    "Weekly GPR & Position Designer",
    "Volatility Analytics",
    "Kelly Comparison",
    "Portfolio Optimization"
])

with tab1:
    st.subheader("Key BTC Metrics")
    cols = st.columns(4)
    metrics = {
        "Latest Price": df['close'].iloc[-1],
        "High (365d)": df['close'].max(),
        "Low (365d)": df['close'].min(),
        "DMA365": df['DMA365'].iloc[-1],
        "Price Std Dev (30d)": df['price_std'].iloc[-1],
        "Cumulative ROI": df['cum_return'].iloc[-1],
        "Max Drawdown": max_dd,
        "Skewness": skewness,
        "Kurtosis": kurt_val,
        "30d Realized Vol": df['vol_30d'].iloc[-1],
        "Kelly Fraction": f_star_monthly,
        "Proximity to 52W High": df['Proximity52W'].iloc[-1],
        "Cross-Sectional Momentum": df['CrossMomentum'].iloc[-1],
        "Relative Momentum": df['RelMomentum'].iloc[-1]
    }
    for i, (k,v) in enumerate(metrics.items()):
        try:
            cols[i%4].markdown(f"**{k}: {v:.4f}**")
        except:
            cols[i%4].markdown(f"**{k}: {v}**")

    st.subheader("Momentum Dashboard")
    colA, colB, colC = st.columns(3)
    momentum_bar(df['Proximity52W'].iloc[-1], "Proximity to 52W High", 0, 1)
    momentum_bar(df['CrossMomentum'].iloc[-1], "Cross-Sectional Momentum", -0.2, 0.2)
    momentum_bar(df['RelMomentum'].iloc[-1], "Relative Momentum", -2, 2)

    st.subheader("BTC Price & DMA365 Trend")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df.index, df['close'], label='Close Price', color='blue')
    ax.plot(df.index, df['DMA365'], label='DMA365', color='orange', linewidth=2)
    ax.plot(df.index, price_trend.ravel(), linestyle='--', label='Linear Trend', color='green')
    ax.set_ylabel("Price USD"); ax.set_xlabel("Date")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    df['dP_pct'] = df['close'].diff() / df['close'] * 100
    df['d2P_pct'] = df['dP_pct'].diff()

    st.subheader("BTC Price & Derivatives Comparison")
    fig_deriv, ax_deriv = plt.subplots(figsize=(10,3))
    ax_deriv.plot(df.index, df['close'], label='BTC Price', color='blue')
    ax_deriv.set_ylabel("Price USD", color='blue')
    ax_deriv.tick_params(axis='y', labelcolor='blue')

    ax2 = ax_deriv.twinx()
    ax2.plot(df.index, df['dP_pct'], label='Price Velocity (%/day)', color='purple')
    ax2.plot(df.index, df['d2P_pct'], label='Price Acceleration (%/day²)', color='red')
    ax2.set_ylabel("Derivative (%)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    lines1, labels1 = ax_deriv.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    ax_deriv.set_xlabel("Date")
    ax_deriv.grid(alpha=0.3)
    st.pyplot(fig_deriv)

    st.subheader("BTC Volatility (7-day rolling) with Regimes")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    colors = {'LOW':'blue','MEDIUM':'orange','HIGH':'red'}
    ax2.scatter(range(len(df)), df['Volatility_7'], c=df['Volatility_Regime'].map(colors), s=10)
    ax2.set_ylabel("Volatility"); ax2.set_xlabel("Date Index")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.subheader("Daily Log Returns Comparison: BTC vs ETH vs SOL")

    df_eth = fetch_crypto_data("ethereum")
    df_sol = fetch_crypto_data("solana")

    if not df_eth.empty and not df_sol.empty:
        df_eth['log_return'] = np.log(df_eth['close'] / df_eth['close'].shift(1))
        df_sol['log_return'] = np.log(df_sol['close'] / df_sol['close'].shift(1))
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        df_returns = pd.DataFrame({
            'BTC': df['log_return'],
            'ETH': df_eth['log_return'],
            'SOL': df_sol['log_return']
        }).dropna()

        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(df_returns.index, df_returns['BTC'], label='BTC', color='blue', alpha=0.8)
        ax.plot(df_returns.index, df_returns['ETH'], label='ETH', color='orange', alpha=0.8)
        ax.plot(df_returns.index, df_returns['SOL'], label='SOL', color='green', alpha=0.8)
        ax.set_title("Daily Log Returns Comparison")
        ax.set_ylabel("Log Return")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Not enough ETH or SOL data to plot returns comparison.")

with tab2:
    st.subheader("BTC Weekly GPR Prediction (Next 12 Weeks)")

    df_weekly = df['close'].resample('W').last().dropna()
    X_week = np.arange(len(df_weekly)).reshape(-1,1)
    y_week = df_weekly.values.ravel()
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_week)
    y_scaled = scaler_y.fit_transform(y_week.reshape(-1,1)).ravel()

    kernel = C(1.0)*RBF() + DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X_scaled, y_scaled)

    X_pred_numeric = np.linspace(0, len(df_weekly)+12, 300).reshape(-1,1)
    X_pred_scaled = scaler_X.transform(X_pred_numeric)
    y_pred_scaled, sigma_scaled = gpr.predict(X_pred_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    sigma = sigma_scaled * (scaler_y.data_max_ - scaler_y.data_min_)

    ci_level = st.slider("Select Confidence Interval (%)", min_value=80, max_value=99, value=95, step=1)
    z_dict = {80:1.282, 90:1.645, 95:1.96, 99:2.576}
    z = z_dict.get(ci_level, 1.96)
    
    lower_bound = y_pred[-1] - z*sigma[-1]
    upper_bound = y_pred[-1] + z*sigma[-1]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(X_pred_numeric.ravel(), y_pred, color='blue', linewidth=2, label='GPR Prediction')
    ax.fill_between(X_pred_numeric.ravel(),
                    y_pred - z*sigma,
                    y_pred + z*sigma,
                    color='blue', alpha=0.2, label=f'{ci_level}% CI')
    ax.scatter(np.arange(len(df_weekly)), y_week, color='black', s=20, label='Observed Prices')
    ax.set_title("BTC/USDT GPR Prediction")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    st.pyplot(fig)

    latest_price = y_week[-1]
    if latest_price < lower_bound:
        position = "Long (Buy Opportunity)"
        color = "green"
    elif latest_price > upper_bound:
        position = "Short (Potential Overbought)"
        color = "red"
    else:
        position = "Neutral (Hold)"
        color = "orange"

    st.markdown(f"**Latest Weekly Price:** {latest_price:.2f} USD")
    st.markdown(f"**GPR Predicted Range:** {lower_bound:.2f} - {upper_bound:.2f} USD")
    st.markdown(f"**Suggested Position:** <span style='color:{color}; font-weight:bold'>{position}</span>", unsafe_allow_html=True)

    expected_return = (y_pred[-1] - latest_price) / latest_price * 100
    st.markdown(f"**Expected Move:** {expected_return:.2f}%")

    future_slope = (y_pred[-1] - y_pred[-13]) / 12
    trend = "Uptrend " if future_slope > 0 else "Downtrend "
    st.markdown(f"**12-Week Trend:** {trend}")

    st.subheader("Recent Weekly Prices")
    st.table(df_weekly.tail(5).rename("Weekly Close Price"))

with tab3:
    st.subheader("Volatility Analytics (7-Day Rolling Volatility + 30-Day Realized Volatility)")

    vol_7d = df['Volatility_7'].dropna()

    df['log_return'] = np.log(df['close']/df['close'].shift(1))
    vol_30d = df['log_return'].rolling(30).std() * np.sqrt(365)
    latest_rv = vol_30d.iloc[-1]

    if vol_7d.empty or np.isnan(latest_rv):
        st.info("Not enough data to compute volatility metrics yet.")
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(f"**Mean 7-Day Volatility:** {vol_7d.mean():.6f}")
            st.markdown(f"**Std Dev 7-Day Volatility:** {vol_7d.std():.6f}")
            st.markdown(f"**Skewness 7-Day Volatility:** {skew(vol_7d):.6f}")
            st.markdown(f"**Kurtosis 7-Day Volatility:** {kurtosis(vol_7d):.6f}")
            st.markdown(f"**Latest 30-Day Realized Volatility:** {latest_rv:.6f} ({latest_rv*100:.2f}%)")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(vol_7d.index, vol_7d.values, label='7-Day Rolling Volatility', color='orange')
        ax.plot(vol_30d.index, vol_30d.values, label='30-Day Realized Volatility', color='blue', linestyle='--')
        ax.set_title("BTC Volatility Comparison")
        ax.set_ylabel("Volatility")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)
        
        p25 = vol_30d.quantile(0.25)
        p75 = vol_30d.quantile(0.75)

        ax.axhline(p25, color='green', linestyle='--', linewidth=1.5, label='25th Percentile (30d RV)')
        ax.axhline(p75, color='red', linestyle='--', linewidth=1.5, label='75th Percentile (30d RV)')

        ax.set_title("BTC Volatility Comparison")
        ax.set_ylabel("Volatility")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

        st.pyplot(fig)

        st.subheader("Normal Distribution of 30-Day Realized Volatility")
        mu = vol_30d.mean()
        sigma = vol_30d.std()
        x = np.linspace(vol_30d.min(), vol_30d.max(), 200)
        y = norm.pdf(x, mu, sigma)

        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.hist(vol_30d.dropna(), bins=30, density=True, alpha=0.6, color='skyblue', label='RV Histogram')
        ax2.plot(x, y, 'r--', linewidth=2, label=f'Normal Fit (μ={mu:.4f}, σ={sigma:.4f})')
        ax2.set_xlabel("30-Day Realized Volatility")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of 30-Day Realized Volatility")
        ax2.legend()
        ax2.grid(True, linestyle='--', linewidth=0.5)
        st.pyplot(fig2)

with tab4:
    st.subheader("Kelly Log Growth & Monte Carlo Portfolio Simulation")

    cryptos = ["bitcoin", "ethereum", "solana"]
    symbols = ["BTC", "ETH", "SOL"]
    kelly_curves = {}
    rv_values = {}
    returns_dict = {}

    for coin, sym in zip(cryptos, symbols):
        df_coin = fetch_crypto_data(coin)
        if df_coin.empty:
            continue
        df_coin['log_return'] = np.log(df_coin['close']/df_coin['close'].shift(1))
        returns = df_coin['log_return'].dropna().values
        returns_dict[sym] = returns
        probs = np.ones_like(returns)/len(returns)

        rv = np.std(returns[-30:]) * np.sqrt(365)
        rv_values[sym] = rv

        def expected_log_growth(f):
            if np.any(1 + f * returns <= 0):
                return -np.inf
            return np.sum(probs * np.log(1 + f * returns))
        f_vals = np.linspace(-2, 5, 300)
        growth_vals = [expected_log_growth(f) for f in f_vals]
        kelly_curves[sym] = (f_vals, growth_vals)

    fig, ax = plt.subplots(figsize=(10,5))
    colors = ['orange', 'blue', 'green']
    for i, sym in enumerate(symbols):
        f_vals, growth_vals = kelly_curves[sym]
        ax.plot(f_vals, growth_vals, label=f"{sym} (RV={rv_values[sym]:.2%})", color=colors[i], linewidth=2)
        max_idx = np.argmax(growth_vals)
        ax.scatter(f_vals[max_idx], growth_vals[max_idx], color=colors[i], s=50, marker='o')
    ax.set_xlabel("Fraction f")
    ax.set_ylabel("Expected Log Growth")
    ax.set_title("Kelly Log Growth Function Comparison with Realized Volatility")
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    st.pyplot(fig)

    with tab5:
        st.subheader("Monte Carlo Portfolio Optimization: BTC / ETH / SOL")
    price_frames = []
    for coin in cryptos:
        df_coin = fetch_crypto_data(coin)
        if not df_coin.empty:
            price_frames.append(df_coin["close"].rename(coin.upper()))
    if len(price_frames) == 3:
        price_df = pd.concat(price_frames, axis=1).dropna()
        returns = price_df.pct_change().dropna()
        mean_daily = returns.mean()
        cov_daily = returns.cov()
        trading_days = 365
        rf_rate = 0.0
        target_return_sortino = 0.0

        def portfolio_stats(weights):
            annual_ret = np.dot(weights, mean_daily) * trading_days
            annual_vol = np.sqrt(weights.T @ cov_daily.values @ weights) * np.sqrt(trading_days)
            sharpe = (annual_ret - rf_rate)/annual_vol if annual_vol>0 else np.nan
            return annual_ret, annual_vol, sharpe

        def sortino_ratio(weights):
            daily_target = target_return_sortino/trading_days
            port_daily = returns.dot(weights)
            downside = np.minimum(port_daily - daily_target, 0)
            dd_daily = np.sqrt(np.mean(downside**2))
            dd_annual = dd_daily * np.sqrt(trading_days)
            annual_ret = port_daily.mean()*trading_days
            sortino = (annual_ret - target_return_sortino)/dd_annual if dd_annual>0 else np.nan
            return sortino

        np.random.seed(42)
        results = []
        weights_list = []
        n_simulations = 20000
        for _ in range(n_simulations):
            w = np.random.random(len(symbols)); w/=w.sum()
            ret, vol, sharpe = portfolio_stats(w)
            sortino = sortino_ratio(w)
            results.append([ret, vol, sharpe, sortino])
            weights_list.append(w)

        pf = pd.DataFrame(results, columns=["ret","vol","sharpe","sortino"])
        for i,s in enumerate(symbols):
            pf[f"w_{s}"] = [w[i] for w in weights_list]

        max_sharpe = pf.iloc[pf["sharpe"].idxmax()]
        max_sortino = pf.iloc[pf["sortino"].idxmax()]
        min_vol = pf.iloc[pf["vol"].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1: st.write("**Max Sharpe**"); st.write(max_sharpe)
        with col2: st.write("**Max Sortino**"); st.write(max_sortino)
        with col3: st.write("**Min Volatility**"); st.write(min_vol)

        fig, ax = plt.subplots(figsize=(10,5))
        sc = ax.scatter(pf["vol"], pf["ret"], c=pf["sharpe"], s=8, cmap="viridis", alpha=0.7)
        ax.set