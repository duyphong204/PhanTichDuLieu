from bs4 import BeautifulSoup
import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="VN30 Dashboard", layout="wide")

# --- Danh sách VN30 (bạn có thể cập nhật lại cho đủ 30 mã)
vn30_tickers = [
    "FPT.VN", "VIC.VN", "VHM.VN", "VNM.VN", "MWG.VN", "HPG.VN",
    "VCB.VN", "BID.VN", "CTG.VN", "TCB.VN", "MBB.VN", "SSI.VN",
    "BVH.VN", "VRE.VN", "GAS.VN", "MSN.VN", "PNJ.VN", "HDB.VN",
    "VJC.VN", "PLX.VN", "STB.VN", "SAB.VN", "KDH.VN", "NVL.VN",
    "POW.VN", "GVR.VN", "VIB.VN", "SHB.VN", "REE.VN", "EIB.VN"
]

# Sidebar
st.sidebar.title("FinDash - VN30")
ticker = st.sidebar.selectbox("Chọn cổ phiếu (ticker)", vn30_tickers)
select_tab = st.sidebar.radio("Chọn tab", [
    'Summary', 'Chart', 'Statistics', 'Financials',
     'Monte Carlo Simulation'
])

@st.cache_data(ttl=3600)
def get_data(ticker):
    t = yf.Ticker(ticker)
    return t.info, t.history(period="1y"), t.financials, t.balance_sheet, t.cashflow, t.earnings, t.quarterly_earnings

info, hist, financials, balance_sheet, cashflow, earnings, q_earnings = get_data(ticker)

# --- Tab 1: Summary ---
def tab1():
    st.title("Summary")
    # Lấy dữ liệu từ yfinance
    @st.cache_data(ttl=3600)
    def get_stock_info(ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")  # 6 tháng gần nhất
        return info, hist

    info, hist = get_stock_info(ticker)

    # --- Hiển thị biểu đồ giá ---
    st.subheader("📈 Biểu đồ giá (6 tháng gần nhất)")
    if not hist.empty:
        fig = px.line(hist, x=hist.index, y="Close", title=f"Giá đóng cửa {ticker}")
        fig.update_layout(xaxis_title="Ngày", yaxis_title="Giá (VND)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không có dữ liệu lịch sử giá cho mã này.")

    # --- Hiển thị bảng tóm tắt thông tin ---
    st.subheader("ℹ️ Thông tin cơ bản")

    # Chọn các thông tin quan trọng từ info
    summary_data = {
        "Tên công ty": info.get("longName"),
        "Mã sàn": info.get("symbol"),
        "Ngành": info.get("sector"),
        "Quốc gia": info.get("country"),
        "Vốn hóa thị trường": info.get("marketCap"),
        "Giá hiện tại": info.get("currentPrice"),
        "EPS (TTM)": info.get("trailingEps"),
        "P/E (TTM)": info.get("trailingPE"),
        "Beta": info.get("beta"),
        "Giá cao nhất 52 tuần": info.get("fiftyTwoWeekHigh"),
        "Giá thấp nhất 52 tuần": info.get("fiftyTwoWeekLow"),
        "Số lượng nhân viên": info.get("fullTimeEmployees"),
        "Website": info.get("website"),
    }

    summary_df = pd.DataFrame(list(summary_data.items()), columns=["Thuộc tính", "Giá trị"])
    st.dataframe(summary_df, use_container_width=True)

    # --- Hiển thị mô tả công ty (nếu có) ---
    st.subheader("🏢 Giới thiệu công ty")
    longBusinessSummary = info.get("longBusinessSummary", "Không có thông tin mô tả.")
    st.write(longBusinessSummary)


# ============================
# Hàm get_history() — lấy dữ liệu linh hoạt
# ============================
@st.cache_data(ttl=3600)
def get_history(ticker_symbol, start=None, end=None, period=None, interval='1d'):
    try:
        # tự động thêm .VN nếu chưa có
        yf_sym = ticker_symbol if ticker_symbol.endswith('.VN') else f"{ticker_symbol}.VN"

        if period is not None and period != '-':
            hist = yf.Ticker(yf_sym).history(period=period, interval=interval, auto_adjust=False)
        else:
            hist = yf.Ticker(yf_sym).history(start=start, end=end, interval=interval, auto_adjust=False)

        if not hist.empty:
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'])
            if hasattr(hist['Date'].dt, 'tz'):
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            return hist
    except Exception:
        try:
            hist = yf.Ticker(ticker_symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date'])
                if hasattr(hist['Date'].dt, 'tz'):
                    hist['Date'] = hist['Date'].dt.tz_localize(None)
                return hist
        except Exception:
            pass
    return pd.DataFrame()
# ============================
# Tab 2: Chart — giống bản chuẩn FinancialDashboard
# ============================
def tab2():
    st.title("Chart")
    st.write(f"Ticker: **{ticker}**")
    if ticker in ['-', '']:
        st.info("Vui lòng chọn mã cổ phiếu.")
        return

    # --- Controls ---
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        start_date = st.date_input("Start date", datetime.today().date() - timedelta(days=180))
    with c2:
        end_date = st.date_input("End date", datetime.today().date())
    with c3:
        period_select = st.selectbox("Period (tùy chọn)", ['-', '1mo', '3mo', '6mo', '1y', '3y', '5y', 'max'], index=0)

    interval = st.selectbox("Interval", ['1d', '1wk', '1mo'], index=0)
    plot_type = st.selectbox("Plot type", ['Line', 'Candle'], index=0)
    show_sma = st.checkbox("SMA50", value=True)
    show_volume = st.checkbox("Volume", value=True)

    # --- Lấy dữ liệu ---
    df = get_history(
        ticker_symbol=ticker,
        start=start_date,
        end=end_date,
        period=(None if period_select == '-' else period_select),
        interval=interval
    )

    if df.empty or 'Close' not in df.columns:
        st.warning("Không có dữ liệu lịch sử cho khoảng chọn.")
        return

    # --- SMA50 ---
    if show_sma:
        df['SMA50'] = df['Close'].rolling(window=50).mean()

    # --- Vẽ chart ---
    secondary_y = show_volume and 'Volume' in df.columns
    fig = make_subplots(specs=[[{"secondary_y": secondary_y}]])
    if plot_type == 'Line':
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'), secondary_y=False)
    else:
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'
        ), secondary_y=False)

    # --- SMA50 ---
    if show_sma and 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA50'], name='SMA50', line=dict(dash='dash', color='orange')
        ), secondary_y=False)

    # --- Volume ---
    if secondary_y:
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', opacity=0.3), secondary_y=True)

    # --- Layout ---
    fig.update_layout(
        title=f"{ticker} - Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right") if secondary_y else None,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Statistics ---
def tab3():
    st.title(f"📊 Financial Statistics - {ticker}")

    metrics = {
        "Vốn hóa thị trường": info.get("marketCap"),
        "Giá hiện tại": info.get("currentPrice"),
        "EPS (TTM)": info.get("trailingEps"),
        "P/E (TTM)": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "ROE (Return on Equity)": info.get("returnOnEquity"),
        "ROA (Return on Assets)": info.get("returnOnAssets"),
        "Tổng nợ / Tổng tài sản": info.get("debtToEquity"),
        "Biên lợi nhuận gộp": info.get("grossMargins"),
        "Biên lợi nhuận ròng": info.get("profitMargins"),
        "Tỷ suất cổ tức": info.get("dividendYield"),
        "Beta": info.get("beta"),
    }

    st.dataframe(pd.DataFrame(list(metrics.items()), columns=["Chỉ số", "Giá trị"]))

    st.subheader("📈 Diễn biến giá 1 năm qua")
    if not hist.empty:
        fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} - Giá đóng cửa (1 năm)")
        st.plotly_chart(fig, use_container_width=True)



# --- Tab 4: Financials ---
def tab4():
    # if tab == "Financials":
    st.title(f"💰 Financial Statements - {ticker}")

    type_select = st.radio("Chọn báo cáo", ["Income Statement", "Balance Sheet", "Cash Flow"])

    if type_select == "Income Statement":
        st.subheader("Báo cáo Kết quả Kinh doanh")
        st.dataframe(financials)

    elif type_select == "Balance Sheet":
        st.subheader("Bảng Cân đối Kế toán")
        st.dataframe(balance_sheet)

    else:
        st.subheader("Báo cáo Dòng tiền")
        st.dataframe(cashflow)

# --- Tab 6: Monte Carlo ---
def tab5():
    st.title(f"🎲 Monte Carlo Simulation - {ticker}")

    c1, c2 = st.columns(2)
    with c1:
        simulations = st.number_input("Số lượng mô phỏng", min_value=100, max_value=2000, value=500, step=100)
    with c2:
        days = st.number_input("Số ngày dự báo", min_value=10, max_value=365, value=90, step=10)

    @st.cache_data(ttl=600)
    def montecarlo(ticker, days, simulations):
        ## chỉ lấy 6 tháng gần nhất để dự đoán vaR
        data = yf.download(ticker, period="6mo", progress=False)
        if data.empty:
            return None

        close_price = data["Close"]
        returns = close_price.pct_change().dropna()
        last_price = close_price.iloc[-1]
        daily_vol = np.std(returns)

        simulation_df = pd.DataFrame()
        for i in range(simulations):
            prices = [last_price]
            for _ in range(days):
                future_return = np.random.normal(0, daily_vol)
                future_price = prices[-1] * (1 + future_return)
                prices.append(future_price)
            simulation_df[i] = prices

        return simulation_df

    sim_data = montecarlo(ticker, days, simulations)

    if sim_data is None:
        st.error("Không thể lấy dữ liệu giá cổ phiếu.")
        return

    # --- Vẽ mô phỏng ---
    st.subheader("📈 Biểu đồ mô phỏng giá cổ phiếu")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(sim_data)
    ax.set_title(f"Monte Carlo {simulations} lần mô phỏng - {ticker}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá cổ phiếu (VND)")
    st.pyplot(fig)

    # --- Phân phối giá cuối cùng ---
    st.subheader("📊 Phân phối giá cuối cùng sau mô phỏng")
    ending_prices = sim_data.iloc[-1, :].to_numpy(dtype=float)
    ending_prices = ending_prices[~np.isnan(ending_prices)]

    if ending_prices.size == 0:
        st.error("Không có dữ liệu hợp lệ để tính VaR.")
        return

    var_95 = np.percentile(ending_prices, 5)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(ending_prices, bins=50, alpha=0.7, color='skyblue')
    ax2.axvline(var_95, color='red', linestyle='--', linewidth=1.5, label="VaR 95%")
    ax2.legend()
    ax2.set_title("Phân phối giá cuối cùng & VaR 95%")
    st.pyplot(fig2)

    # --- Tính Value at Risk ---
    current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    VaR = current_price - var_95

    st.success(f"📉 Value at Risk (95%): {VaR:,.0f} VND")
    st.write(f"Giá hiện tại: {current_price:,.0f} VND — Dự báo 5% tệ nhất: {var_95:,.0f} VND")

# Main switch
if select_tab == 'Summary':
    tab1()
elif select_tab == 'Chart':
    tab2()
elif select_tab == 'Statistics':
    tab3()
elif select_tab == 'Financials':
    tab4()
elif select_tab == 'Monte Carlo Simulation':
    tab5()
