
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    st_autorefresh = None
    AUTOREFRESH_AVAILABLE = False


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Telegram Trading Signal Scanner",
    page_icon="📡",
    layout="wide"
)


# =====================================================
# CONFIG
# =====================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TELEGRAM_SENT_LOG_FILE = DATA_DIR / "telegram_sent_signals.csv"

PAIRS = {
    "EURUSD": {
        "ticker": "EURUSD=X",
        "pip": 0.0001,
        "contract": 100000,
        "decimals": 5,
        "default": 1.10000,
    },
    "XAUUSD": {
        "ticker": "GC=F",
        "pip": 0.01,
        "contract": 100,
        "decimals": 2,
        "default": 2350.00,
    },
    "BTCUSD": {
        "ticker": "BTC-USD",
        "pip": 1.0,
        "contract": 1,
        "decimals": 2,
        "default": 78000.00,
    },
}

SCAN_MARKETS = ["EURUSD", "XAUUSD", "BTCUSD"]

RR_OPTIONS = {
    "1:1": 1.0,
    "1:1.5": 1.5,
    "1:2": 2.0,
    "1:2.5": 2.5,
    "1:3": 3.0,
    "1:4": 4.0,
}

BUY_BG = "#dcfce7"
SELL_BG = "#fee2e2"
NEUTRAL_BG = "#ffedd5"


# =====================================================
# CSS
# =====================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
}
.main-title {
    font-size: 34px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 0.15rem;
}
.subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 1rem;
}
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    margin-bottom: 14px;
}
.stButton > button {
    border-radius: 12px;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================
# BASIC HELPERS
# =====================================================

def fmt_price(market, value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{PAIRS[market]['decimals']}f}"


def price_step(market):
    if market == "BTCUSD":
        return 1.0
    if market == "XAUUSD":
        return 0.01
    if "JPY" in market:
        return 0.01
    return 0.0001


def color_rows(df):
    def apply(row):
        signal = str(row.get("Signal", "")).upper()
        direction = str(row.get("Direction", "")).upper()

        if signal == "BUY" or direction == "BUY":
            return [f"background-color: {BUY_BG}; color: #065f46"] * len(row)

        if signal == "SELL" or direction == "SELL":
            return [f"background-color: {SELL_BG}; color: #991b1b"] * len(row)

        return [f"background-color: {NEUTRAL_BG}; color: #9a3412"] * len(row)

    return df.style.apply(apply, axis=1)


# =====================================================
# MARKET DATA
# =====================================================

@st.cache_data(ttl=90)
def get_live_price(market):
    fallback = PAIRS[market]["default"]
    ticker = PAIRS[market]["ticker"]

    try:
        tk = yf.Ticker(ticker)
        fast = getattr(tk, "fast_info", None)

        if fast is not None:
            try:
                last = fast.get("last_price", None)
            except Exception:
                last = None

            if last is not None and not pd.isna(last) and float(last) > 0:
                return float(last), "Yahoo fast_info"

    except Exception:
        pass

    try:
        raw = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)

        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close = pd.to_numeric(raw["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1]), "Yahoo latest close"

    except Exception:
        pass

    return fallback, "Fallback default"


def clean_data(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df.sort_values("Date").reset_index(drop=True)

    return df


@st.cache_data(ttl=600)
def load_market_data(market, daily_start):
    ticker = PAIRS[market]["ticker"]

    daily_raw = yf.download(ticker, start=daily_start, interval="1d", progress=False, auto_adjust=False)
    h1_raw = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=False)
    m15_raw = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=False)

    return clean_data(daily_raw), clean_data(h1_raw), clean_data(m15_raw)


def add_cameroon_time(df, start_hour, end_hour):
    df = df.copy()
    local = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert("Africa/Douala")
    df["Cameroon_Time"] = local.dt.strftime("%Y-%m-%d %H:%M")
    hour = local.dt.hour + (local.dt.minute / 60)

    if start_hour < end_hour:
        df["Trading_Window"] = (hour >= start_hour) & (hour < end_hour)
    else:
        df["Trading_Window"] = (hour >= start_hour) | (hour < end_hour)

    return df


# =====================================================
# INDICATORS
# =====================================================

def add_rsi(df, period=14):
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    df = df.copy()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def add_atr(df, period=14):
    df = df.copy()
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(period).mean()
    return df


def add_ema(df):
    df = df.copy()
    for span in [9, 20, 21, 50, 200]:
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


def add_sma(df):
    df = df.copy()
    for span in [20, 50, 200]:
        df[f"SMA_{span}"] = df["Close"].rolling(span).mean()
    return df


# =====================================================
# STRATEGY: FX 15M MOMENTUM
# =====================================================

def build_fx15_strategy(daily, h1, m15, rr, atr_mult, start_hour, end_hour, enforce_window):
    daily = add_sma(daily)
    daily = add_rsi(daily)
    daily = add_macd(daily)

    daily["Daily_Bias"] = "Neutral"
    daily.loc[
        (daily["Close"] > daily["SMA_200"])
        & (daily["SMA_20"] > daily["SMA_50"])
        & (daily["RSI_14"] > 50)
        & (daily["MACD"] > daily["MACD_Signal"]),
        "Daily_Bias",
    ] = "Bullish"

    daily.loc[
        (daily["Close"] < daily["SMA_200"])
        & (daily["SMA_20"] < daily["SMA_50"])
        & (daily["RSI_14"] < 50)
        & (daily["MACD"] < daily["MACD_Signal"]),
        "Daily_Bias",
    ] = "Bearish"

    daily["Daily_Bias"] = daily["Daily_Bias"].shift(1).fillna("Neutral")

    h1 = add_ema(h1)
    h1 = add_rsi(h1)
    h1 = add_macd(h1)

    h1["H1_Structure"] = "Neutral"
    h1.loc[
        (h1["EMA_9"] > h1["EMA_21"])
        & (h1["EMA_21"] > h1["EMA_50"])
        & (h1["RSI_14"] > 50)
        & (h1["MACD"] > h1["MACD_Signal"]),
        "H1_Structure",
    ] = "Bullish"

    h1.loc[
        (h1["EMA_9"] < h1["EMA_21"])
        & (h1["EMA_21"] < h1["EMA_50"])
        & (h1["RSI_14"] < 50)
        & (h1["MACD"] < h1["MACD_Signal"]),
        "H1_Structure",
    ] = "Bearish"

    h1["H1_Structure"] = h1["H1_Structure"].shift(1).fillna("Neutral")

    m15 = add_ema(m15)
    m15 = add_rsi(m15)
    m15 = add_macd(m15)
    m15 = add_atr(m15)

    daily_state = daily[["Date", "Daily_Bias"]].dropna().sort_values("Date")
    h1_state = h1[["Date", "H1_Structure"]].dropna().sort_values("Date")
    m15 = m15.sort_values("Date").reset_index(drop=True)

    if not daily_state.empty:
        m15 = pd.merge_asof(m15, daily_state, on="Date", direction="backward")
    else:
        m15["Daily_Bias"] = "Neutral"

    if not h1_state.empty:
        m15 = pd.merge_asof(m15, h1_state, on="Date", direction="backward")
    else:
        m15["H1_Structure"] = "Neutral"

    buy = (
        (m15["Daily_Bias"] == "Bullish")
        & (m15["H1_Structure"] == "Bullish")
        & (m15["EMA_9"] > m15["EMA_21"])
        & (m15["RSI_14"] > 50)
        & (m15["MACD"] > m15["MACD_Signal"])
    )

    sell = (
        (m15["Daily_Bias"] == "Bearish")
        & (m15["H1_Structure"] == "Bearish")
        & (m15["EMA_9"] < m15["EMA_21"])
        & (m15["RSI_14"] < 50)
        & (m15["MACD"] < m15["MACD_Signal"])
    )

    m15["Strategy"] = "FX 15m Momentum"
    m15["Signal"] = "NEUTRAL"
    m15["Direction"] = "NEUTRAL"
    m15["Reason"] = "No aligned 15m momentum setup"
    m15["Entry"] = np.nan
    m15["SL"] = np.nan
    m15["TP"] = np.nan

    risk_dist = atr_mult * m15["ATR_14"]

    m15.loc[buy, "Signal"] = "BUY"
    m15.loc[buy, "Direction"] = "BUY"
    m15.loc[buy, "Reason"] = "Daily bullish + 1H bullish + 15m momentum confirmation"
    m15.loc[buy, "Entry"] = m15["Close"]
    m15.loc[buy, "SL"] = m15["Close"] - risk_dist
    m15.loc[buy, "TP"] = m15["Close"] + risk_dist * rr

    m15.loc[sell, "Signal"] = "SELL"
    m15.loc[sell, "Direction"] = "SELL"
    m15.loc[sell, "Reason"] = "Daily bearish + 1H bearish + 15m momentum confirmation"
    m15.loc[sell, "Entry"] = m15["Close"]
    m15.loc[sell, "SL"] = m15["Close"] + risk_dist
    m15.loc[sell, "TP"] = m15["Close"] - risk_dist * rr

    m15 = add_cameroon_time(m15, start_hour, end_hour)

    if enforce_window:
        outside = ~m15["Trading_Window"]
        m15.loc[outside & m15["Signal"].isin(["BUY", "SELL"]), ["Signal", "Direction", "Reason"]] = [
            "NEUTRAL",
            "NEUTRAL",
            "Outside Cameroon watch time",
        ]
        m15.loc[outside, ["Entry", "SL", "TP"]] = np.nan

    return daily, h1, m15


# =====================================================
# STRATEGY: SMC MARKET STRUCTURE
# =====================================================

def add_swings(df, n):
    df = df.copy()
    df["Swing_High"] = np.nan
    df["Swing_Low"] = np.nan

    if len(df) < n * 2 + 1:
        df["Last_Swing_High"] = np.nan
        df["Last_Swing_Low"] = np.nan
        return df

    for i in range(n, len(df) - n):
        if df["High"].iloc[i] == df["High"].iloc[i - n : i + n + 1].max():
            df.loc[df.index[i], "Swing_High"] = df["High"].iloc[i]
        if df["Low"].iloc[i] == df["Low"].iloc[i - n : i + n + 1].min():
            df.loc[df.index[i], "Swing_Low"] = df["Low"].iloc[i]

    df["Last_Swing_High"] = df["Swing_High"].ffill()
    df["Last_Swing_Low"] = df["Swing_Low"].ffill()
    return df


def add_structure(df):
    df = df.copy()
    prev_high = df["Last_Swing_High"].shift(1)
    prev_low = df["Last_Swing_Low"].shift(1)

    df["BOS_Bullish"] = (df["Close"] > prev_high) & prev_high.notna()
    df["BOS_Bearish"] = (df["Close"] < prev_low) & prev_low.notna()

    df["Structure"] = "Neutral"
    state = "Neutral"

    for i in range(len(df)):
        if bool(df["BOS_Bullish"].iloc[i]):
            state = "Bullish"
        elif bool(df["BOS_Bearish"].iloc[i]):
            state = "Bearish"
        df.loc[df.index[i], "Structure"] = state

    df["CHOCH_Bullish"] = (df["Structure"].shift(1) == "Bearish") & (df["Structure"] == "Bullish")
    df["CHOCH_Bearish"] = (df["Structure"].shift(1) == "Bullish") & (df["Structure"] == "Bearish")
    df["Sell_Side_Sweep"] = (df["Low"] < prev_low) & (df["Close"] > prev_low) & prev_low.notna()
    df["Buy_Side_Sweep"] = (df["High"] > prev_high) & (df["Close"] < prev_high) & prev_high.notna()

    return df


def prep_smc(df, swing_len):
    df = add_atr(df)
    df = add_ema(df)
    df = add_swings(df, swing_len)
    df = add_structure(df)
    df["Equilibrium"] = (df["Last_Swing_High"] + df["Last_Swing_Low"]) / 2
    df["Premium_Discount"] = "Neutral"
    df.loc[df["Close"] < df["Equilibrium"], "Premium_Discount"] = "Discount"
    df.loc[df["Close"] > df["Equilibrium"], "Premium_Discount"] = "Premium"
    return df


def build_smc_strategy(daily, h1, m15, rr, atr_mult, swing_len, strict_mode, start_hour, end_hour, enforce_window):
    daily = prep_smc(daily, swing_len)
    h1 = prep_smc(h1, swing_len)
    m15 = prep_smc(m15, swing_len)

    daily["Daily_Bias"] = daily["Structure"].shift(1).fillna("Neutral")
    h1["H1_Structure"] = h1["Structure"].shift(1).fillna("Neutral")

    daily_state = daily[["Date", "Daily_Bias"]].dropna().sort_values("Date")
    h1_state = h1[["Date", "H1_Structure"]].dropna().sort_values("Date")
    m15 = m15.sort_values("Date").reset_index(drop=True)

    if not daily_state.empty:
        m15 = pd.merge_asof(m15, daily_state, on="Date", direction="backward")
    else:
        m15["Daily_Bias"] = "Neutral"

    if not h1_state.empty:
        m15 = pd.merge_asof(m15, h1_state, on="Date", direction="backward")
    else:
        m15["H1_Structure"] = "Neutral"

    bullish_trigger = m15["Sell_Side_Sweep"] | m15["CHOCH_Bullish"] | m15["BOS_Bullish"]
    bearish_trigger = m15["Buy_Side_Sweep"] | m15["CHOCH_Bearish"] | m15["BOS_Bearish"]

    if strict_mode:
        buy = (
            (m15["Daily_Bias"] == "Bullish")
            & (m15["H1_Structure"] == "Bullish")
            & bullish_trigger
            & (m15["Premium_Discount"] == "Discount")
        )

        sell = (
            (m15["Daily_Bias"] == "Bearish")
            & (m15["H1_Structure"] == "Bearish")
            & bearish_trigger
            & (m15["Premium_Discount"] == "Premium")
        )
    else:
        buy = (m15["H1_Structure"] == "Bullish") & bullish_trigger
        sell = (m15["H1_Structure"] == "Bearish") & bearish_trigger

    m15["Strategy"] = "SMC Market Structure"
    m15["Signal"] = "NEUTRAL"
    m15["Direction"] = "NEUTRAL"
    m15["Reason"] = "No valid SMC confluence"
    m15["Entry"] = np.nan
    m15["SL"] = np.nan
    m15["TP"] = np.nan

    m15.loc[buy, "Signal"] = "BUY"
    m15.loc[buy, "Direction"] = "BUY"
    m15.loc[buy, "Reason"] = "Bullish structure + liquidity/CHOCH/BOS trigger"
    m15.loc[buy, "Entry"] = m15["Close"]

    m15.loc[sell, "Signal"] = "SELL"
    m15.loc[sell, "Direction"] = "SELL"
    m15.loc[sell, "Reason"] = "Bearish structure + liquidity/CHOCH/BOS trigger"
    m15.loc[sell, "Entry"] = m15["Close"]

    buy_sl = np.minimum(m15["Last_Swing_Low"], m15["Close"] - atr_mult * m15["ATR_14"])
    sell_sl = np.maximum(m15["Last_Swing_High"], m15["Close"] + atr_mult * m15["ATR_14"])

    m15.loc[buy, "SL"] = buy_sl
    m15.loc[sell, "SL"] = sell_sl

    buy_risk = m15["Close"] - m15["SL"]
    sell_risk = m15["SL"] - m15["Close"]

    m15.loc[buy, "TP"] = m15["Close"] + buy_risk * rr
    m15.loc[sell, "TP"] = m15["Close"] - sell_risk * rr

    invalid_buy = (m15["Signal"] == "BUY") & ((m15["SL"] >= m15["Entry"]) | (m15["TP"] <= m15["Entry"]))
    invalid_sell = (m15["Signal"] == "SELL") & ((m15["SL"] <= m15["Entry"]) | (m15["TP"] >= m15["Entry"]))
    invalid = invalid_buy | invalid_sell

    m15.loc[invalid, ["Signal", "Direction", "Reason"]] = ["NEUTRAL", "NEUTRAL", "Invalid SL/TP"]
    m15.loc[invalid, ["Entry", "SL", "TP"]] = np.nan

    m15 = add_cameroon_time(m15, start_hour, end_hour)

    if enforce_window:
        outside = ~m15["Trading_Window"]
        m15.loc[outside & m15["Signal"].isin(["BUY", "SELL"]), ["Signal", "Direction", "Reason"]] = [
            "NEUTRAL",
            "NEUTRAL",
            "Outside Cameroon watch time",
        ]
        m15.loc[outside, ["Entry", "SL", "TP"]] = np.nan

    return daily, h1, m15


# =====================================================
# TELEGRAM
# =====================================================

def load_telegram_sent_log():
    if TELEGRAM_SENT_LOG_FILE.exists():
        try:
            return pd.read_csv(TELEGRAM_SENT_LOG_FILE)
        except Exception:
            return pd.DataFrame(columns=["Signal_ID", "Timestamp", "Strategy", "Market", "Direction", "Message"])
    return pd.DataFrame(columns=["Signal_ID", "Timestamp", "Strategy", "Market", "Direction", "Message"])


def save_telegram_sent_signal(signal_id, strategy, market, direction, message):
    log = load_telegram_sent_log()

    if "Signal_ID" in log.columns and signal_id in set(log["Signal_ID"].astype(str)):
        return

    new_row = pd.DataFrame(
        [
            {
                "Signal_ID": signal_id,
                "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Strategy": strategy,
                "Market": market,
                "Direction": direction,
                "Message": message,
            }
        ]
    )

    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(TELEGRAM_SENT_LOG_FILE, index=False)


def already_sent_telegram_signal(signal_id):
    log = load_telegram_sent_log()

    if log.empty or "Signal_ID" not in log.columns:
        return False

    return signal_id in set(log["Signal_ID"].astype(str))


def clear_telegram_sent_log():
    if TELEGRAM_SENT_LOG_FILE.exists():
        TELEGRAM_SENT_LOG_FILE.unlink()


def make_telegram_signal_id(strategy, market, direction, entry, sl, tp, source_time=""):
    return (
        f"TG|{strategy}|{market}|{direction}|"
        f"{round(float(entry), 8)}|{round(float(sl), 8)}|{round(float(tp), 8)}|{source_time}"
    )


def send_telegram_message(bot_token, chat_id, message):
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()

    if not bot_token:
        return False, "Telegram BOT_TOKEN is missing."

    if not chat_id:
        return False, "Telegram CHAT_ID is missing."

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = urllib.parse.urlencode(
            {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")

        request = urllib.request.Request(url, data=payload, method="POST")

        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="ignore")

        if '"ok":true' in body:
            return True, "Telegram message sent successfully."

        return False, f"Telegram API returned: {body[:300]}"

    except Exception as e:
        return False, f"Telegram send failed: {e}"


def format_telegram_signal_message(strategy, market, direction, entry, sl, tp, risk_percent, rr_label, cameroon_time="", reason=""):
    emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "🟠"

    lines = [
        f"{emoji} <b>NEW TRADE SIGNAL</b>",
        "",
        f"<b>Strategy:</b> {strategy}",
        f"<b>Market:</b> {market}",
        f"<b>Direction:</b> {direction}",
        f"<b>Entry:</b> {entry}",
        f"<b>Stop Loss:</b> {sl}",
        f"<b>Take Profit:</b> {tp}",
        f"<b>Risk:</b> {risk_percent}%",
        f"<b>Risk Reward:</b> {rr_label}",
    ]

    if cameroon_time:
        lines.append(f"<b>Cameroon Time:</b> {cameroon_time}")

    if reason:
        lines.extend(["", f"<b>Reason:</b> {reason}"])

    lines.extend(["", "⚠️ Confirm broker price, spread, and news risk before trading."])

    return "\n".join(lines)


# =====================================================
# SCANNER
# =====================================================

def build_strategy_for_market(scan_market, selected_strategy, daily_start, rr_ratio, atr_mult, swing_len, strict_mode, session_start, session_end, enforce_session):
    d_daily, d_h1, d_m15_raw = load_market_data(scan_market, str(daily_start))

    if d_daily.empty or d_h1.empty or d_m15_raw.empty:
        return d_daily, d_h1, pd.DataFrame(), None, "Not enough data"

    if selected_strategy == "SMC Market Structure":
        d_daily, d_h1, d_m15 = build_smc_strategy(
            d_daily,
            d_h1,
            d_m15_raw,
            rr_ratio,
            atr_mult,
            swing_len,
            strict_mode,
            session_start,
            session_end,
            enforce_session,
        )
    else:
        d_daily, d_h1, d_m15 = build_fx15_strategy(
            d_daily,
            d_h1,
            d_m15_raw,
            rr_ratio,
            atr_mult,
            session_start,
            session_end,
            enforce_session,
        )

    valid = d_m15.dropna(subset=["Close"])
    latest_row = valid.iloc[-1] if not valid.empty else None

    return d_daily, d_h1, d_m15, latest_row, None


def scan_all_markets(selected_strategy, daily_start, rr_ratio, atr_mult, swing_len, strict_mode, session_start, session_end, enforce_session):
    rows = []
    cache = {}

    for scan_market in SCAN_MARKETS:
        try:
            d_daily, d_h1, d_m15, latest_row, err = build_strategy_for_market(
                scan_market,
                selected_strategy,
                daily_start,
                rr_ratio,
                atr_mult,
                swing_len,
                strict_mode,
                session_start,
                session_end,
                enforce_session,
            )

            cache[scan_market] = {"daily": d_daily, "h1": d_h1, "m15": d_m15, "latest": latest_row, "error": err}

            if err or latest_row is None:
                rows.append(
                    {
                        "Market": scan_market,
                        "Strategy": selected_strategy,
                        "Signal": "NEUTRAL",
                        "Direction": "NEUTRAL",
                        "Entry": np.nan,
                        "SL": np.nan,
                        "TP": np.nan,
                        "Cameroon_Time": "N/A",
                        "Trading_Window": False,
                        "Reason": err or "No latest candle",
                    }
                )
                continue

            rows.append(
                {
                    "Market": scan_market,
                    "Strategy": selected_strategy,
                    "Signal": latest_row.get("Signal", "NEUTRAL"),
                    "Direction": latest_row.get("Direction", "NEUTRAL"),
                    "Entry": latest_row.get("Entry", np.nan),
                    "SL": latest_row.get("SL", np.nan),
                    "TP": latest_row.get("TP", np.nan),
                    "Cameroon_Time": latest_row.get("Cameroon_Time", "N/A"),
                    "Trading_Window": bool(latest_row.get("Trading_Window", False)),
                    "Reason": latest_row.get("Reason", ""),
                }
            )

        except Exception as e:
            cache[scan_market] = {"daily": pd.DataFrame(), "h1": pd.DataFrame(), "m15": pd.DataFrame(), "latest": None, "error": str(e)}
            rows.append(
                {
                    "Market": scan_market,
                    "Strategy": selected_strategy,
                    "Signal": "NEUTRAL",
                    "Direction": "NEUTRAL",
                    "Entry": np.nan,
                    "SL": np.nan,
                    "TP": np.nan,
                    "Cameroon_Time": "N/A",
                    "Trading_Window": False,
                    "Reason": str(e),
                }
            )

    return pd.DataFrame(rows), cache


def send_scanner_telegram_alerts(scanner_df, strategy, risk_percent, rr_label, bot_token, chat_id, enable_telegram, auto_send):
    statuses = []

    if not enable_telegram or not auto_send:
        return statuses

    if scanner_df.empty:
        return statuses

    active = scanner_df[scanner_df["Signal"].isin(["BUY", "SELL"])].copy()

    if active.empty:
        return statuses

    for _, row in active.iterrows():
        market = row["Market"]
        direction = row["Direction"]
        entry = row["Entry"]
        sl = row["SL"]
        tp = row["TP"]
        cameroon_time = str(row.get("Cameroon_Time", ""))

        if pd.isna(entry) or pd.isna(sl) or pd.isna(tp):
            statuses.append(f"{market}: skipped because Entry/SL/TP is missing.")
            continue

        signal_id = make_telegram_signal_id(strategy, market, direction, entry, sl, tp, cameroon_time)

        if already_sent_telegram_signal(signal_id):
            statuses.append(f"{market}: duplicate Telegram signal skipped.")
            continue

        message = format_telegram_signal_message(
            strategy=strategy,
            market=market,
            direction=direction,
            entry=fmt_price(market, entry),
            sl=fmt_price(market, sl),
            tp=fmt_price(market, tp),
            risk_percent=risk_percent,
            rr_label=rr_label,
            cameroon_time=cameroon_time,
            reason=str(row.get("Reason", "")),
        )

        ok, response = send_telegram_message(bot_token, chat_id, message)

        if ok:
            save_telegram_sent_signal(signal_id, strategy, market, direction, response)

        statuses.append(f"{market}: {response}")

    return statuses


# =====================================================
# CHARTS
# =====================================================

def candle_chart(df, title):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for col in ["EMA_20", "EMA_50", "EMA_200", "SMA_20", "SMA_50", "SMA_200"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))

    if "Signal" in df.columns:
        buys = df[df["Signal"] == "BUY"]
        sells = df[df["Signal"] == "SELL"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["Date"],
                    y=buys["Entry"],
                    mode="markers",
                    name="BUY",
                    marker=dict(size=12, symbol="triangle-up", color="green"),
                )
            )

        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["Date"],
                    y=sells["Entry"],
                    mode="markers",
                    name="SELL",
                    marker=dict(size=12, symbol="triangle-down", color="red"),
                )
            )

    fig.update_layout(title=title, height=560, xaxis_rangeslider_visible=False)
    return fig


# =====================================================
# APP
# =====================================================

st.markdown('<div class="main-title">📡 Telegram Trading Signal Scanner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Streamlit Cloud friendly scanner for EURUSD, XAUUSD, and BTCUSD. No MT5. No trade execution. Telegram signals only.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scanner Controls")

    strategy = st.radio(
        "Trading strategy",
        ["SMC Market Structure", "FX 15m Momentum"],
        horizontal=False,
    )

    chart_market = st.selectbox("Chart market", SCAN_MARKETS, index=0)
    daily_start = st.date_input("Daily data start", value=pd.to_datetime("2020-01-01"))

    st.divider()
    st.subheader("Risk Display")
    risk_percent = st.selectbox("Risk % displayed in alerts", [0.25, 0.5, 1.0, 1.5, 2.0], index=2)
    rr_label = st.selectbox("Risk Reward", list(RR_OPTIONS.keys()), index=2)
    rr_ratio = RR_OPTIONS[rr_label]

    st.divider()
    st.subheader("Strategy Settings")
    atr_mult = st.selectbox("ATR safety buffer", [0.5, 1.0, 1.5, 2.0], index=1)
    swing_len = st.selectbox("SMC swing sensitivity", [2, 3, 4, 5], index=1)
    strict_mode = st.checkbox("SMC strict Daily + 1H alignment", value=True)

    st.divider()
    st.subheader("Cameroon Watch Time")
    enforce_session = st.checkbox("Only allow signals during watch time", value=True)
    session_start = st.selectbox("Start", list(range(0, 24)), index=6, format_func=lambda x: f"{x:02d}:00 Cameroon")
    session_end = st.selectbox("Stop", list(range(1, 25)), index=21, format_func=lambda x: f"{x if x < 24 else 0:02d}:00 Cameroon")

    st.divider()
    st.subheader("Telegram Alerts")

    enable_telegram = st.checkbox("Enable Telegram alerts", value=True)
    auto_send_telegram = st.checkbox("Auto-send BUY/SELL signals to Telegram", value=True)

    try:
        default_bot = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
        default_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")
    except Exception:
        default_bot = ""
        default_chat = ""

    bot_token = st.text_input("Telegram BOT_TOKEN", value=default_bot, type="password")
    chat_id = st.text_input("Telegram CHAT_ID", value=default_chat)

    if st.button("Send Telegram Test Message"):
        ok, msg = send_telegram_message(bot_token, chat_id, "✅ Telegram test from Streamlit Trading Signal Scanner.")
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    if st.button("Clear Telegram sent-signal log"):
        clear_telegram_sent_log()
        st.success("Telegram sent-signal log cleared.")

    st.divider()
    st.subheader("Auto Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_minutes = st.selectbox("Refresh every", [1, 3, 5, 10, 15], index=1)

    if auto_refresh:
        if AUTOREFRESH_AVAILABLE and st_autorefresh is not None:
            st_autorefresh(
                interval=refresh_minutes * 60 * 1000,
                key="telegram_signal_scanner_autorefresh",
            )
            st.caption(f"Auto-refresh is active every {refresh_minutes} minute(s).")
        else:
            st.warning("streamlit-autorefresh is not installed.")

    if st.button("Refresh market data now"):
        st.cache_data.clear()
        st.rerun()


# Live prices
live_cols = st.columns(3)
for idx, market in enumerate(SCAN_MARKETS):
    price, source = get_live_price(market)
    live_cols[idx].metric(f"{market} live price", fmt_price(market, price))
    live_cols[idx].caption(source)

# Scan all markets
scanner_df, scanner_cache = scan_all_markets(
    selected_strategy=strategy,
    daily_start=daily_start,
    rr_ratio=rr_ratio,
    atr_mult=atr_mult,
    swing_len=swing_len,
    strict_mode=strict_mode,
    session_start=session_start,
    session_end=session_end,
    enforce_session=enforce_session,
)

telegram_statuses = send_scanner_telegram_alerts(
    scanner_df=scanner_df,
    strategy=strategy,
    risk_percent=risk_percent,
    rr_label=rr_label,
    bot_token=bot_token,
    chat_id=chat_id,
    enable_telegram=enable_telegram,
    auto_send=auto_send_telegram,
)

# Dashboard
st.subheader("All-Pair Signal Scanner")
st.caption("The app scans EURUSD, XAUUSD, and BTCUSD after every refresh. It sends Telegram alerts only for active BUY/SELL signals.")

if scanner_df.empty:
    st.info("Scanner has no data yet.")
else:
    display_df = scanner_df.copy()
    for col in ["Entry", "SL", "TP"]:
        display_df[col] = display_df.apply(lambda r: fmt_price(r["Market"], r[col]) if pd.notna(r[col]) else "N/A", axis=1)
    st.dataframe(color_rows(display_df), use_container_width=True)

active = scanner_df[scanner_df["Signal"].isin(["BUY", "SELL"])]
if active.empty:
    st.info("No active BUY/SELL signals right now.")
else:
    st.success(f"Active signals found: {len(active)}")

if telegram_statuses:
    st.subheader("Telegram Status")
    for status in telegram_statuses:
        st.info(status)

st.divider()

# Chart and details
st.subheader(f"{chart_market} Chart and Latest Setups")

cache_item = scanner_cache.get(chart_market, {})
m15 = cache_item.get("m15", pd.DataFrame())

if m15 is None or m15.empty:
    st.warning(f"No chart data available for {chart_market}.")
else:
    st.plotly_chart(candle_chart(m15.tail(250), f"{chart_market} 15m Chart - {strategy}"), use_container_width=True)

    setups = m15[m15["Signal"].isin(["BUY", "SELL"])].dropna(subset=["Entry", "SL", "TP"]).tail(30)
    st.subheader("Latest Valid Setups")

    if setups.empty:
        st.info("No recent valid setups for this chart market.")
    else:
        show_cols = [
            "Date",
            "Cameroon_Time",
            "Trading_Window",
            "Strategy",
            "Signal",
            "Direction",
            "Daily_Bias",
            "H1_Structure",
            "Premium_Discount",
            "Entry",
            "SL",
            "TP",
            "Reason",
        ]
        show_cols = [col for col in show_cols if col in setups.columns]
        st.dataframe(color_rows(setups[show_cols]), use_container_width=True)

st.warning("Signals are for educational and monitoring purposes only. Confirm broker price, spread, news risk, and your own trading plan before entering any trade.")
