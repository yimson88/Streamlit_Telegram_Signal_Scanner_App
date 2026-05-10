# Telegram Trading Signal Scanner

This is a Streamlit Cloud friendly app.

## What it does

- Scans only:
  - EURUSD
  - XAUUSD
  - BTCUSD
- Supports:
  - SMC Market Structure
  - FX 15m Momentum
- Sends BUY/SELL alerts to Telegram
- Uses duplicate protection so the same signal is not sent repeatedly
- Includes auto-refresh
- No MetaTrader 5
- No MT5 trade execution
- No manual risk calculator
- No trade ticket
- No journal
- No performance tab

## Streamlit Cloud Secrets

In Streamlit Cloud, add these secrets:

TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

## Run locally

streamlit run app.py

## Deploy files

Upload these files to GitHub:

- app.py
- requirements.txt
- README.md
