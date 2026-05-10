# Telegram Signal Scanner - NameError Fixed

## Fixed

This version fixes:

NameError: name 'scan_speed' is not defined

## What changed

- Added a safe default: scan_speed = "Fast"
- Added/confirmed the Scan Speed selector in the sidebar
- Passed scan_speed properly through the scanner functions
- Kept the Streamlit Cloud fast data windows

## Recommended settings

- Scan speed: Fast
- Refresh every: 5 minutes
- Strategy: FX 15m Momentum first
