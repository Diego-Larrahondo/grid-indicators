
"""
Example script: single-impact, short time range (ENTSO-E + Brightway2)
---------------------------------------------------------------------
This script runs GridIndicators for one country, one impact method,
and a short time window (e.g., one day). It exports an Excel file
with hourly consumption impact and day-ahead price.

Usage:
    python scripts/run_example_minimal.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from grid_indicators import GridIndicators

# 1) Load environment variables from .env file
#    Make sure you have created .env with your ENTSO-E API key
load_dotenv()
api_key = os.getenv("ENTSOE_API_KEY")

if not api_key:
    raise RuntimeError("Missing ENTSOE_API_KEY in .env or environment variables")

# 2) Define impact method, country and time range
impact_key = "GWP100"          # Choose one EF v3.1 midpoint, e.g. GWP100
country_code = "ES"            # Spain
t_start = pd.Timestamp(year=2025, month=7, day=31, tz="UTC")
t_end = pd.Timestamp(year=2025, month=8, day=1, tz="UTC")

# 3) Create GridIndicators object and compute results
grid = GridIndicators(api_key, impact_key=impact_key)
prod_indic, cons_indic, prices = grid.get_indicators(country_code, t_start, t_end)

# 4) Prepare a single DataFrame with consumption and price
out = pd.DataFrame({
    f"cons_{impact_key}": cons_indic["IMPACT"].astype(float)
}).join(prices.rename("price"))

out.index.name = "timestamp_utc"

# 5) Export to Excel
output_file = f"{country_code}_{t_start.date()}_{impact_key}.xlsx"
out.to_excel(output_file)

print(f"✅ Done! Exported results to {output_file}")
