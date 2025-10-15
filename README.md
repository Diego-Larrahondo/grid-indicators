
# Grid-Indicators

Hourly electricity impact intensities (EF 3.1 midpoints) for production and consumption, combining ENTSO‑E data with Brightway2/Ecoinvent lookups. Includes a script to export all consumption intensities + day‑ahead price for a full year for a given country.

## Features
- Pulls hourly generation and cross‑border flows from ENTSO‑E.
- Computes produced and consumed impact intensities (g per MWh) for EF v3.1 categories.
- Uses Brightway2 to map technologies and neighbor "market" processes (imports).
- Caches LCI results to `lci_cache_<impact>.xlsx` for faster reruns.
- Exports a single Excel file with all consumption impact series and price.

## Quick start

1. **Install** (recommend a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (copy `.env.example` to `.env` and fill it):
   - `ENTSOE_API_KEY` — your ENTSO‑E API key.
   - `BW_PROJECT` — Brightway project (e.g., `eco-310`).
   - `BW_DATABASE` — Brightway database name (e.g., `PkBudg1150-2025`).

3. **Run** an example year export (Spain, 2024):
   ```bash
   python scripts/fetch_year.py --country ES --year 2024
   ```

   This will create: `ES_2024_all_consumptions_and_price.xlsx`

## Notes
- ENTSO‑E returns UTC time; the code keeps everything timezone‑naive in UTC (hours).
- If UK data are requested, the code handles `GB` and `GB_NIR`.
- Some Brightway lookups fall back to FR or global when country‑specific acts aren't found.
- Caching avoids repeated LCI computations. Delete `lci_cache_*.xlsx` to force recompute.

## Repo layout
```
src/grid_indicators/
  __init__.py
  grid_indicators.py
scripts/
  fetch_year.py
requirements.txt
.env.example
```

## License
MIT
