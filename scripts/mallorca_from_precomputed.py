
"""
Mallorca impact builder from precomputed Spain (Peninsula) consumption impacts
-----------------------------------------------------------------------------
Pipeline:
1) Use GridIndicators to compute Spain (ES) hourly consumption impact(s) + price
   and export them (already done by scripts/fetch_year.py or fetch_range.py).
2) Use REE Mallorca hourly generation/exchanges to compute Mallorca's hourly
   impacts by combining local tech factors (from ES LCI cache) + imports from
   the Peninsula (pe_ma) weighted by ES consumption intensity.

This script reads:
- <ES_excel>: an Excel with columns price and cons_<impact_key>
- lci_cache_<impact_key>.xlsx: Brightway/Ecoinvent tech intensities for ES

It writes:
- mallorca_<year>_from_precomputed.xlsx with one sheet per impact key.
"""

import os
import pandas as pd
from logging import Logger, getLogger
from typing import Optional, Dict, List

from requests import Session

# Local collector helpers
from scripts.ree_mallorca_collect import collect_generation_and_exchanges_for_year


def compute_mallorca_from_precomputed(
    year: int,
    impact_keys: List[str],
    es_excel: str,
    lci_cache_template: str,
    session: Optional[Session] = None,
    logger: Logger = getLogger(__name__),
) -> str:
    """
    For each impact_key:
      - Read 'cons_<impact_key>' and 'price' from es_excel
      - Read Brightway LCI row 'ES' from lci_cache_<impact_key>.xlsx
      - Collect Mallorca year hourly data
      - Compute local impact, import impact, totals and intensity
      - Write each impact to a sheet in a single Excel output
    Returns the output file path.
    """
    ses = session or Session()

    # 1) Fetch Mallorca base data once for the whole year
    df_base = collect_generation_and_exchanges_for_year(year, ses, logger)
    if df_base.empty:
        raise RuntimeError("Mallorca yearly DataFrame is empty. Check your REE client/connection.")

    # 2) Load ES consumption+price once
    df_es = pd.read_excel(es_excel, index_col=0, parse_dates=True)
    # Normalize index to timezone-naive UTC for alignment with df_base
    df_es.index = pd.to_datetime(df_es.index, utc=True).tz_convert(None)

    # 3) Prepare output writer
    output_file = f"mallorca_{year}_from_precomputed.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        for key in impact_keys:
            logger.info(f"[Mallorca] Processing impact: {key}")
            # ES series
            cons_series = df_es[f"cons_{key}"]
            price_series = df_es["price"]

            # LCI factors for ES
            lci_cache = pd.read_excel(lci_cache_template.format(impact_key=key), sheet_name="lci", index_col=0)
            row_es = lci_cache.loc["ES"]
            factors_es = row_es.drop("IMPORT_MARKET").to_dict()
            import_market_es = float(row_es["IMPORT_MARKET"])  # kept in case fallback is needed

            # Work on a copy of the Mallorca base frame
            df_m = df_base.copy()

            # Map REE Mallorca columns to ENTSO-E technology keys used in the cache
            tech_map = {
                "biomass":   "Biomass",
                "coal":      "Fossil Hard coal",
                "gas":       "Fossil Gas",
                "combined":  "Fossil Gas",
                "hydro":     "Hydro Run-of-river and poundage",
                "vapor":     "Hydro Water Reservoir",
                "diesel":    "Fossil Oil",
                "solar":     "Solar",
                "wind":      "Wind Onshore",
                "unknown":   "Other",
                "nuclear":   "Nuclear",
            }

            # Compute tech-by-tech local impact (g)
            for col, tech in tech_map.items():
                df_m[f"impact_{tech}"] = df_m.get(col, 0.0) * float(factors_es.get(tech, 0.0))

            # Sum of local impacts
            tech_cols = [f"impact_{t}" for t in tech_map.values()]
            df_m["impact_local_g"] = df_m[tech_cols].sum(axis=1)

            # Import impact: Mallorca's pe_ma (MW/MWh) × ES hourly intensity (g/MWh)
            pe_ma = df_m.get("pe_ma", 0.0)
            cons_intensity = cons_series.reindex(df_m.index).fillna(method="ffill").fillna(0.0)
            df_m["impact_import_g"] = pe_ma * cons_intensity

            # Totals & intensity
            df_m["impact_total_g"] = df_m["impact_local_g"] + df_m["impact_import_g"]
            demand = df_m["demand"].replace(0, pd.NA)
            df_m["intensity_g_per_MWh"] = df_m["impact_total_g"] / demand

            # Attach price column (same for all impacts)
            df_m["price"] = price_series.reindex(df_m.index).fillna(method="ffill")

            # Columns to export
            export_cols = [
                "demand", "pe_ma", *tech_map.keys(),
                *tech_cols,
                "impact_local_g", "impact_import_g",
                "impact_total_g", "intensity_g_per_MWh",
                "price",
            ]
            sheet = key[:31]  # Excel sheet name limit
            df_m[export_cols].to_excel(writer, sheet_name=sheet)

    return output_file


if __name__ == "__main__":
    # Example run: build Mallorca 2024 using precomputed ES 2024
    YEAR = 2024
    ES_EXCEL = "ES_2024_all_consumptions_and_price.xlsx"  # created by scripts/fetch_year.py
    LCI_TEMPLATE = "lci_cache_{impact_key}.xlsx"

    # Minimal single-impact example (you can list more):
    IMPACT_KEYS = ["GWP100"]

    out = compute_mallorca_from_precomputed(
        year=YEAR,
        impact_keys=IMPACT_KEYS,
        es_excel=ES_EXCEL,
        lci_cache_template=LCI_TEMPLATE,
    )
    print(f"✅ Mallorca impacts saved to {out}")
