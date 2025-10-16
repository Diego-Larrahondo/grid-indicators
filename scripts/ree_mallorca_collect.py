
"""
REE Mallorca hourly generator & exchange collector
--------------------------------------------------
This module provides two functions to collect hourly generation and exchange
data for Mallorca using a user-provided `ree` client.

It expects a `ree` package/module exposing:
    - class Mallorca(Session) with method: get(datetime) -> Response | list[Response]
    - class Response with attributes:
        demand, waste, carbon, gas, combined, hydraulic, vapor, diesel,
        solar, wind, other, nuclear, link (dict with 'pe_ma','ma_ib','ma_me')

If your `ree` client has different names, adapt the mapping below.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from logging import getLogger, Logger
from typing import Optional, List, Dict, Any

from requests import Session

# Optional dependency: arrow for convenience (not strictly required here)
try:
    from arrow import get as arrow_get   # noqa: F401
except Exception:
    arrow_get = None

# External REE client expected by the user
try:
    from ree import Mallorca, Response  # type: ignore
except Exception as e:
    Mallorca = None
    Response = None

# Optional validation & exception types (fallbacks provided if user libs are absent)
try:
    from lib.exceptions import ParserException  # type: ignore
    from lib.validation import validate  # type: ignore
except Exception:
    class ParserException(Exception):
        """Lightweight parser exception fallback."""
        pass
    def validate(data: dict, logger: Logger, floor: float, expected_range=None) -> dict:
        """Fallback validation: apply a non-negative floor to demand and return the row."""
        data["demand"] = max(data.get("demand", 0.0), floor)
        return data

# Minimum valid demand floor for Mallorca (set to 0 unless you have stricter constraints)
FLOOR = 0.0

# Exchange link fields present in REE responses
EXCHANGE_FIELDS = ["pe_ma", "ma_ib", "ma_me"]


def collect_generation_and_exchanges_for_date(
    date_str: str,
    session: Optional[Session] = None,
    logger: Logger = getLogger(__name__),
) -> pd.DataFrame:
    """
    Collect hourly generation & exchange data for Mallorca for a given ISO date (YYYY-MM-DD).
    Returns a DataFrame with: datetime, demand, tech columns, and exchange link columns.
    """
    if Mallorca is None:
        raise ImportError("Missing 'ree' client. Please ensure `from ree import Mallorca, Response` works.")

    ses = session or Session()
    try:
        base_date = datetime.fromisoformat(date_str)
    except Exception as e:
        raise ParserException(f"Invalid date format: {date_str}") from e

    records: List[Dict[str, Any]] = []
    for hour in range(24):
        ts = datetime(base_date.year, base_date.month, base_date.day, hour)
        try:
            result = Mallorca(ses).get(ts)
        except Exception as e:
            logger.warning(f"Fetch failed at {ts}: {e}")
            continue

        responses = result if isinstance(result, list) else [result]
        for resp in responses:
            row = {
                "datetime": ts,
                "demand": getattr(resp, "demand", None),
                "biomass": getattr(resp, "waste", None),
                "coal": getattr(resp, "carbon", None),
                "gas": getattr(resp, "gas", None),
                "combined": getattr(resp, "combined", None),
                "hydro": getattr(resp, "hydraulic", None),
                "vapor": getattr(resp, "vapor", None),
                "diesel": getattr(resp, "diesel", None),
                "solar": getattr(resp, "solar", None),
                "wind": getattr(resp, "wind", None),
                "unknown": getattr(resp, "other", None),
                "nuclear": getattr(resp, "nuclear", None),
            }
            link = getattr(resp, "link", {}) or {}
            for field in EXCHANGE_FIELDS:
                row[field] = link.get(field)
            valid_row = validate(row, logger, floor=FLOOR)
            if valid_row:
                records.append(valid_row)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning(f"No data for {date_str}")
    else:
        df.sort_values("datetime", inplace=True)
    return df


def collect_generation_and_exchanges_for_year(
    year: int,
    session: Optional[Session] = None,
    logger: Logger = getLogger(__name__),
) -> pd.DataFrame:
    """
    Collect hourly generation & exchange data for Mallorca for an entire year.
    Returns a DataFrame with datetime index covering the whole year.
    """
    records: List[pd.DataFrame] = []
    for month in range(1, 13):
        start_date = datetime(year, month, 1)
        next_month = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
        days = (next_month - start_date).days

        for day in range(1, days + 1):
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            df_day = collect_generation_and_exchanges_for_date(date_str, session, logger)
            if not df_day.empty:
                records.append(df_day)

    if not records:
        return pd.DataFrame(columns=["datetime"]).set_index("datetime")

    df = pd.concat(records, ignore_index=True)
    df.sort_values("datetime", inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).tz_convert(None)
    df = df.set_index("datetime")
    return df
