
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from grid_indicators import GridIndicators

def _to_utc_ts(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts

def main():
    parser = argparse.ArgumentParser(
        description="Export hourly consumption impact(s) + day-ahead price for an arbitrary UTC time range."
    )
    parser.add_argument("--country", required=True, help="ENTSO-E code, e.g. ES, FR, DE")
    parser.add_argument("--start", required=True, help="UTC start (e.g., 2025-07-31T00:00Z or 2025-07-31)")
    parser.add_argument("--end", required=True, help="UTC end (exclusive), e.g., 2025-08-01")
    parser.add_argument("--impacts", nargs="*", default=["GWP100"], help="Impact keys (default: GWP100)")
    parser.add_argument("--outfile", default=None, help="Excel output path (auto if omitted)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY. Set it in .env or env vars.")

    t_start = _to_utc_ts(args.start)
    t_end   = _to_utc_ts(args.end)

    idx = pd.date_range(start=t_start, end=t_end - pd.Timedelta(hours=1), freq="H", tz="UTC").tz_convert(None)
    out_df = pd.DataFrame(index=idx)

    price_added = False
    for key in args.impacts:
        grid = GridIndicators(api_key, impact_key=key)
        _, cons_indic, prices = grid.get_indicators(args.country, t_start, t_end)

        cons = cons_indic["IMPACT"].reindex(idx).ffill()
        out_df[f"cons_{key}"] = cons

        if not price_added:
            p = prices.copy()
            p.index = pd.to_datetime(p.index, utc=True).tz_convert(None)
            out_df["price"] = p.reindex(idx).ffill()
            price_added = True

    outfile = args.outfile or f"{args.country}_{t_start.date()}_{t_end.date()}_cons_and_price.xlsx"
    out_df.to_excel(outfile, sheet_name="consumption_and_price")
    print("✅ Export:", outfile)

if __name__ == "__main__":
    main()
