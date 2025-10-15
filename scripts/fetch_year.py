
import os
import argparse
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from datetime import timedelta
from dotenv import load_dotenv

from grid_indicators import GridIndicators


def main():
    parser = argparse.ArgumentParser(description="Export hourly consumption impacts + price for a full year.")
    parser.add_argument("--country", required=True, help="ENTSO-E country code (e.g., ES, FR, DE)")
    parser.add_argument("--year", type=int, required=True, help="Year (e.g., 2024)")
    parser.add_argument("--impacts", nargs="*", default=list(GridIndicators.IMPACT_METHODS.keys()),
                        help="Subset of impact keys (default: all)")
    args = parser.parse_args()

    load_dotenv()  # loads ENTSOE_API_KEY, BW_PROJECT, BW_DATABASE if present
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY. Set it in your environment or .env file.")

    country_code = args.country
    impact_keys = args.impacts

    all_months = []

    for m in range(1, 13):
        t_start = pd.Timestamp(year=args.year, month=m, day=1, tz='UTC')
        t_end = t_start + MonthBegin(1)
        idx = pd.date_range(start=t_start, end=t_end - timedelta(hours=1), freq='h', tz='UTC').tz_convert(None)
        month_df = pd.DataFrame(index=idx)

        for i, key in enumerate(impact_keys):
            grid = GridIndicators(api_key, impact_key=key)
            _, cons_indic, prices = grid.get_indicators(country_code, t_start, t_end)

            cons = cons_indic['IMPACT'].reindex(idx).fillna(method='ffill')
            month_df[f'cons_{key}'] = cons

            if i == 0:
                price_series = prices.copy()
                price_series.index = pd.to_datetime(price_series.index, utc=True).tz_convert(None)
                price_series = price_series.reindex(idx).fillna(method='ffill')
                month_df['price'] = price_series

        all_months.append(month_df)

    df_year = pd.concat(all_months).sort_index()
    output_path = f"{country_code}_{args.year}_all_consumptions_and_price.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        df_year.to_excel(writer, sheet_name='consumption_and_price')

    print(f"\n✅ Exported year-long table with all cons_ columns + price to {output_path}")


if __name__ == "__main__":
    main()
