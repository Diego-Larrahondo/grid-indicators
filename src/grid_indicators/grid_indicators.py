
import os
import time
import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import entsoe
import brightway2 as bw


class GridIndicators:
    """
    Compute hourly impact intensities for electricity production and consumption
    using ENTSO-E data + Brightway2/Ecoinvent (EF v3.1 midpoints).
    """

    IMPACT_METHODS = {
        "AE_acidification": ('EF v3.1', 'acidification', 'accumulated exceedance (AE)'),
        "GWP100": ('EF v3.1', 'climate change', 'global warming potential (GWP100)'),
        "CTUe_ecotoxicity": ('EF v3.1', 'ecotoxicity: freshwater', 'comparative toxic unit for ecosystems (CTUe)'),
        "ADP_fossil": ('EF v3.1', 'energy resources: non-renewable', 'abiotic depletion potential (ADP): fossil fuels'),
        "P_eutrophication_fw": ('EF v3.1', 'eutrophication: freshwater', 'fraction of nutrients reaching freshwater end compartment (P)'),
        "N_eutrophication_marine": ('EF v3.1', 'eutrophication: marine', 'fraction of nutrients reaching marine end compartment (N)'),
        "AE_terrestrial": ('EF v3.1', 'eutrophication: terrestrial', 'accumulated exceedance (AE)'),
        "CTUh_carcinogenic": ('EF v3.1', 'human toxicity: carcinogenic', 'comparative toxic unit for human (CTUh)'),
        "CTUh_noncarcinogenic": ('EF v3.1', 'human toxicity: non-carcinogenic', 'comparative toxic unit for human (CTUh)'),
        "IR_human": ('EF v3.1', 'ionising radiation: human health', 'human exposure efficiency relative to u235'),
        "land_use": ('EF v3.1', 'land use', 'soil quality index'),
        "ADP_elements": ('EF v3.1', 'material resources: metals/minerals', 'abiotic depletion potential (ADP): elements (ultimate reserves)'),
        "ODP": ('EF v3.1', 'ozone depletion', 'ozone depletion potential (ODP)'),
        "PM": ('EF v3.1', 'particulate matter formation', 'impact on human health'),
        "POF": ('EF v3.1', 'photochemical oxidant formation: human health', 'tropospheric ozone concentration increase'),
        "water_use": ('EF v3.1', 'water use', 'user deprivation potential (deprivation-weighted water consumption)'),
    }

    CTRY_NAME = {
        'AT': 'Austria', 'BE': 'Belgium', 'BA': 'Bosnia and Herzegovia', 'BG': 'Bulgaria',
        'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France',
        'DE': 'Germany', 'GR': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
        'LV': 'Latvia', 'LT': 'Lithuania', 'ME': 'Montenegro', 'NL': 'Netherlands',
        'MK': 'North Macedonia', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
        'RO': 'Romania', 'RS': 'Serbia', 'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain',
        'SE': 'Sweden', 'CH': 'Switzerland', 'GB': 'Great Britain'
    }

    CTRY_OUT_NAME = {
        'AL': 'Albania', 'BY': 'Belarus', 'HR': 'Croatia', 'LU': 'Luxembourg', 'MT': 'Malta',
        'MD': 'Moldova', 'RU': 'Russia', 'RU-KGD': 'Russia (Kaliningrad)', 'TR': 'Turkey',
        'UA': 'Ukraine', 'UK': 'United Kingdom', 'GB_NIR': 'Northern Ireland'
    }

    CTRY_NBR = {
        'AT': ['CZ', 'DE', 'HU', 'IT', 'SI', 'CH'],
        'BE': ['FR', 'LU', 'NL', 'GB'],
        'BA': ['HR', 'ME', 'RS'],
        'BG': ['GR', 'MK', 'RO', 'RS', 'TR'],
        'CZ': ['AT', 'DE', 'PL', 'SK'],
        'DK': ['DE', 'NO', 'SE'],
        'EE': ['FI', 'LV', 'RU'],
        'FI': ['EE', 'NO', 'RU', 'SE'],
        'FR': ['BE', 'DE', 'IT', 'ES', 'CH', 'GB'],
        'DE': ['AT', 'CZ', 'DK', 'FR', 'LU', 'NL', 'PL', 'SE', 'CH'],
        'GR': ['AL', 'BG', 'IT', 'MK', 'TR'],
        'HU': ['AT', 'HR', 'RO', 'RS', 'SK', 'UA'],
        'IE': ['GB'],
        'IT': ['AT', 'FR', 'GR', 'MT', 'SI', 'CH'],
        'LV': ['EE', 'LT', 'RU'],
        'LT': ['BY', 'LV', 'PL', 'RU-KGD', 'SE'],
        'ME': ['AL', 'BA', 'RS'],
        'NL': ['BE', 'DE', 'NO', 'GB'],
        'MK': ['BG', 'GR', 'RS'],
        'NO': ['DK', 'FI', 'NL', 'SE'],
        'PL': ['CZ', 'DE', 'LT', 'SK', 'SE', 'UA'],
        'PT': ['ES'],
        'RO': ['BG', 'HU', 'RS', 'UA'],
        'RS': ['AL', 'BA', 'BG', 'HR', 'HU', 'MK', 'ME', 'RO'],
        'SK': ['CZ', 'HU', 'PL', 'UA'],
        'SI': ['AT', 'HR', 'IT'],
        'ES': ['FR', 'PT'],
        'SE': ['DK', 'FI', 'DE', 'LT', 'NO', 'PL'],
        'CH': ['AT', 'FR', 'DE', 'IT'],
        'GB': ['BE', 'IE', 'NL', 'FR']
    }

    PSR_TYPES = {
        'Biomass': 'B01',
        'Fossil Brown coal/Lignite': 'B02',
        'Fossil Coal-derived gas': 'B03',
        'Fossil Gas': 'B04',
        'Fossil Hard coal': 'B05',
        'Fossil Oil': 'B06',
        'Fossil Oil shale': 'B07',
        'Fossil Peat': 'B08',
        'Geothermal': 'B09',
        'Hydro Pumped Storage': 'B10',
        'Hydro Run-of-river and poundage': 'B11',
        'Hydro Water Reservoir': 'B12',
        'Marine': 'B13',
        'Nuclear': 'B14',
        'Other renewable': 'B15',
        'Solar': 'B16',
        'Waste': 'B17',
        'Wind Offshore': 'B18',
        'Wind Onshore': 'B19',
        'Other': 'B20'
    }

    def __init__(self, api_key: str, impact_key: str,
                 bw_project: Optional[str] = None,
                 bw_database: Optional[str] = None):
        # ENTSO-E API
        self.api_key = api_key
        self.client = entsoe.EntsoePandasClient(api_key)

        # Impact method
        if impact_key not in self.IMPACT_METHODS:
            raise ValueError(f"Unsupported impact key: {impact_key}")
        self.method = self.IMPACT_METHODS[impact_key]

        # Brightway config from env or args
        bw_project = bw_project or os.getenv("BW_PROJECT", "eco-310")
        bw_database = bw_database or os.getenv("BW_DATABASE", "PkBudg1150-2025")

        bw.projects.set_current(bw_project)
        if bw_database not in bw.databases:
            raise RuntimeError(f"Brightway database '{bw_database}' not found. "
                               f"Available: {list(bw.databases)}")
        self.db = bw.Database(bw_database)

        # Mapping ENTSO-E tech -> Brightway processes
        self.brightway_map = {
            'Biomass': ['heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014'],
            'Fossil Brown coal/Lignite': ['electricity production, hard coal'],
            'Fossil Coal-derived gas': ['treatment of blast furnace gas, in power plant'],
            'Fossil Gas': ['electricity production, natural gas, combined cycle power plant',
                           'electricity production, natural gas, conventional power plant'],
            'Fossil Hard coal': ['electricity production, hard coal'],
            'Fossil Oil': ['electricity production, oil'],
            'Fossil Oil shale': ['treatment of coal gas, in power plant'],
            'Fossil Peat': [],
            'Geothermal': ['electricity production, deep geothermal'],
            'Hydro Pumped Storage': ['electricity production, hydro, pumped storage'],
            'Energy storage': ['electricity production, hydro, pumped storage'],
            'Hydro Run-of-river and poundage': ['electricity production, hydro, run-of-river'],
            'Hydro Water Reservoir': ['electricity production, hydro, reservoir, non-alpine region'],
            'Marine': [],
            'Nuclear': ['electricity production, nuclear, pressure water reactor',
                        'electricity production, nuclear, boiling water reactor'],
            'Other renewable': ['electricity production, wind, <1MW turbine, onshore',
                                'electricity production, wind, 1-3MW turbine, onshore',
                                'electricity production, wind, >3MW turbine, onshore',
                                'electricity production, wind, 1-3MW turbine, offshore',
                                'electricity production, photovoltaic, 570kWp open ground installation, multi-Si'],
            'Solar': ['electricity production, photovoltaic, 570kWp open ground installation, multi-Si'],
            'Waste': ['heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014'],
            'Wind Offshore': ['electricity production, wind, 1-3MW turbine, offshore'],
            'Wind Onshore': ['electricity production, wind, <1MW turbine, onshore',
                             'electricity production, wind, 1-3MW turbine, onshore',
                             'electricity production, wind, >3MW turbine, onshore'],
            'Other': ['electricity, high voltage, residual mix']
        }

        # Out-of-area "market" process names (per country code)
        self.market_process_map = {code: f"electricity, high voltage, production mix, {code}"
                                   for code in self.CTRY_OUT_NAME}

        # Cache (Excel) for LCI intensities
        self.cache_path = f"lci_cache_{impact_key}.xlsx"
        self.cache_sheet = "lci"
        if os.path.exists(self.cache_path):
            self._cache_df = pd.read_excel(self.cache_path, sheet_name=self.cache_sheet, index_col=0)
        else:
            self._cache_df = pd.DataFrame(columns=list(self.brightway_map.keys()) + ["IMPORT_MARKET"])
        self._cache_df.index = self._cache_df.index.astype(str)

    # -------------------- Brightway helpers --------------------
    def _compute_market_impact_once(self, country_code: str) -> float:
        """LCIA score (per kWh → g per MWh) for out-of-area 'market' electricity."""
        market_proc = self.market_process_map.get(country_code, "")
        acts = self.db.search(market_proc, filter={'location': country_code}) or self.db.search(market_proc)
        if not acts:
            return 0.0
        lca = bw.LCA({acts[0]: 1}, self.method)
        lca.lci(); lca.lcia()
        return lca.score * 1000.0

    def _load_market_impact(self, country_code: str) -> float:
        """Get cached import market intensity; compute & cache if missing."""
        if country_code not in self._cache_df.index:
            self._load_ecoinvent_lci(country_code)
        return float(self._cache_df.loc[country_code, "IMPORT_MARKET"])

    def _load_ecoinvent_lci(self, country_code: str) -> Dict[str, float]:
        """Return dict tech->LCI intensity (g/MWh). Compute & cache if needed."""
        if country_code in self._cache_df.index:
            row = self._cache_df.loc[country_code]
            return row.drop("IMPORT_MARKET").to_dict()

        tech_lci = {}
        for tech, processes in self.brightway_map.items():
            scores = []
            for proc in processes:
                acts = self.db.search(proc, filter={'location': country_code}) or \
                       self.db.search(proc, filter={'location': 'FR'}) or \
                       self.db.search(proc)
                acts = [a for a in acts if a.get('unit') == 'kilowatt hour']
                if acts:
                    lca = bw.LCA({acts[0]: 1}, self.method)
                    lca.lci(); lca.lcia()
                    scores.append(lca.score)
            tech_lci[tech] = max(0.0, float(np.mean(scores) * 1000.0) if scores else 0.0)

        import_factor = self._compute_market_impact_once(country_code)
        row = {**tech_lci, "IMPORT_MARKET": import_factor}
        self._cache_df.loc[country_code] = row

        with pd.ExcelWriter(self.cache_path, engine="openpyxl", mode="w") as w:
            self._cache_df.to_excel(w, sheet_name=self.cache_sheet)

        return tech_lci

    # -------------------- ENTSO-E helpers --------------------
    def __process_missing_data(self, df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
        dt = pd.date_range(start=t_start, end=t_end - pd.Timedelta('1h'), freq='h', tz=t_end.tz)
        df = df.resample('h').mean().reindex(dt).interpolate().fillna(0)
        return df

    def __check_data_structure(self, country_code: str):
        if country_code == 'UK':
            country_code = 'GB'
        t_start = pd.Timestamp('201901010000', tz='UTC')
        t_end = pd.Timestamp('201901010100', tz='UTC')
        client = entsoe.EntsoePandasClient(self.api_key)
        df = client.query_generation(country_code, start=t_start, end=t_end)
        if df.columns.nlevels == 2:
            df = df.iloc[:, df.columns.get_level_values(1) == 'Actual Aggregated']
            df.columns = df.columns.droplevel(1)
        return (1/len(df.index), df.columns.tolist())

    def __get_prod(self, country_code: str, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
        data_dir = os.path.join(os.getcwd(), "Data")
        os.makedirs(data_dir, exist_ok=True)
        fname = f"GEN_{country_code}_{t_start:%Y%m%d%H%M}_{t_end:%Y%m%d%H%M}.csv"
        full_path = os.path.join(data_dir, fname)

        if country_code == "UK":
            df_gb = self.__get_prod("GB", t_start, t_end)
            df_nir = self.__get_prod("GB_NIR", t_start, t_end)
            df = df_gb.add(df_nir, fill_value=0)
        elif os.path.exists(full_path):
            df = pd.read_csv(full_path, sep=";", index_col=0, parse_dates=True)
        else:
            client = entsoe.EntsoePandasClient(self.api_key)
            timestep, techs = self.__check_data_structure(country_code)
            if (timestep < 0.5 or len(techs) >= 12) and (t_end - t_start) > datetime.timedelta(days=90):
                pieces = []
                for tech in techs:
                    if tech == "Hydro Pumped Storage":
                        continue
                    try:
                        df_t = client.query_generation(country_code, start=t_start, end=t_end,
                                                       psr_type=self.PSR_TYPES.get(tech))
                        if df_t.columns.nlevels == 2:
                            df_t = df_t.xs("Actual Aggregated", axis=1, level=1)
                        pieces.append(df_t)
                    except Exception:
                        pass
                df = pd.concat(pieces, axis=1)
            else:
                df = client.query_generation(country_code, start=t_start, end=t_end)
                if df.columns.nlevels == 2:
                    df = df.xs("Actual Aggregated", axis=1, level=1)

            df = df.drop(columns=["Hydro Pumped Storage"], errors="ignore")
            df.to_csv(full_path, sep=";")

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        df = df.resample("h").mean().interpolate().fillna(0)
        return df

    def __get_trade(self, country_code: str, nbr: str, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
        if country_code == "UK":
            country_code = "GB"
        if nbr == "UK":
            nbr = "GB"

        data_dir = os.path.join(os.getcwd(), "Data")
        os.makedirs(data_dir, exist_ok=True)
        fname = f"TRADE_{country_code}_{nbr}_{t_start:%Y%m%d%H%M}_{t_end:%Y%m%d%H%M}.csv"
        full_path = os.path.join(data_dir, fname)

        if os.path.exists(full_path):
            df = pd.read_csv(full_path, sep=";", index_col=0, parse_dates=True)
        else:
            client = entsoe.EntsoePandasClient(self.api_key)
            exp = client.query_crossborder_flows(country_code, nbr, start=t_start, end=t_end)
            imp = client.query_crossborder_flows(nbr, country_code, start=t_start, end=t_end)
            df = pd.concat([exp.to_frame(name="exports"), imp.to_frame(name="imports")], axis=1)
            df = self.__process_missing_data(df, t_start, t_end)
            df.to_csv(full_path, sep=";")

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        df = df.resample("h").mean().interpolate().fillna(0)
        return df

    # -------------------- Indicators --------------------
    def __calc_indic_prod(self, df: pd.DataFrame, tech_factors: Dict[str, float]) -> pd.DataFrame:
        indic = pd.DataFrame(index=df.index)
        weighted = pd.Series(0.0, index=df.index)
        for tech, series in df.items():
            factor = tech_factors.get(tech, 0.0)
            weighted += series * factor  # MWh * (g/MWh) = g
        total_gen = df.sum(axis=1)
        indic["IMPACT"] = weighted.where(total_gen > 0, 0.0) / total_gen
        return indic

    def __calc_indic_cons(self,
                          df_prod: Dict[str, pd.DataFrame],
                          df_prod_indic: Dict[str, pd.DataFrame],
                          df_trade: Dict[str, pd.DataFrame],
                          country_code: str,
                          tech_factors: Dict[str, Dict[str, float]],
                          out_factors: Dict[str, float]) -> pd.DataFrame:
        idx = df_trade["global"].index
        indic = pd.DataFrame(index=idx)

        prod_sum = df_prod[country_code].sum(axis=1)
        indic["IMPACT"] = df_prod_indic[country_code]["IMPACT"] * prod_sum

        for nbr in self.CTRY_NBR[country_code]:
            imp = df_trade[nbr]["imports"]
            if nbr in tech_factors:
                indic["IMPACT"] += df_prod_indic[nbr]["IMPACT"] * imp
            else:
                factor = out_factors.get(nbr, 0.0)
                indic["IMPACT"] += factor * imp

        total_cons = prod_sum + df_trade["global"]["total_imports"]
        indic["IMPACT"] = indic["IMPACT"] / total_cons.replace(0, np.nan)
        indic["IMPACT"] = indic["IMPACT"].fillna(0.0)
        return indic

    def get_indicators(self, country_code: str, t_start: pd.Timestamp, t_end: pd.Timestamp):
        print(f"\n--- GRID INDICATORS for {country_code} ---")
        if country_code not in self.CTRY_NAME:
            raise SystemExit(f"Country {country_code} not in CTRY_NAME")

        neighbours = self.CTRY_NBR[country_code]
        in_area = [n for n in neighbours if n in self.CTRY_NAME]
        out_of_area = [n for n in neighbours if n not in self.CTRY_NAME and n in self.CTRY_OUT_NAME]

        # Brightway LCI for generation (country + in-area neighbors)
        gen_countries = [country_code] + in_area
        gen_factors = {c: self._load_ecoinvent_lci(c) for c in gen_countries}

        # Market intensity for out-of-area neighbors
        imp_factors = {c: self._load_market_impact(c) for c in out_of_area}

        # Hourly generation
        data_prod = {c: self.__get_prod(c, t_start, t_end) for c in gen_countries}

        # Hourly trade flows
        data_trade = {nbr: self.__get_trade(country_code, nbr, t_start, t_end) for nbr in neighbours}

        # Global trade summary
        df_glob = pd.DataFrame(index=data_prod[country_code].index)
        df_glob["total_imports"] = sum(data_trade[n]["imports"] for n in neighbours)
        df_glob["total_exports"] = sum(data_trade[n]["exports"] for n in neighbours)
        df_glob["balance"] = df_glob["total_imports"] - df_glob["total_exports"]
        df_glob["net_importer"] = df_glob["balance"] > 0
        data_trade["global"] = df_glob

        # Produced intensity (main country + neighbors for consumption calc)
        all_prod_indic = {c: self.__calc_indic_prod(data_prod[c], gen_factors[c]) for c in gen_countries}

        # Consumed intensity
        cons_indic = self.__calc_indic_cons(
            df_prod=data_prod,
            df_prod_indic=all_prod_indic,
            df_trade=data_trade,
            country_code=country_code,
            tech_factors=gen_factors,
            out_factors=imp_factors,
        )

        # Day-ahead prices
        prices = self.client.query_day_ahead_prices(country_code, start=t_start, end=t_end)
        prices = prices.tz_convert(None)

        # Produced intensity of the focus country (for completeness)
        prod_indic = all_prod_indic[country_code]
        return prod_indic, cons_indic, prices
