import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_FILE = "feature_engineered_dataset.csv"
BASELINE_FORECASTS = "forecasts_baseline.csv"
OUT_CSV = "scenario_predictions_2030.csv"

def baseline_gdp_2030(bf_df, country):
    sub = bf_df[(bf_df["country"] == country) & (bf_df["year"] <= 2030)]
    if "gdp_current_usd" not in sub.columns or sub.empty:
        return np.nan
    coeffs = np.polyfit(sub["year"], sub["gdp_current_usd"], 1)
    return float(np.polyval(coeffs, 2030.0))

def baseline_poverty_2030(fe_df, country):
    col = "poverty_headcount_ratio_3ppp_pct"
    if col not in fe_df.columns:
        return np.nan
    sub = fe_df[fe_df["country"] == country].dropna(subset=[col, "year"])
    if len(sub) >= 4:
        coeffs = np.polyfit(sub["year"].astype(float), sub[col].astype(float), 1)
        pred = np.polyval(coeffs, 2030.0)
        return float(np.clip(pred, 0.0, 100.0))
    elif len(sub) > 0:
        return float(sub.sort_values("year").iloc[-1][col])
    return np.nan

def baseline_unemployment_2030(fe_df, country):
    possible_cols = [
        "unemployment_total_pct_labor_force_social",
        "unemployment_total_pct_labor_force",
        "unemployment_rate",
        "unemployment_pct",
        "unemployment"
    ]
    col = next((c for c in possible_cols if c in fe_df.columns), None)
    if col is None:
        return np.nan
    sub = fe_df[fe_df["country"] == country].dropna(subset=[col, "year"])
    if len(sub) >= 4:
        coeffs = np.polyfit(sub["year"].astype(float), sub[col].astype(float), 1)
        pred = np.polyval(coeffs, 2030.0)
        return float(np.clip(pred, 0.0, 100.0))
    elif len(sub) > 0:
        return float(sub.sort_values("year").iloc[-1][col])
    return np.nan

def apply_scenarios(gdp_base, pov_base, unemp_base):
    disaster_gdp = gdp_base * 0.955
    disaster_pov = pov_base * 1.02 if pov_base is not None else np.nan
    disaster_unemp = unemp_base * 1.05 if unemp_base is not None else np.nan
    tradewar_gdp = gdp_base * 0.92
    tradewar_pov = pov_base * 1.04 if pov_base is not None else np.nan
    tradewar_unemp = unemp_base * 1.07 if unemp_base is not None else np.nan
    return {
        "baseline": (gdp_base, pov_base, unemp_base),
        "disaster_2026_sev>8": (disaster_gdp, disaster_pov, disaster_unemp),
        "trade_war_2027_-20pct": (tradewar_gdp, tradewar_pov, tradewar_unemp),
    }

def main():
    print("Running Question 10 scenario predictions...\n")
    if not Path(FEATURES_FILE).exists() or not Path(BASELINE_FORECASTS).exists():
        print("Missing input files.")
        return
    fe = pd.read_csv(FEATURES_FILE)
    bf = pd.read_csv(BASELINE_FORECASTS)
    print("📌 Columns in baseline_forecasts.csv:")
    print(bf.columns.tolist())

    print("\n📌 First 5 rows:")
    print(bf.head())
    countries = sorted(set(fe["country"]).intersection(set(bf["country"])))
    gdp_baseline = {c: baseline_gdp_2030(bf, c) for c in countries}
    pov_baseline = {c: baseline_poverty_2030(fe, c) for c in countries}
    unemp_baseline = {c: baseline_unemployment_2030(fe, c) for c in countries}
    rows = []
    for c in countries:
        gdp = gdp_baseline.get(c, np.nan)
        pov = pov_baseline.get(c, np.nan)
        unemp = unemp_baseline.get(c, np.nan)
        scen = apply_scenarios(gdp, pov, unemp)
        for s, vals in scen.items():
            rows.append({
                "country": c,
                "scenario": s,
                "gdp_2030_usd": vals[0],
                "poverty_2030_pct": vals[1],
                "unemployment_2030_pct": vals[2]
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Scenario predictions saved to {OUT_CSV}\n")
    print("Preview of results:")
    print(out.head(5))

if __name__ == "__main__":
    main()
