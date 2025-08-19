import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_FILE = "feature_engineered_dataset.csv"
BASELINE_FORECASTS = "forecasts_baseline.csv"
OUT_CSV = "scenario11_predictions_2030.csv"

def safe_minmax(s):
    v = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if v.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    mn, mx = v.min(), v.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (v - mn) / (mx - mn)

def baseline_gdp_2030(bf_df, country):
    sub = bf_df[(bf_df["country"] == country) & (bf_df["year"] <= 2030)]
    if sub.empty or "gdp_current_usd" not in sub.columns:
        return np.nan
    x = sub["year"].astype(float).values
    y = sub["gdp_current_usd"].astype(float).values
    if len(sub) >= 3:
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 2030.0))
    return float(sub.sort_values("year").iloc[-1]["gdp_current_usd"])

def baseline_poverty_2030(fe_df, country):
    col = "poverty_headcount_ratio_3ppp_pct"
    if col not in fe_df.columns:
        return np.nan
    sub = fe_df[fe_df["country"] == country].dropna(subset=[col, "year"])
    if sub.empty:
        return np.nan
    x = sub["year"].astype(float).values
    y = sub[col].astype(float).values
    if len(sub) >= 4:
        coeffs = np.polyfit(x, y, 1)
        pred = np.polyval(coeffs, 2030.0)
        return float(np.clip(pred, 0.0, 100.0))
    return float(sub.sort_values("year").iloc[-1][col])

def main():
    print(" Running Question 11 scenario predictions...\n")
    if not Path(FEATURES_FILE).exists() or not Path(BASELINE_FORECASTS).exists():
        print("Missing input files.")
        return

    fe = pd.read_csv(FEATURES_FILE)
    bf = pd.read_csv(BASELINE_FORECASTS)

    needed_cols = {
        "trade_dependency_index": "trade_dependency_index",
        "resilience_score": "resilience_score",
        "num_disasters": "num_disasters",
        "avg_magnitude": "avg_magnitude",
        "total_damage_adjusted_usd": "total_damage_adjusted_usd",
        "shock_impact_score": "shock_impact_score",
        "country": "country"
    }
    for c in needed_cols.values():
        if c not in fe.columns:
            fe[c] = np.nan

    latest_fe = fe.sort_values("year").groupby("country").tail(1).reset_index(drop=True)

    tdep_norm = safe_minmax(latest_fe["trade_dependency_index"])
    resil_norm = safe_minmax(latest_fe["resilience_score"])
    dis_num_norm = safe_minmax(latest_fe["num_disasters"])
    dis_mag_norm = safe_minmax(latest_fe["avg_magnitude"])
    dis_dmg_norm = safe_minmax(latest_fe["total_damage_adjusted_usd"])
    shock_norm = safe_minmax(latest_fe["shock_impact_score"])

    risk_components = []
    for s in [dis_num_norm, dis_mag_norm, dis_dmg_norm, shock_norm]:
        risk_components.append(s.fillna(s.median() if not np.isnan(s.median()) else 0.5))
    risk_norm = pd.concat(risk_components, axis=1).mean(axis=1)

    latest_fe = latest_fe.assign(
        trade_dep_norm=tdep_norm.fillna(0.5),
        resil_norm=resil_norm.fillna(0.5),
        risk_norm=risk_norm.fillna(0.5)
    )

    countries = sorted(set(latest_fe["country"]).intersection(set(bf["country"])))
    gdp_base = {c: baseline_gdp_2030(bf, c) for c in countries}
    pov_base = {c: baseline_poverty_2030(fe, c) for c in countries}

    rows = []
    alpha_poverty = 0.6

    for _, row in latest_fe[latest_fe["country"].isin(countries)].iterrows():
        c = row["country"]
        gb = gdp_base.get(c, np.nan)
        pb = pov_base.get(c, np.nan)

        td = float(row["trade_dep_norm"])
        rs = float(row["resil_norm"])
        rk = float(row["risk_norm"])

        best_gdp_factor = 1.0 + 0.02 + 0.04 * rs + 0.03 * (1.0 - td)
        worst_gdp_factor = 1.0 - (0.03 + 0.04 * rk + 0.05 * td + 0.02 * (rk * td))
        worst_gdp_factor = float(np.clip(worst_gdp_factor, 0.75, 1.0))
        best_gdp = gb * best_gdp_factor if not np.isnan(gb) else np.nan
        worst_gdp = gb * worst_gdp_factor if not np.isnan(gb) else np.nan

        if not np.isnan(gb) and not np.isnan(best_gdp) and not np.isnan(pb):
            best_pov = float(pb * ((gb / best_gdp) ** alpha_poverty))
        else:
            best_pov = float(pb * 0.95) if not np.isnan(pb) else np.nan

        if not np.isnan(gb) and not np.isnan(worst_gdp) and not np.isnan(pb):
            worst_pov = float(pb * ((gb / worst_gdp) ** alpha_poverty))
        else:
            worst_pov = float(pb * 1.08) if not np.isnan(pb) else np.nan

        rows.append({"country": c, "scenario": "baseline", "gdp_2030_usd": gb, "poverty_2030_pct": pb})
        rows.append({"country": c, "scenario": "best_case_diversification_resilience", "gdp_2030_usd": best_gdp, "poverty_2030_pct": best_pov})
        rows.append({"country": c, "scenario": "worst_case_recurring_disasters_concentration", "gdp_2030_usd": worst_gdp, "poverty_2030_pct": worst_pov})

    out = pd.DataFrame(rows)[["country", "scenario", "gdp_2030_usd", "poverty_2030_pct"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"Scenario 11 predictions saved to {OUT_CSV}\n")
    print("Preview:")
    print(out.head(5))

if __name__ == "__main__":
    main()
