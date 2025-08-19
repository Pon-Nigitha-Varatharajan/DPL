import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_FILE = "feature_engineered_dataset.csv"
OUT_CSV = "scenario12_top_resilience_2030.csv"

def predict_resilience_2030(fe_df, country):
    col = "resilience_score"
    sub = fe_df[fe_df["country"] == country].dropna(subset=[col, "year"])
    if sub.empty:
        return np.nan
    x = sub["year"].astype(float).values
    y = sub[col].astype(float).values
    if len(sub) >= 4:
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 2030.0))
    return float(sub.sort_values("year").iloc[-1][col])

def main():
    print("🚀 Running Question 12 resilience ranking...\n")
    if not Path(FEATURES_FILE).exists():
        print("❌ Missing feature_engineered_dataset.csv")
        return

    fe = pd.read_csv(FEATURES_FILE)

    countries = fe["country"].unique()
    resilience_pred = {c: predict_resilience_2030(fe, c) for c in countries}

    df = pd.DataFrame([
        {"country": c, "resilience_2030": resilience_pred[c]}
        for c in countries
    ]).dropna().sort_values("resilience_2030", ascending=False).reset_index(drop=True)

    top5 = df.head(5)

    factor_cols = [
        "trade_dependency_index", "gdp_growth_pct", "poverty_headcount_ratio_3ppp_pct",
        "shock_impact_score", "external_debt_stocks_gni_pct"
    ]
    factor_importance = []
    for _, row in top5.iterrows():
        c = row["country"]
        sub = fe[fe["country"] == c].dropna(subset=factor_cols)
        if sub.empty:
            continue
        latest = sub.sort_values("year").iloc[-1][factor_cols]
        factor_importance.append({"country": c, **latest.to_dict()})

    factors_df = pd.DataFrame(factor_importance)

    merged = top5.merge(factors_df, on="country", how="left")
    merged.to_csv(OUT_CSV, index=False)

    print(f"✅ Top 5 resilience predictions saved to {OUT_CSV}\n")
    print("📌 Preview:")
    print(merged)

if __name__ == "__main__":
    main()
