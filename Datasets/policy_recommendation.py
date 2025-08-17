import pandas as pd
import numpy as np

# Inputs
BASELINE_PATH = "forecasts_baseline.csv"
FEATURES_PATH = "feature_engineered_dataset.csv"

# Outputs
OUTPUT_PATH = "policy_recommendations_single.csv"

# Load data
baseline = pd.read_csv(BASELINE_PATH)
fe = pd.read_csv(FEATURES_PATH)

# Candidate columns we want to enrich
cand_cols = {
    "unemployment_total_pct_labor_force": "unemployment_rate",
    "external_debt_stocks_gni_pct": "debt_gni_pct",
    "trade_balance_gdp_pct": "trade_balance_gdp_pct",
    "Gini index": "gini_index",
    "gini_index": "gini_index",
    "shock_impact_score": "shock_impact_score",
}
keep_cols = ["country", "year"] + [c for c in cand_cols.keys() if c in fe.columns]
fe_small = fe[keep_cols].copy()
rename_map = {c: cand_cols[c] for c in cand_cols if c in fe_small.columns}
fe_small = fe_small.rename(columns=rename_map)

# Most recent structural indicators
latest_struct = (
    fe_small.sort_values(["country", "year"])
    .groupby("country")
    .tail(1)
    .drop(columns=["year"], errors="ignore")
    .set_index("country")
)

# Extract baseline forecasts for 2030
base2030 = baseline[baseline["year"] == 2030].copy()
base2030 = base2030.set_index("country")

# Merge structural info
base2030 = base2030.join(latest_struct, how="left")

# Add missing cols if needed
for c in ["unemployment_rate", "debt_gni_pct", "trade_balance_gdp_pct", "gini_index", "shock_impact_score"]:
    if c not in base2030.columns:
        base2030[c] = np.nan

# Columns of interest
pub_cols = [
    "year",
    "gdp_growth_pct",
    "poverty_rate",
    "resilience_score",
    "unemployment_rate",
    "debt_gni_pct",
    "trade_balance_gdp_pct",
    "gini_index",
    "shock_impact_score",
]
for c in pub_cols:
    if c not in base2030.columns:
        base2030[c] = np.nan
base2030 = base2030[pub_cols].copy()

# Scenario transformer
def apply_scenario(df, scenario):
    out = df.copy()
    if scenario == "social_spending":
        out["poverty_rate"] *= 0.9
        out["gdp_growth_pct"] += 0.5
        out["unemployment_rate"] *= 0.95
        out["gini_index"] *= 0.97
        out["debt_gni_pct"] += 1.0
    elif scenario == "trade_diversification":
        out["gdp_growth_pct"] += 0.3
        out["resilience_score"] *= 1.15
        out["trade_balance_gdp_pct"] += 0.5
        out["unemployment_rate"] *= 0.98
    elif scenario == "global_crisis":
        out["gdp_growth_pct"] -= 2.5
        out["poverty_rate"] *= 1.1
        out["resilience_score"] *= 0.85
        out["unemployment_rate"] *= 1.1
        out["debt_gni_pct"] += 5.0
        out["trade_balance_gdp_pct"] -= 1.0
        out["gini_index"] *= 1.02
        out["shock_impact_score"] *= 1.1
    elif scenario == "infrastructure_investment":
        out["gdp_growth_pct"] += 0.7
        out["resilience_score"] *= 1.05
        out["unemployment_rate"] *= 0.97
        out["debt_gni_pct"] += 1.0
        out["trade_balance_gdp_pct"] += 0.2
    elif scenario == "debt_stabilization":
        out["debt_gni_pct"] -= 3.0
        out["gdp_growth_pct"] -= 0.1
        out["resilience_score"] *= 1.02
    elif scenario == "climate_adaptation":
        out["resilience_score"] *= 1.1
        out["shock_impact_score"] *= 0.9
        out["debt_gni_pct"] += 1.0
    return out

# Recommended actions text
def rec_text(scenario):
    return {
        "social_spending": "Expand social transfers, education, and healthcare to cut poverty.",
        "trade_diversification": "Diversify exports and markets; strengthen logistics and value chains.",
        "global_crisis": "Prepare fiscal buffers, ensure FX liquidity, expand safety nets.",
        "infrastructure_investment": "Invest in transport, energy, and digital infrastructure to boost growth.",
        "debt_stabilization": "Stabilize debt through fiscal reforms, tax base expansion, and subsidy rationalization.",
        "climate_adaptation": "Invest in climate adaptation: resilient infrastructure, agriculture, and insurance.",
    }.get(scenario, "Maintain baseline policies.")

# Scenarios considered
scenarios = [
    "social_spending",
    "trade_diversification",
    "global_crisis",
    "infrastructure_investment",
    "debt_stabilization",
    "climate_adaptation",
]

# Evaluate all scenarios for each country
rows = []
for country, row in base2030.iterrows():
    base_row = row.to_frame().T
    for scen in scenarios:
        proj = apply_scenario(base_row, scen).iloc[0].to_dict()
        out = {
            "country": country,
            "scenario": scen,
            "year": int(proj.get("year", 2030)),
            "projected_gdp_growth_2030": float(proj.get("gdp_growth_pct", np.nan)),
            "projected_poverty_2030": float(proj.get("poverty_rate", np.nan)),
            "projected_resilience_2030": float(proj.get("resilience_score", np.nan)),
            "projected_unemployment_2030": float(proj.get("unemployment_rate", np.nan)),
            "projected_debt_gni_pct_2030": float(proj.get("debt_gni_pct", np.nan)),
            "projected_trade_balance_gdp_pct_2030": float(proj.get("trade_balance_gdp_pct", np.nan)),
            "projected_gini_2030": float(proj.get("gini_index", np.nan)),
            "projected_shock_impact_2030": float(proj.get("shock_impact_score", np.nan)),
            "recommended_actions": rec_text(scen),
        }
        rows.append(out)

scen_df = pd.DataFrame(rows)

# --- Select ONE recommendation per country based on vulnerabilities ---
final_rows = []

for country, row in base2030.iterrows():
    poverty = row.get("poverty_rate", np.nan)
    debt = row.get("debt_gni_pct", np.nan)
    shock = row.get("shock_impact_score", np.nan)
    trade = row.get("trade_balance_gdp_pct", np.nan)
    resilience = row.get("resilience_score", np.nan)

    if pd.isna(poverty): poverty = 25
    if pd.isna(debt): debt = 50
    if pd.isna(shock): shock = 40
    if pd.isna(trade): trade = 0
    if pd.isna(resilience): resilience = 50

    if poverty > 30:
        scen = "social_spending"
    elif debt > 70:
        scen = "debt_stabilization"
    elif shock > 60:
        scen = "climate_adaptation"
    elif trade < -5:
        scen = "trade_diversification"
    elif resilience < 40:
        scen = "infrastructure_investment"
    else:
        scen = "baseline"

    final_rows.append({
        "country": country,
        "year": 2030,
        "recommended_scenario": scen,
        "recommended_actions": rec_text(scen),
        "poverty_rate": poverty,
        "resilience_score": resilience,
        "debt_gni_pct": debt,
        "trade_balance_gdp_pct": trade,
        "shock_impact_score": shock
    })

final_df = pd.DataFrame(final_rows)
final_df.to_csv("policy_recommendations_single.csv", index=False)

print("✅ Saved: policy_recommendations_single.csv")
print(final_df.head(10))