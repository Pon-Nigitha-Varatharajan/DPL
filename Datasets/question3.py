import pandas as pd
import numpy as np

# --- Load datasets ---
prod_file = "crop_and_livestock.csv"           # FAO or similar crop yield/production file
dis_file = "disasters_by_type.csv"          # Disaster severity file
int_file = "integrated_country_year.csv"    # Macro dataset

prod_df = pd.read_csv(prod_file)
dis_df = pd.read_csv(dis_file)
int_df = pd.read_csv(int_file)

# --- Clean column names ---
prod_df.columns = [c.strip().lower() for c in prod_df.columns]
dis_df.columns = [c.strip().lower() for c in dis_df.columns]
int_df.columns = [c.strip().lower() for c in int_df.columns]

# --- Crop production: estimate baseline yield ---
if "area" not in prod_df.columns or "year" not in prod_df.columns:
    raise KeyError("Production file must contain 'Area' (country) and 'Year' columns")

if "value" not in prod_df.columns:
    raise KeyError("Production file must contain 'Value' column with production/tons")

prod_latest = (
    prod_df.groupby("area")
    .apply(lambda g: g.sort_values("year").tail(3)["value"].mean() if not g.empty else np.nan)
    .reset_index()
    .rename(columns={0: "baseline_yield"})
)

prod_latest.rename(columns={"area": "country"}, inplace=True)

# --- Disaster data: get drought severity multiplier ---
if "disaster_type" not in dis_df.columns or "year" not in dis_df.columns:
    raise KeyError("Disaster file must contain 'disaster_type' and 'year'")

drought_hist = dis_df[dis_df["disaster_type"].str.contains("drought", case=False, na=False)]

# average severity factor for droughts (using total_damage_adjusted_usd proxy if exists)
if "total_damage_adjusted_usd" in drought_hist.columns:
    drought_severity = drought_hist.groupby("country")["total_damage_adjusted_usd"].mean().reset_index()
    drought_severity["drought_impact_frac"] = np.log1p(drought_severity["total_damage_adjusted_usd"]) / 1e5
else:
    drought_severity = pd.DataFrame({"country": drought_hist["country"].unique(), "drought_impact_frac": 0.15})

# --- Integrated dataset: get GDP and export share ---
if "gdp_current_usd" not in int_df.columns:
    raise KeyError("integrated_country_year.csv must contain gdp_current_usd column")

if "exports_goods_services_gdp_pct" not in int_df.columns:
    raise KeyError("integrated_country_year.csv must contain exports_goods_services_gdp_pct column")

# use latest year available
int_latest = int_df.sort_values("year").groupby("country").tail(1)

# compute total exports in USD
int_latest["total_exports_usd"] = int_latest["gdp_current_usd"] * (int_latest["exports_goods_services_gdp_pct"] / 100.0)

# assume agriculture is ~10–20% of exports depending on crop yields
# proxy: countries with higher crop yields -> higher agri share
yield_scale = prod_latest.set_index("country")["baseline_yield"] / prod_latest["baseline_yield"].max()
yield_scale = yield_scale.fillna(0.1)

int_latest = int_latest.set_index("country")
int_latest["agri_export_share"] = 0.1 + 0.2 * yield_scale.reindex(int_latest.index).fillna(0.1)

int_latest["agri_exports_usd"] = int_latest["total_exports_usd"] * int_latest["agri_export_share"]

# --- Simulate 3-year drought (2028–2030) ---
impact_df = int_latest.reset_index()[["country", "gdp_current_usd", "total_exports_usd", "agri_exports_usd"]]
impact_df = impact_df.merge(drought_severity[["country", "drought_impact_frac"]], on="country", how="left")

impact_df["drought_impact_frac"] = impact_df["drought_impact_frac"].fillna(0.15)

# compounding effect of 3 years: (1 - f)^3
impact_df["ag_export_loss_usd"] = impact_df["agri_exports_usd"] * (1 - (1 - impact_df["drought_impact_frac"]) ** 3)
impact_df["ag_export_loss_pct"] = impact_df["ag_export_loss_usd"] / impact_df["agri_exports_usd"] * 100

# --- Top 10 vulnerable countries ---
top10 = impact_df.sort_values("ag_export_loss_usd", ascending=False).head(10)

print("\nTop 10 countries by agricultural export USD loss in 2030 (3-year drought 2028-2030):")
print(top10[["country", "ag_export_loss_usd", "ag_export_loss_pct"]])

# --- Save output ---
impact_df.to_csv("question3_drought_agri_impact_2030.csv", index=False)
print("\nSaved results to question3_drought_agri_impact_2030.csv")