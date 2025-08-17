import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("removing duplicates in my disaster sector")

input_path = "integrated_country_year.csv"  
df = pd.read_csv(input_path)

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

sum_keywords = ["num_disasters", "deaths", "affected", "homeless", "damage"]
agg_dict = {}

for col in numeric_cols:
    if any(key in col.lower() for key in sum_keywords):
        agg_dict[col] = "sum"  
    else:
        agg_dict[col] = "mean"


for col in non_numeric_cols:
    if col not in ["country", "year"]:
        agg_dict[col] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None

df_clean = df.groupby(["country", "year"], as_index=False).agg(agg_dict)

output_path = "integrated_country_year_deduplicated.csv"
df_clean.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df_clean.shape}")

input_path = "integrated_country_year.csv"  
df = pd.read_csv(input_path)
print("start with feature engineering")

df["trade_dependency_index"] = df["trade_gdp_pct"]

resilience_factors = ["current_account_balance_gdp_pct", "fdi_net_inflows_gdp_pct", "external_debt_stocks_gni_pct"]
scaler = MinMaxScaler()
df["resilience_score"] = scaler.fit_transform(df[resilience_factors].fillna(0)).mean(axis=1)

if "government_spending_gdp_pct" in df.columns:
    df["spending_efficiency"] = df["gdp_per_capita_current_usd"] / df["government_spending_gdp_pct"]

if "total_damage_adjusted_usd" in df.columns:
    df["shock_impact_score"] = df["total_damage_adjusted_usd"] / df["gdp_current_usd"]

lag_columns = [
    "gdp_growth_pct",
    "trade_gdp_pct",
    "inflation_consumer_prices_pct",
    "unemployment_total_pct_labor_force",
    "poverty_headcount_ratio_3ppp_pct"
]

for col in lag_columns:
    if col in df.columns:
        df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("country")[col].shift(2)

if "exports_goods_services_gdp_pct" in df.columns and "imports_goods_services_gdp_pct" in df.columns:
    df["exports_imports_ratio"] = df["exports_goods_services_gdp_pct"] / df["imports_goods_services_gdp_pct"]

if "gdp_per_capita_current_usd" in df.columns:
    df["gdp_per_capita_growth"] = df.groupby("country")["gdp_per_capita_current_usd"].pct_change() * 100

if "exports_goods_services_gdp_pct" in df.columns and "imports_goods_services_gdp_pct" in df.columns:
    df["trade_balance_gdp_pct"] = df["exports_goods_services_gdp_pct"] - df["imports_goods_services_gdp_pct"]

output_path = "feature_engineered_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Feature engineered dataset saved to: {output_path}")
print(f"Final shape: {df.shape}")

df = pd.read_csv("feature_engineered_dataset.csv")
duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {len(duplicate_rows)}")
print(duplicate_rows.head())
