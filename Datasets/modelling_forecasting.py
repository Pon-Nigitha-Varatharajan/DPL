import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings("ignore")

print(" Loading dataset...")
df = pd.read_csv("feature_engineered_dataset.csv")
df = df.sort_values(["country", "year"])
print(f" Dataset: {df.shape}")

forecast_years = list(range(2025, 2031))

# GDP Growth Forecasting (VAR)
print("\n Forecasting GDP Growth using VAR...")

gdp_forecasts = []
for country in df["country"].unique():
    cols_needed = ["year", "gdp_growth_pct", "inflation_consumer_prices_pct", "trade_gdp_pct"]
    country_df = df[df["country"] == country][[c for c in cols_needed if c in df.columns]].dropna()

    if len(country_df) < 6 or "gdp_growth_pct" not in country_df.columns:
        continue

    country_df = country_df.set_index("year")
    try:
        country_df = country_df.loc[:, country_df.nunique() > 1]
        if country_df.shape[1] < 2:
            continue
        model = VAR(country_df)
        results = model.fit(maxlags=2, ic="aic")
        forecast = results.forecast(country_df.values, steps=len(forecast_years))
        forecast_df = pd.DataFrame(forecast, columns=country_df.columns, index=forecast_years)
        for yr in forecast_years:
            row = {"country": country, "year": yr}
            for col in forecast_df.columns:
                row[col] = forecast_df.loc[yr, col]
            gdp_forecasts.append(row)
    except Exception as e:
        print(f"⚠️ Skipped {country} due to error: {e}")

gdp_df = pd.DataFrame(gdp_forecasts)
print(f" GDP forecasts generated: {gdp_df.shape}")

# Poverty Forecasting (RandomForest with lag)
print("\n Forecasting Poverty using RandomForest...")

possible_features = [
    "gdp_growth_pct",
    "trade_gdp_pct",
    "unemployment_total_pct_labor_force",   
    "unemployment_total_pct_labor_force_social",  
    "resilience_score",
    "shock_impact_score",
    "total_damage_adjusted_usd",
    "trade_dependency_index"
]
features = [col for col in possible_features if col in df.columns]

target = "poverty_headcount_ratio_3ppp_pct"
if target not in df.columns:
    raise ValueError(f" Target column '{target}' not found in dataset!")

print(f" Using features for poverty model: {features}")
valid_subset = [col for col in features + [target] if col in df.columns]
train_df = df.dropna(subset=valid_subset).copy()
train_df["poverty_lag1"] = train_df.groupby("country")[target].shift(1)
train_df = train_df.dropna(subset=["poverty_lag1"])

X_train = train_df[features + ["poverty_lag1"]]
y_train = train_df[target]

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

poverty_preds = []
for country in gdp_df["country"].unique():
    last_known = df[df["country"] == country].iloc[-1]
    last_poverty = last_known.get(target, np.nan)

    for yr in forecast_years:
        row = gdp_df[(gdp_df["country"] == country) & (gdp_df["year"] == yr)]
        if row.empty:
            continue
        row = row.iloc[0]

        feat_row = []
        for f in features:
            if f in row.index:  
                feat_row.append(row[f])
            else:              
                feat_row.append(last_known.get(f, np.nan))
        feat_row.append(last_poverty)

        if np.isnan(feat_row).any():
            continue

        poverty_pred = rf.predict([feat_row])[0]
        poverty_preds.append({"country": country, "year": yr, "poverty_rate": poverty_pred})
        last_poverty = poverty_pred 

poverty_df = pd.DataFrame(poverty_preds)
print(f" Poverty forecasts generated: {poverty_df.shape}")

# Resilience Forecasting (Ridge Regression)
print("\nForecasting Resilience using Ridge Regression...")
res_possible = [
    "trade_balance_gdp_pct", "fdi_net_inflows_gdp_pct",
    "external_debt_stocks_gni_pct", "shock_impact_score",
    "total_damage_adjusted_usd"
]
res_features = [col for col in res_possible if col in df.columns]
res_target = "resilience_score"

if res_target not in df.columns:
    print(" Resilience score not in dataset. Skipping resilience forecasting.")
    resilience_df = pd.DataFrame()
else:
    print(f" Using features for resilience model: {res_features}")
    train_res = df.dropna(subset=[c for c in res_features + [res_target] if c in df.columns])
    X_res, y_res = train_res[res_features], train_res[res_target]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_res, y_res)

    resilience_preds = []
    for country in gdp_df["country"].unique():
        last_known = df[df["country"] == country].iloc[-1]
        for yr in forecast_years:
            feat_row = [last_known.get(f, np.nan) for f in res_features]
            if np.isnan(feat_row).any():
                continue
            pred = ridge.predict([feat_row])[0]
            resilience_preds.append({"country": country, "year": yr, "resilience_score": pred})

    resilience_df = pd.DataFrame(resilience_preds)
    print(f" Resilience forecasts generated: {resilience_df.shape}")

#Combine Forecasts
print("\n Combining all forecasts...")
baseline_df = gdp_df.merge(poverty_df, on=["country", "year"], how="left")
if not resilience_df.empty:
    baseline_df = baseline_df.merge(resilience_df, on=["country", "year"], how="left")
baseline_df.to_csv("forecasts_baseline.csv", index=False)
print(" Saved forecasts_baseline.csv")

#Scenario Simulations
print("\n Applying scenarios...")

def apply_scenario(df, scenario):
    df_copy = df.copy()
    if scenario == "social_spending":
        if "poverty_rate" in df_copy.columns:
            df_copy["poverty_rate"] *= 0.9
        if "gdp_growth_pct" in df_copy.columns:
            df_copy["gdp_growth_pct"] += 0.5
    elif scenario == "trade_diversification":
        if "resilience_score" in df_copy.columns:
            df_copy["resilience_score"] *= 1.15
        if "gdp_growth_pct" in df_copy.columns:
            df_copy["gdp_growth_pct"] += 0.3
    elif scenario == "global_crisis":
        if "gdp_growth_pct" in df_copy.columns:
            df_copy["gdp_growth_pct"] -= 2.5
        if "poverty_rate" in df_copy.columns:
            df_copy["poverty_rate"] *= 1.1
        if "resilience_score" in df_copy.columns:
            df_copy["resilience_score"] *= 0.85
    return df_copy

for scen in ["baseline", "social_spending", "trade_diversification", "global_crisis"]:
    scen_df = baseline_df.copy() if scen == "baseline" else apply_scenario(baseline_df, scen)
    scen_df.to_csv(f"forecasts_{scen}.csv", index=False)
    print(f" Saved forecasts_{scen}.csv")

print("\n All forecasting & scenario files generated successfully!")
print("\n Sample Baseline Forecasts:")
print(baseline_df.head(5))
