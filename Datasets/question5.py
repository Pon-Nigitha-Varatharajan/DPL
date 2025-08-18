import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Files (adjust names if different)
INTEGRATED = "integrated_country_year.csv"
FEATURED = "feature_engineered_dataset.csv"             # optional (has engineered features)
FORECASTS_BASE = "forecasts_baseline.csv"               # optional (existing baseline GDP forecasts)

OUT_CSV = "youth_unemp_2030_predictions.csv"

# Scenario parameters
SIM_YEARS = list(range(2025, 2031))   # we forecast through 2030 inclusive
SLOWDOWN_START = 2026
SLOWDOWN_END = 2029
SLOWDOWN_GDP_PPT = -2.5   # reduce GDP growth by 2.5 percentage points in slowdown years

# Load integrated dataset
if not os.path.exists(INTEGRATED):
    raise FileNotFoundError(f"{INTEGRATED} not found in working directory")
df_int = pd.read_csv(INTEGRATED)
df_int.columns = [c.strip() for c in df_int.columns]

# helper to find column with many possible names
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy match by lowercase substring
    lowcols = {col.lower(): col for col in df.columns}
    for cand in candidates:
        for col_lower, col_orig in lowcols.items():
            if cand.lower() in col_lower:
                return col_orig
    return None

# Identify important columns (try multiple possible names)
col_youth = find_col(df_int, [
    "unemployment_youth_total", "Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)",
    "unemployment_youth_total_pct", "youth_unemployment", "youth_unemp", "unemployment_youth"
])
col_gdp_growth = find_col(df_int, ["gdp_growth_pct", "gdp_growth", "GDP growth (annual %)"])
col_resilience = find_col(df_int, ["resilience_score", "resilience"])
col_country = find_col(df_int, ["country", "Country", "Country Name"])
col_year = find_col(df_int, ["year", "Year", "Year_Code"])

if col_country is None or col_year is None:
    raise KeyError("Could not find country/year columns in integrated file")

if col_gdp_growth is None:
    raise KeyError("Could not find GDP growth column in integrated file")

if col_youth is None:
    raise KeyError("Could not find a youth unemployment column in integrated file. Expected names like 'unemployment_youth_total'.")

# normalize names
df = df_int.rename(columns={col_country: "country", col_year: "year", col_gdp_growth: "gdp_growth_pct", col_youth: "youth_unemp_pct"})
if col_resilience:
    df = df.rename(columns={col_resilience: "resilience_score"})
else:
    df["resilience_score"] = np.nan

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df.sort_values(["country", "year"]).reset_index(drop=True)

# prepare historical training panel where youth_unemp_pct and gdp_growth_pct are present
panel = df[["country", "year", "youth_unemp_pct", "gdp_growth_pct", "resilience_score"]].copy()
panel["youth_unemp_pct"] = pd.to_numeric(panel["youth_unemp_pct"], errors="coerce")
panel["gdp_growth_pct"] = pd.to_numeric(panel["gdp_growth_pct"], errors="coerce")
panel["resilience_score"] = pd.to_numeric(panel["resilience_score"], errors="coerce")

# create lag (1 year) of youth unemployment by country
panel["youth_unemp_lag1"] = panel.groupby("country")["youth_unemp_pct"].shift(1)

# keep rows with lag and predictors
train = panel.dropna(subset=["youth_unemp_pct", "youth_unemp_lag1", "gdp_growth_pct"]).copy()
if train.shape[0] < 30:
    # fallback: relax requirement to use available rows (still need target)
    train = panel.dropna(subset=["youth_unemp_pct", "gdp_growth_pct"]).copy()
    train["youth_unemp_lag1"] = train.groupby("country")["youth_unemp_pct"].shift(1).fillna(train["youth_unemp_pct"].mean())

# build regression matrix
X = train[["youth_unemp_lag1", "gdp_growth_pct", "resilience_score"]].fillna(0.0)
X = sm.add_constant(X)
y = train["youth_unemp_pct"]

model = sm.OLS(y, X).fit()

# Prepare base GDP growth forecasts for each country-year 2025-2030
# Prefer using existing forecasts file if present
forecasts = None
if os.path.exists(FORECASTS_BASE):
    try:
        fdf = pd.read_csv(FORECASTS_BASE)
        fdf.columns = [c.strip() for c in fdf.columns]
        cf = find_col(fdf, ["country"])
        yf = find_col(fdf, ["year"])
        gg = find_col(fdf, ["gdp_growth_pct","gdp_growth"])
        if cf and yf and gg:
            fdf = fdf.rename(columns={cf:"country", yf:"year", gg:"gdp_growth_pct"})[["country","year","gdp_growth_pct"]]
            forecasts = fdf
    except Exception:
        forecasts = None

# If no forecasts file, project gdp_growth_pct using last observed value per country
if forecasts is None:
    recent = df.sort_values(["country","year"]).groupby("country").tail(3)
    recent = recent[["country","gdp_growth_pct","year"]].copy()
    recent_avg = recent.groupby("country")["gdp_growth_pct"].mean().reset_index().rename(columns={"gdp_growth_pct":"recent_gdp_growth"})
    # use recent_avg as projected growth for SIM_YEARS
    countries = df["country"].dropna().unique()
    rows = []
    for c in countries:
        base_val = recent_avg.loc[recent_avg["country"]==c,"recent_gdp_growth"]
        base_val = float(base_val.values[0]) if not base_val.empty and not pd.isna(base_val.values[0]) else 1.0
        for y in SIM_YEARS:
            rows.append({"country": c, "year": y, "gdp_growth_pct": base_val})
    forecasts = pd.DataFrame(rows)

# Merge resilience (last known) into forecasts
res_latest = df.sort_values(["country","year"]).groupby("country").tail(1)[["country","resilience_score"]].set_index("country")
forecasts = forecasts.merge(res_latest.reset_index(), on="country", how="left")
forecasts["resilience_score"] = forecasts["resilience_score"].fillna(0.0)

# Build baseline forecast dataframe keyed by country and year
forecasts = forecasts.sort_values(["country","year"]).reset_index(drop=True)

# Create baseline youth unemployment projection iteratively using the dynamic model:
# y_t = const + b1*y_{t-1} + b2*gdp_growth_t + b3*resilience
coef = model.params.to_dict()
const = coef.get("const", 0.0)
b_lag = coef.get("youth_unemp_lag1", 0.0)
b_gdp = coef.get("gdp_growth_pct", 0.0)
b_res = coef.get("resilience_score", 0.0)

# prepare last known youth_unemp per country to seed forecasts
last_known = panel.sort_values(["country","year"]).groupby("country").tail(1)[["country","youth_unemp_pct"]].set_index("country")["youth_unemp_pct"].to_dict()

def run_projection(fore_df, gdp_adjustments):
    result_rows = []
    # create a copy and apply adjustments (gdp_adjustments is dict year->additive adjustment in percentage points)
    fdf = fore_df.copy()
    fdf["gdp_growth_adj"] = fdf["gdp_growth_pct"].astype(float) + fdf["year"].map(gdp_adjustments).fillna(0.0)
    fdf = fdf.sort_values(["country","year"])
    for country, group in fdf.groupby("country"):
        prev_y = float(last_known.get(country, np.nan))
        if np.isnan(prev_y):
            # seed with mean youth unemp from training set
            prev_y = float(train["youth_unemp_pct"].mean())
        for _, row in group.iterrows():
            gx = float(row["gdp_growth_adj"])
            res = float(row.get("resilience_score", 0.0) or 0.0)
            ypred = const + b_lag * prev_y + b_gdp * gx + b_res * res
            # enforce bounds 0-100
            ypred = max(0.0, min(100.0, ypred))
            result_rows.append({"country": country, "year": int(row["year"]), "youth_unemp_pct": ypred, "gdp_growth_pct_adj": gx, "resilience_score": res})
            prev_y = ypred
    return pd.DataFrame(result_rows)

# Baseline: no adjustment
baseline_proj = run_projection(forecasts, gdp_adjustments={})

# Slowdown scenario: reduce gdp_growth by SLOWDOWN_GDP_PPT for years SLOWDOWN_START..SLOWDOWN_END
gdp_adj = {yr: SLOWDOWN_GDP_PPT for yr in range(SLOWDOWN_START, SLOWDOWN_END+1)}
slow_proj = run_projection(forecasts, gdp_adjustments=gdp_adj)

# Extract 2030 results and flag >25%
def extract_2030(df_proj):
    df30 = df_proj[df_proj["year"] == 2030].copy()
    df30["youth_unemp_gt_25pct"] = df30["youth_unemp_pct"] > 25.0
    return df30

res_baseline_2030 = extract_2030(baseline_proj).set_index("country")
res_slow_2030 = extract_2030(slow_proj).set_index("country")

# combine
countries_all = sorted(set(res_baseline_2030.index).union(set(res_slow_2030.index)))
out_rows = []
for c in countries_all:
    b = res_baseline_2030.loc[c] if c in res_baseline_2030.index else pd.Series()
    s = res_slow_2030.loc[c] if c in res_slow_2030.index else pd.Series()
    out_rows.append({
        "country": c,
        "youth_unemp_2030_baseline_pct": float(b.get("youth_unemp_pct", np.nan)),
        "youth_unemp_2030_baseline_flag_gt_25pct": bool(b.get("youth_unemp_gt_25pct", False)),
        "youth_unemp_2030_slowdown_pct": float(s.get("youth_unemp_pct", np.nan)),
        "youth_unemp_2030_slowdown_flag_gt_25pct": bool(s.get("youth_unemp_gt_25pct", False)),
        "gdp_growth_2030_baseline_pct": float(b.get("gdp_growth_pct_adj", np.nan)),
        "gdp_growth_2030_slowdown_pct": float(s.get("gdp_growth_pct_adj", np.nan))
    })

out_df = pd.DataFrame(out_rows)
out_df = out_df.sort_values("youth_unemp_2030_slowdown_pct", ascending=False).reset_index(drop=True)
out_df.to_csv(OUT_CSV, index=False)

print("Saved predictions to", OUT_CSV)
print("\nCountries predicted to have youth unemployment >25% in 2030 under slowdown (flag = True):")
print(out_df[out_df["youth_unemp_2030_slowdown_flag_gt_25pct"]][["country","youth_unemp_2030_slowdown_pct"]].to_string(index=False))