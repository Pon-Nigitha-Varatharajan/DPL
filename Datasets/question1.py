import pandas as pd
import numpy as np
import os

# files (adjust paths if needed)
BILATERAL_EXPORTS = "bilateral_export_data.csv"
INTEGRATED = "integrated_country_year.csv"
OUTFILE = "question1_trade_dependency_simulation_results.csv"

# parameters
TARGET_YEAR = 2024   # latest-year to compute partner shares from (fall back to last available)
SIM_YEAR = 2026
SHOCK_DROP = 0.40    # 40% drop

# load data
if not os.path.exists(BILATERAL_EXPORTS):
    raise FileNotFoundError(f"{BILATERAL_EXPORTS} not found")
if not os.path.exists(INTEGRATED):
    raise FileNotFoundError(f"{INTEGRATED} not found")

df_bil = pd.read_csv(BILATERAL_EXPORTS)   # expects columns: country, partner_country, year, export_value_usd
df_int = pd.read_csv(INTEGRATED)          # expects columns including: country, year, gdp_current_usd, gdp_growth_pct, trade_gdp_pct

# sanitize column names
df_bil.columns = [c.strip() for c in df_bil.columns]
df_int.columns = [c.strip() for c in df_int.columns]

# ensure numeric
df_bil['export_value_usd'] = pd.to_numeric(df_bil.get('export_value_usd', df_bil.get('primaryValue', np.nan)), errors='coerce')
df_bil['year'] = pd.to_numeric(df_bil['year'], errors='coerce').astype('Int64')

# pick latest year to compute partner shares (prefer TARGET_YEAR)
years_available = sorted(df_bil['year'].dropna().unique())
if TARGET_YEAR not in years_available:
    latest_year = int(years_available[-1])
else:
    latest_year = TARGET_YEAR

df_latest = df_bil[df_bil['year'] == latest_year].copy()
if df_latest.empty:
    raise ValueError(f"No bilateral export rows for year {latest_year}")

# aggregate exports by country-partner
agg = df_latest.groupby(['country', 'partner_country'], as_index=False).agg({'export_value_usd': 'sum'})

# total exports by country (from bilateral data)
total_by_country = agg.groupby('country', as_index=False)['export_value_usd'].sum().rename(columns={'export_value_usd': 'total_exports_usd'})

agg = agg.merge(total_by_country, on='country', how='left')
agg['partner_share'] = agg['export_value_usd'] / agg['total_exports_usd']
agg['partner_share'] = agg['partner_share'].fillna(0)

# compute HHI (Herfindahl) and top partner share per country
hh = agg.groupby('country').apply(lambda g: (g['partner_share'] ** 2).sum()).rename('hh_index').reset_index()
top = agg.sort_values(['country', 'partner_share'], ascending=[True, False]).groupby('country').first().reset_index()
top = top[['country', 'partner_country', 'partner_share', 'export_value_usd']].rename(columns={
    'partner_country': 'top_partner',
    'partner_share': 'top_partner_share',
    'export_value_usd': 'top_partner_exports_usd'
})

country_trade = total_by_country.merge(hh, on='country', how='left').merge(top, on='country', how='left')

# bring in trade_gdp_pct and GDP info from integrated dataset using latest available year per country
# find latest year row per country in integrated dataset
df_int['year'] = pd.to_numeric(df_int['year'], errors='coerce').astype('Int64')
df_int_sorted = df_int.sort_values(['country', 'year'])
latest_int = df_int_sorted.groupby('country').tail(1).set_index('country')

# join structural metrics
country_trade = country_trade.set_index('country').join(latest_int[['gdp_current_usd', 'gdp_growth_pct', 'trade_gdp_pct']], how='left').reset_index()

# compute Trade Dependency Index (TDI) = top_partner_share * trade_gdp_pct (as percentage points)
country_trade['trade_gdp_pct'] = pd.to_numeric(country_trade['trade_gdp_pct'], errors='coerce')
country_trade['top_partner_share'] = pd.to_numeric(country_trade['top_partner_share'], errors='coerce').fillna(0)
country_trade['TDI'] = country_trade['top_partner_share'] * country_trade['trade_gdp_pct']  # higher = more dependent

# identify top 3 vulnerable nations by TDI (descending)
ranked = country_trade.sort_values('TDI', ascending=False).reset_index(drop=True)
top3 = ranked.head(3).copy()

# Helper: project GDP to SIM_YEAR using recent growth
def project_gdp(row, target_year=SIM_YEAR):
    try:
        gdp = float(row.get('gdp_current_usd', np.nan))
        # attempt to compute avg growth from integrated dataset (last 3 years)
        c = row['country']
        hist = df_int[df_int['country'] == c].sort_values('year', ascending=False).head(3)
        if not hist.empty and hist['gdp_growth_pct'].notna().any():
            gr = hist['gdp_growth_pct'].dropna().astype(float).mean()
        else:
            gr = float(row.get('gdp_growth_pct', 0) if pd.notna(row.get('gdp_growth_pct')) else 0)
        # if gdp missing, return NaN
        if np.isnan(gdp):
            return np.nan
        years = target_year - int(latest_year) if target_year >= latest_year else 0
        proj = gdp * ((1 + gr/100) ** years)
        return proj
    except Exception:
        return np.nan

# compute projected exports in SIM_YEAR from trade_gdp_pct * projected_gdp
results = []
for _, r in top3.iterrows():
    country = r['country']
    top_partner = r['top_partner']
    top_share = r['top_partner_share']
    trade_pct = r.get('trade_gdp_pct', np.nan)
    gdp_latest = r.get('gdp_current_usd', np.nan)

    gdp_2026 = project_gdp(r, SIM_YEAR)
    # if trade_pct missing, try to take last known from integrated
    if pd.isna(trade_pct):
        try:
            trade_pct = float(df_int[(df_int['country']==country) & (df_int['year']<=latest_year)].sort_values('year', ascending=False).iloc[0]['trade_gdp_pct'])
        except Exception:
            trade_pct = np.nan

    # projected total exports in SIM_YEAR
    projected_exports_2026 = (trade_pct/100.0) * gdp_2026 if pd.notna(trade_pct) and pd.notna(gdp_2026) else np.nan

    # projected exports to top partner in 2026 (assuming same partner share)
    exports_to_top_2026 = top_share * projected_exports_2026 if pd.notna(projected_exports_2026) else np.nan

    # shock: partner cuts imports by SHOCK_DROP
    export_loss_usd = SHOCK_DROP * exports_to_top_2026 if pd.notna(exports_to_top_2026) else np.nan

    # GDP loss percent
    gdp_loss_pct = (export_loss_usd / gdp_2026 * 100) if pd.notna(export_loss_usd) and pd.notna(gdp_2026) and gdp_2026 != 0 else np.nan

    results.append({
        'country': country,
        'top_partner': top_partner,
        'top_partner_share_pct': top_share * 100,
        'trade_gdp_pct': trade_pct,
        'gdp_latest_year': latest_year,
        'gdp_latest_usd': gdp_latest,
        f'gdp_{SIM_YEAR}_proj_usd': gdp_2026,
        'projected_exports_2026_usd': projected_exports_2026,
        'exports_to_top_partner_2026_usd': exports_to_top_2026,
        'export_loss_usd_if_top_partner_cuts_40pct': export_loss_usd,
        'gdp_loss_pct_if_top_partner_cuts_40pct': gdp_loss_pct,
        'TDI': r['TDI'],
        'hh_index': r['hh_index'],
        'total_exports_latest_usd': r['total_exports_usd']
    })

res_df = pd.DataFrame(results)

# print top3 summary
print("Top 3 countries by Trade Dependency Index (TDI):")
print(ranked[['country','TDI','top_partner','top_partner_share','trade_gdp_pct']].head(10).to_string(index=False))

print("\nSimulation results for top 3 (saved to file):")
print(res_df.to_string(index=False))

res_df.to_csv(OUTFILE, index=False)
print(f"\nSaved results to {OUTFILE}")