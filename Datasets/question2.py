import pandas as pd
import numpy as np
import os

BILATERAL_FILE = "bilateral_export_data.csv"
INTEGRATED_FILE = "integrated_country_year.csv"
SIM_YEAR = 2028
CHINA_NAME = "China"
EXPORT_DROP_FRAC = 0.25
MAX_ITERS = 10
CONVERGENCE_THRESH = 1e-6

if not os.path.exists(BILATERAL_FILE):
    raise FileNotFoundError(BILATERAL_FILE + " not found")
if not os.path.exists(INTEGRATED_FILE):
    raise FileNotFoundError(INTEGRATED_FILE + " not found")

df_bil = pd.read_csv(BILATERAL_FILE)
df_int = pd.read_csv(INTEGRATED_FILE)

df_bil.columns = [c.strip() for c in df_bil.columns]
for col in ["country", "partner_country", "year"]:
    if col not in df_bil.columns:
        raise KeyError(f"Expected column '{col}' in {BILATERAL_FILE}")
val_cols = [c for c in df_bil.columns if c.lower() in ("export_value_usd","primaryvalue","primary_value","export_value","value","fobvalue","primaryvalue")]
if 'export_value_usd' in df_bil.columns:
    df_bil['export_value_usd'] = pd.to_numeric(df_bil['export_value_usd'], errors='coerce')
else:
    found = None
    for cand in ['primaryValue','primary_value','export_value','value','fobvalue']:
        if cand in df_bil.columns:
            df_bil['export_value_usd'] = pd.to_numeric(df_bil[cand], errors='coerce')
            found = cand
            break
    if found is None:
        raise KeyError("Could not find an export value column in bilateral file; add/export_value_usd or primaryValue column.")

df_bil['year'] = pd.to_numeric(df_bil['year'], errors='coerce').astype('Int64')

df_int.columns = [c.strip() for c in df_int.columns]
if 'country' not in df_int.columns or 'year' not in df_int.columns:
    raise KeyError("integrated_country_year.csv must contain 'country' and 'year' columns")
df_int['year'] = pd.to_numeric(df_int['year'], errors='coerce').astype('Int64')
if 'gdp_current_usd' not in df_int.columns:
    candidates = ['gdp_current_usd','GDP (current US$)','gdp_current']
    found = None
    for cand in candidates:
        if cand in df_int.columns:
            df_int['gdp_current_usd'] = pd.to_numeric(df_int[cand], errors='coerce')
            found = cand
            break
    if found is None:
        raise KeyError("integrated file missing gdp_current_usd column")
if 'gdp_growth_pct' not in df_int.columns:
    for cand in ['gdp_growth_pct','gdp_growth','GDP growth (annual %)']:
        if cand in df_int.columns:
            df_int['gdp_growth_pct'] = pd.to_numeric(df_int[cand], errors='coerce')
            break

years_bil = sorted(df_bil['year'].dropna().unique())
if not years_bil:
    raise ValueError("No year data in bilateral exports file")
latest_bil_year = int(years_bil[-1])

df_to_china = df_bil[df_bil['partner_country'].str.strip().eq(CHINA_NAME) & df_bil['year'].notna()].copy()
if df_to_china.empty:
    raise ValueError("No bilateral rows where partner_country == China found. Check partner naming.")

exports_to_china_latest = df_to_china[df_to_china['year'] == latest_bil_year].groupby('country', as_index=False)['export_value_usd'].sum().rename(columns={'export_value_usd':'exports_to_china_latest_usd'})

last_by_country = df_to_china.sort_values(['country','year']).groupby('country').tail(1)[['country','year','export_value_usd']].rename(columns={'year':'last_year','export_value_usd':'exports_to_china_last_usd'})
exports_to_china = exports_to_china_latest.set_index('country').join(last_by_country.set_index('country'), how='outer').reset_index()
exports_to_china['exports_to_china_latest_usd'] = exports_to_china['exports_to_china_latest_usd'].fillna(exports_to_china['exports_to_china_last_usd'])
exports_to_china = exports_to_china[['country','exports_to_china_latest_usd']].copy()
exports_to_china['exports_to_china_latest_usd'] = pd.to_numeric(exports_to_china['exports_to_china_latest_usd'], errors='coerce').fillna(0)

df_int_sorted = df_int.sort_values(['country','year'])
recent_growth = df_int_sorted.groupby('country').apply(lambda g: g.sort_values('year',ascending=False).head(3)['gdp_growth_pct'].dropna().astype(float).mean() if g['gdp_growth_pct'].notna().any() else np.nan).rename('recent_growth_pct').reset_index()
exports_to_china = exports_to_china.merge(recent_growth, on='country', how='left')

delta_years = SIM_YEAR - latest_bil_year if SIM_YEAR >= latest_bil_year else 0
def project_val(row):
    base = row['exports_to_china_latest_usd']
    gr = row['recent_growth_pct']
    if pd.isna(gr):
        return base
    try:
        return base * ((1 + gr/100.0) ** delta_years)
    except Exception:
        return base

exports_to_china['exports_to_china_2028_usd'] = exports_to_china.apply(project_val, axis=1)

df_china_exports = df_bil[df_bil['country'].str.strip().eq(CHINA_NAME)].copy()
china_exports_latest = df_china_exports[df_china_exports['year'] == latest_bil_year]['export_value_usd'].sum()
if china_exports_latest == 0:
    china_latest_row = df_china_exports.sort_values('year').groupby('country').tail(1)
    china_exports_latest = china_latest_row['export_value_usd'].sum()
china_recent_growth = recent_growth[recent_growth['country']==CHINA_NAME]['recent_growth_pct'].values
china_recent_growth = float(china_recent_growth[0]) if len(china_recent_growth)>0 and not pd.isna(china_recent_growth[0]) else 0.0
china_exports_2028 = china_exports_latest * ((1 + china_recent_growth/100.0) ** delta_years)

china_export_loss = EXPORT_DROP_FRAC * china_exports_2028

china_gdp_row = df_int[df_int['country'].str.strip().eq(CHINA_NAME)].sort_values('year').tail(1)
if china_gdp_row.empty:
    raise ValueError("China GDP not found in integrated file")
china_gdp_latest = float(china_gdp_row['gdp_current_usd'].values[0])

china_gdp_frac_drop = china_export_loss / china_gdp_latest if china_gdp_latest != 0 else 0.0
china_gdp_frac_drop = float(china_gdp_frac_drop)

print(f"China exports 2028 (proj): {china_exports_2028:,.0f} USD")
print(f"China export loss (25%): {china_export_loss:,.0f} USD -> GDP fractional drop {china_gdp_frac_drop:.6f}")

exports_to_china['direct_loss_to_partner_usd'] = exports_to_china['exports_to_china_2028_usd'] * china_gdp_frac_drop

countries = set(exports_to_china['country'].unique()).union(set(df_bil['country'].unique())).union(set(df_int['country'].unique()))
countries = sorted([c for c in countries if pd.notna(c)])
cumulative_loss = {c: 0.0 for c in countries}

for _, r in exports_to_china.iterrows():
    exporter = r['country']
    loss = float(r['direct_loss_to_partner_usd'])
    cumulative_loss[exporter] = cumulative_loss.get(exporter, 0.0) + loss

trade_mat = df_bil[df_bil['year'] == latest_bil_year].groupby(['country','partner_country'], as_index=False)['export_value_usd'].sum()
if trade_mat.empty:
    trade_mat = df_bil.groupby(['country','partner_country'], as_index=False)['export_value_usd'].sum()

trade_map = {}
for _, row in trade_mat.iterrows():
    a = row['country']
    b = row['partner_country']
    v = float(row['export_value_usd'])
    trade_map.setdefault(a, {})[b] = v

def project_gdp_country(country):
    hist = df_int[df_int['country']==country].sort_values('year',ascending=False)
    if hist.empty:
        return np.nan
    latest_gdp = hist['gdp_current_usd'].dropna().astype(float).iloc[0]
    recent_gr = hist.head(3)['gdp_growth_pct'].dropna().astype(float)
    if recent_gr.empty:
        avg_gr = 0.0
    else:
        avg_gr = recent_gr.mean()
    years = SIM_YEAR - int(hist['year'].dropna().max()) if SIM_YEAR >= int(hist['year'].dropna().max()) else 0
    try:
        return latest_gdp * ((1 + avg_gr/100.0) ** years)
    except Exception:
        return latest_gdp

gdp_proj = {}
for c in countries:
    try:
        gdp_proj[c] = project_gdp_country(c)
    except Exception:
        gdp_proj[c] = np.nan

iter_losses = {c: float(exports_to_china.loc[exports_to_china['country']==c,'direct_loss_to_partner_usd'].sum()) if c in list(exports_to_china['country']) else 0.0 for c in countries}
for c in countries:
    iter_losses.setdefault(c, 0.0)

for it in range(MAX_ITERS):
    new_losses = {c: 0.0 for c in countries}
    for c in countries:
        loss_usd = iter_losses.get(c, 0.0)
        gdp_c = gdp_proj.get(c, np.nan)
        if pd.isna(gdp_c) or gdp_c == 0:
            frac = 0.0
        else:
            frac = loss_usd / gdp_c
        if frac <= 0:
            continue
        for p, dests in trade_map.items():
            val = dests.get(c, 0.0)
            if val <= 0:
                continue
            loss_to_p = val * frac
            new_losses[p] += loss_to_p
    total_new = 0.0
    for c in countries:
        inc = new_losses.get(c, 0.0)
        cumulative_loss[c] = cumulative_loss.get(c, 0.0) + inc
        total_new += inc
    print(f"Iteration {it+1}: incremental total loss = {total_new:,.2f} USD")
    if total_new < CONVERGENCE_THRESH:
        print("Converged — incremental losses negligible.")
        break
    iter_losses = new_losses

results = []
for c in countries:
    gdp_c = gdp_proj.get(c, np.nan)
    total_loss = cumulative_loss.get(c, 0.0)
    pct_loss = (total_loss / gdp_c * 100) if (not pd.isna(gdp_c) and gdp_c != 0) else np.nan
    results.append({'country': c, 'gdp_proj_2028_usd': gdp_c, 'total_trade_loss_usd': total_loss, 'gdp_pct_loss': pct_loss})

res_df = pd.DataFrame(results)
res_df = res_df.dropna(subset=['gdp_pct_loss'])
res_df = res_df.sort_values('gdp_pct_loss', ascending=False).reset_index(drop=True)

top5 = res_df.head(5)

print("\nTop 5 countries by GDP % loss due to cascading shock from China exports drop 25% in 2028:")
print(top5[['country','gdp_pct_loss','total_trade_loss_usd','gdp_proj_2028_usd']].to_string(index=False))

res_df.to_csv("question2_china_export_cascade_results_all.csv", index=False)
top5.to_csv("question2_china_export_cascade_top5.csv", index=False)
print("\nSaved question2_china_export_cascade_results_all.csv and question2_china_export_cascade_top5.csv")
