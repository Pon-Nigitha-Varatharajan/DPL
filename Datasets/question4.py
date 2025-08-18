import os
import pandas as pd
import numpy as np

BILATERAL_IMPORTS = "bilateral_import_data.csv"
CROP_FILE = "crop_and_livestock.csv"
INTEGRATED = "integrated_country_year.csv"
OUT_CSV = "question4_agri_import_dependency_and_ban_simulation.csv"

LATEST_YEAR_PREFERRED = 2024
TOP_N = 3
DEPENDENCY_THRESHOLD_TOP3 = 0.75
PRICE_PER_TONNE_USD = 300.0
PER_CAPITA_FOOD_KG_PER_YEAR = 400.0

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path + " not found")
    return pd.read_csv(path)

df_imp = load_csv(BILATERAL_IMPORTS)
df_crop = load_csv(CROP_FILE)
df_int = load_csv(INTEGRATED)

df_imp.columns = [c.strip() for c in df_imp.columns]
df_crop.columns = [c.strip() for c in df_crop.columns]
df_int.columns = [c.strip() for c in df_int.columns]

if 'year' in df_imp.columns:
    df_imp['year'] = pd.to_numeric(df_imp['year'], errors='coerce').astype('Int64')

val_col_candidates = ['import_value_usd','import_value','primaryValue','primary_value','primaryvalue','value','cifvalue']
imp_value_col = next((c for c in val_col_candidates if c in df_imp.columns), None)
if imp_value_col is None:
    raise KeyError("No import value column found in bilateral_import_data.csv")
df_imp['import_value_usd'] = pd.to_numeric(df_imp[imp_value_col], errors='coerce').fillna(0.0)

wgt_col_candidates = ['import_weight_tonnes','netWgt','import_weight','netwgt','altQty','qty']
imp_wgt_col = next((c for c in wgt_col_candidates if c in df_imp.columns), None)
if imp_wgt_col:
    df_imp['import_weight_tonnes'] = pd.to_numeric(df_imp[imp_wgt_col], errors='coerce')
    df_imp.loc[df_imp['import_weight_tonnes'] > 1e6, 'import_weight_tonnes'] = df_imp.loc[df_imp['import_weight_tonnes'] > 1e6, 'import_weight_tonnes'] / 1000.0
else:
    df_imp['import_weight_tonnes'] = np.nan

reporter_col = next((c for c in ['country','reporter','reporterDesc','reporterDesc.','reporter_desc'] if c in df_imp.columns), None)
partner_col = next((c for c in ['partner_country','partner','partnerDesc','partnerDesc.','partner_desc'] if c in df_imp.columns), None)
if reporter_col is None or partner_col is None:
    raise KeyError("Could not find reporter/partner columns in bilateral_import_data.csv")
df_imp['reporter'] = df_imp[reporter_col].astype(str).str.strip()
df_imp['partner'] = df_imp[partner_col].astype(str).str.strip()

if LATEST_YEAR_PREFERRED in df_imp['year'].dropna().unique():
    latest_year = LATEST_YEAR_PREFERRED
else:
    years = sorted(df_imp['year'].dropna().unique())
    if not years:
        latest_year = None
    else:
        latest_year = int(years[-1])
df_imp_ly = df_imp[df_imp['year'] == latest_year].copy() if latest_year is not None else df_imp.copy()

if 'cmdDesc' in df_imp_ly.columns:
    cmd_col = 'cmdDesc'
else:
    cmd_col = next((c for c in df_imp_ly.columns if 'cmd' in c.lower() or 'class' in c.lower() or 'commodity' in c.lower()), None)

ag_keywords = {'wheat','rice','maize','corn','soy','soybean','barley','sugar','coffee','cocoa','cotton','tea','banana','fruit','vegetable','meat','livestock','dairy','milk','poultry','potato','tomato','onion','oilseed','rapeseed','canola','olive'}
if cmd_col:
    df_imp_ly['cmd_lower'] = df_imp_ly[cmd_col].astype(str).str.lower()
    df_imp_ly['is_ag'] = df_imp_ly['cmd_lower'].apply(lambda s: any(k in s for k in ag_keywords))
else:
    df_imp_ly['is_ag'] = False

if df_imp_ly['is_ag'].sum() == 0:
    flow_guess_cols = [c for c in df_imp_ly.columns if 'flow' in c.lower() or 'desc' in c.lower()]
    for c in flow_guess_cols:
        df_imp_ly['is_ag'] = df_imp_ly['is_ag'] | df_imp_ly[c].astype(str).str.lower().apply(lambda s: any(k in s for k in ag_keywords))
if df_imp_ly['is_ag'].sum() == 0:
    df_imp_ly['is_ag'] = True

ag_df = df_imp_ly[df_imp_ly['is_ag']].copy()

agg_by_importer_partner = ag_df.groupby(['reporter','partner'], as_index=False).agg(imports_usd=('import_value_usd','sum'), imports_tonnes=('import_weight_tonnes','sum'))
total_ag_imports_by_importer = agg_by_importer_partner.groupby('reporter', as_index=False).agg(total_ag_imports_usd=('imports_usd','sum'), total_ag_imports_tonnes=('imports_tonnes','sum'))
agg_by_importer_partner = agg_by_importer_partner.merge(total_ag_imports_by_importer, on='reporter', how='left')
agg_by_importer_partner['partner_share'] = np.where(agg_by_importer_partner['total_ag_imports_usd']>0, agg_by_importer_partner['imports_usd'] / agg_by_importer_partner['total_ag_imports_usd'], 0.0)

def top_n_partners(df, n=3):
    df2 = df.sort_values('partner_share', ascending=False).head(n)
    partners = df2['partner'].tolist()
    shares = df2['partner_share'].tolist()
    usds = df2['imports_usd'].tolist()
    tonnes = df2['imports_tonnes'].tolist()
    while len(partners) < n:
        partners.append(None); shares.append(0.0); usds.append(0.0); tonnes.append(np.nan)
    return partners, shares, usds, tonnes

rows = []
importers = sorted(agg_by_importer_partner['reporter'].unique())
for imp in importers:
    sub = agg_by_importer_partner[agg_by_importer_partner['reporter']==imp]
    total_usd = float(sub['total_ag_imports_usd'].iloc[0]) if not sub.empty else 0.0
    partners, shares, usds, tonnes = top_n_partners(sub, n=TOP_N)
    top3_share = float(sum(shares))
    rows.append({
        'importer': imp,
        'total_ag_imports_usd': total_usd,
        'top1_partner': partners[0],
        'top1_share': shares[0],
        'top1_usd': usds[0],
        'top1_tonnes': tonnes[0],
        'top2_partner': partners[1],
        'top2_share': shares[1],
        'top2_usd': usds[1],
        'top2_tonnes': tonnes[1],
        'top3_partner': partners[2],
        'top3_share': shares[2],
        'top3_usd': usds[2],
        'top3_tonnes': tonnes[2],
        'top3_share_sum': top3_share,
        'dependent_on_top3': top3_share >= DEPENDENCY_THRESHOLD_TOP3
    })

dep_df = pd.DataFrame(rows)

pop_col_candidates = [c for c in df_int.columns if 'pop' in c.lower() and 'year' not in c.lower()]
pop_col = pop_col_candidates[0] if pop_col_candidates else None
df_int_latest = df_int.sort_values(['country','year']).groupby('country').tail(1).set_index('country')
if pop_col and pop_col in df_int_latest.columns:
    df_int_latest['population'] = pd.to_numeric(df_int_latest[pop_col], errors='coerce')
else:
    df_int_latest['population'] = np.nan

prod_df = df_crop.copy()
if 'Area' in prod_df.columns and 'Element' in prod_df.columns and 'Year' in prod_df.columns and 'Value' in prod_df.columns:
    prod_df['Element'] = prod_df['Element'].astype(str)
    prod_df = prod_df[prod_df['Element'].str.contains('Production', case=False, na=False)]
    prod_df['Year'] = pd.to_numeric(prod_df['Year'], errors='coerce').astype('Int64')
    prod_latest = prod_df.groupby('Area').apply(lambda g: g.sort_values('Year', ascending=False).head(3)['Value'].mean() if not g.empty else np.nan).reset_index().rename(columns={0:'prod_avg_tonnes', 'Area':'country'})
    prod_latest['country'] = prod_latest['country'].astype(str).str.strip()
    prod_latest['prod_avg_tonnes'] = pd.to_numeric(prod_latest['prod_avg_tonnes'], errors='coerce').fillna(0.0)
else:
    prod_latest = pd.DataFrame(columns=['country','prod_avg_tonnes'])

prod_latest = prod_latest.rename(columns={'Area':'country'})

dep_df = dep_df.merge(df_int_latest[['population']], left_on='importer', right_index=True, how='left')
dep_df = dep_df.merge(prod_latest[['country','prod_avg_tonnes']].rename(columns={'country':'importer'}), on='importer', how='left')

dep_df['imports_tonnes_est'] = dep_df['total_ag_imports_usd'] / PRICE_PER_TONNE_USD
dep_df['top1_tonnes_est'] = dep_df['top1_usd'] / PRICE_PER_TONNE_USD
dep_df['top2_tonnes_est'] = dep_df['top2_usd'] / PRICE_PER_TONNE_USD
dep_df['top3_tonnes_est'] = dep_df['top3_usd'] / PRICE_PER_TONNE_USD

dep_df['annual_consumption_tonnes_est'] = dep_df['population'] * (PER_CAPITA_FOOD_KG_PER_YEAR / 1000.0)
dep_df['domestic_prod_tonnes'] = dep_df['prod_avg_tonnes'].fillna(0.0)
dep_df['coverage_before'] = (dep_df['domestic_prod_tonnes'] + dep_df['imports_tonnes_est']) / dep_df['annual_consumption_tonnes_est']
dep_df['coverage_before'] = dep_df['coverage_before'].replace([np.inf, -np.inf], np.nan)

def simulate_ban(row, ban_list=['top1']):
    dom = float(row['domestic_prod_tonnes'] or 0.0)
    imports = float(row['imports_tonnes_est'] or 0.0)
    top1 = float(row['top1_tonnes_est'] or 0.0)
    top2 = float(row['top2_tonnes_est'] or 0.0)
    top3 = float(row['top3_tonnes_est'] or 0.0)
    lost = 0.0
    if 'top1' in ban_list:
        lost += top1
    if 'top2' in ban_list:
        lost += top2
    if 'top3' in ban_list:
        lost += top3
    remaining_imports = max(0.0, imports - lost)
    denom = row['annual_consumption_tonnes_est'] if (row['annual_consumption_tonnes_est'] and row['annual_consumption_tonnes_est']>0) else np.nan
    coverage_after = (dom + remaining_imports) / denom if not np.isnan(denom) else np.nan
    coverage_drop_pct = (row['coverage_before'] - coverage_after) / row['coverage_before'] * 100 if (row['coverage_before'] and not np.isnan(row['coverage_before'])) else np.nan
    return lost, lost * PRICE_PER_TONNE_USD, coverage_after, coverage_drop_pct

sims = []
for _, r in dep_df.iterrows():
    lost1, lost1usd, cov1, drop1 = simulate_ban(r, ['top1'])
    lost2, lost2usd, cov2, drop2 = simulate_ban(r, ['top1','top2'])
    lost3, lost3usd, cov3, drop3 = simulate_ban(r, ['top1','top2','top3'])
    sims.append({
        'importer': r['importer'],
        'total_ag_imports_usd': r['total_ag_imports_usd'],
        'top3_share_sum': r['top3_share_sum'],
        'dependent_on_top3': r['dependent_on_top3'],
        'top1_partner': r['top1_partner'],
        'top2_partner': r['top2_partner'],
        'top3_partner': r['top3_partner'],
        'lost_usd_top1': lost1usd,
        'lost_usd_top2': lost2usd,
        'lost_usd_top3': lost3usd,
        'coverage_before': r['coverage_before'],
        'coverage_after_top1': cov1,
        'coverage_after_top2': cov2,
        'coverage_after_top3': cov3,
        'coverage_drop_pct_top1': drop1,
        'coverage_drop_pct_top2': drop2,
        'coverage_drop_pct_top3': drop3
    })

sim_df = pd.DataFrame(sims)
sim_df = sim_df.sort_values(['dependent_on_top3','top3_share_sum','lost_usd_top3'], ascending=[False, False, False])
sim_df.to_csv(OUT_CSV, index=False)

print("Results saved to", OUT_CSV)
print("\nTop dependent importers (by top3 share):")
print(sim_df[['importer','top3_share_sum','dependent_on_top3','top1_partner','top2_partner','top3_partner']].head(20).to_string(index=False))
print("\nSample simulation columns (importer, lost_usd_top3, coverage_before, coverage_after_top3):")
print(sim_df[['importer','lost_usd_top3','coverage_before','coverage_after_top3']].head(20).to_string(index=False))
