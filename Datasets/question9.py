import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from uuid import uuid4

# Config
BASE_PATH = "C:/Users/ADMIN/Desktop/DPL_2025/DPL/Datasets"
BILATERAL_EXPORT = os.path.join(BASE_PATH, "bilateral_export_data.csv")
BILATERAL_IMPORT = os.path.join(BASE_PATH, "bilateral_import_data.csv")
INTEGRATED_FILE = os.path.join(BASE_PATH, "feature_engineered_dataset.csv")
OUTDIR = os.path.join(BASE_PATH, "viz_outputs")
OUTFILE = os.path.join(BASE_PATH, "question9_trade_diversification.csv")
PLOT_FILE = os.path.join(BASE_PATH, "question9_hhi_heatmap.png")

os.makedirs(OUTDIR, exist_ok=True)

# Countries
COUNTRIES = [
    'India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 'Argentina',
    'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 'Pakistan', 'Afghanistan',
    'Israel', 'Iran', 'Iraq', 'Bangladesh', 'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia'
]

# Load data
print("Loading datasets...")
if not os.path.exists(BILATERAL_EXPORT) or not os.path.exists(BILATERAL_IMPORT):
    raise FileNotFoundError("Bilateral trade files not found")
df_exp = pd.read_csv(BILATERAL_EXPORT)
df_imp = pd.read_csv(BILATERAL_IMPORT)
df_integrated = pd.read_csv(INTEGRATED_FILE)

# Filter for target countries and latest year
df_exp = df_exp[df_exp['country'].isin(COUNTRIES) & df_exp['partner_country'].isin(COUNTRIES)]
df_imp = df_imp[df_imp['country'].isin(COUNTRIES) & df_imp['partner_country'].isin(COUNTRIES)]
latest_year = max(df_exp['year'].max(), df_imp['year'].max())
df_exp = df_exp[df_exp['year'] == latest_year]
df_imp = df_imp[df_imp['year'] == latest_year]
df_integrated = df_integrated[df_integrated['country'].isin(COUNTRIES) & (df_integrated['year'] == latest_year)]

# Calculate HHI for exports and imports
def calculate_hhi(df, value_col, group_col, partner_col):
    total = df.groupby(group_col)[value_col].sum().reset_index(name='total_value')
    df = df.merge(total, on=group_col)
    df['share'] = df[value_col] / df['total_value']
    df['share_squared'] = df['share'] ** 2
    hhi = df.groupby(group_col)['share_squared'].sum().reset_index(name='hhi')
    return hhi

exp_hhi = calculate_hhi(df_exp, 'export_value_usd', 'country', 'partner_country')
imp_hhi = calculate_hhi(df_imp, 'import_value_usd', 'country', 'partner_country')
hhi_df = exp_hhi.merge(imp_hhi, on='country', suffixes=('_export', '_import'))
hhi_df['avg_hhi'] = (hhi_df['hhi_export'] + hhi_df['hhi_import']) / 2

# Recommend new trade partners
results = []
for country in COUNTRIES:
    current_partners_exp = df_exp[df_exp['country'] == country][['partner_country', 'export_value_usd']].sort_values('export_value_usd', ascending=False)
    current_partners_imp = df_imp[df_imp['country'] == country][['partner_country', 'import_value_usd']].sort_values('import_value_usd', ascending=False)
    current_partners = set(current_partners_exp.head(5)['partner_country']).union(current_partners_imp.head(5)['partner_country'])
    
    potential_partners = [c for c in COUNTRIES if c not in current_partners and c != country]
    partner_scores = []
    for partner in potential_partners:
        gdp_partner = df_integrated[df_integrated['country'] == partner]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == partner].empty else np.nan
        trade_gdp_pct_partner = df_integrated[df_integrated['country'] == partner]['trade_gdp_pct'].iloc[0] if not df_integrated[df_integrated['country'] == partner].empty else np.nan
        if pd.isna(gdp_partner) or pd.isna(trade_gdp_pct_partner):
            score = 0
        else:
            score = gdp_partner * (trade_gdp_pct_partner / 100)  # Score by economic size and trade openness
        partner_scores.append({'partner': partner, 'score': score})
    
    partner_scores = sorted(partner_scores, key=lambda x: x['score'], reverse=True)[:3]
    results.append({
        'country': country,
        'hhi_export': hhi_df[hhi_df['country'] == country]['hhi_export'].iloc[0] if not hhi_df[hhi_df['country'] == country].empty else np.nan,
        'hhi_import': hhi_df[hhi_df['country'] == country]['hhi_import'].iloc[0] if not hhi_df[hhi_df['country'] == country].empty else np.nan,
        'avg_hhi': hhi_df[hhi_df['country'] == country]['avg_hhi'].iloc[0] if not hhi_df[hhi_df['country'] == country].empty else np.nan,
        'new_partner_1': partner_scores[0]['partner'] if len(partner_scores) > 0 else None,
        'new_partner_2': partner_scores[1]['partner'] if len(partner_scores) > 1 else None,
        'new_partner_3': partner_scores[2]['partner'] if len(partner_scores) > 2 else None
    })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('avg_hhi', ascending=False)
res_df.to_csv(OUTFILE, index=False)
print(f"Saved results to {OUTFILE}")

# Visualization: Heatmap of HHI scores
hhi_pivot = df_integrated.pivot_table(index='country', columns='year', values='trade_dependency_index').fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(hhi_pivot.loc[COUNTRIES], cmap='viridis', annot=False)
plt.title(f"Trade Dependency Index Heatmap (2000-{latest_year})")
plt.xlabel("Year")
plt.ylabel("Country")
plt.savefig(PLOT_FILE, dpi=200)
plt.close()
print(f"Saved heatmap to {PLOT_FILE}")

print("\nTop 5 countries by trade concentration (HHI):")
print(res_df[['country', 'avg_hhi', 'new_partner_1', 'new_partner_2', 'new_partner_3']].head(5).to_string(index=False))