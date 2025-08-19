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
OUTFILE = os.path.join(BASE_PATH, "question8_trade_relationships.csv")
PLOT_FILE = os.path.join(BASE_PATH, "question8_mutual_benefit_heatmap.png")

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

# Calculate total trade value
edges = []
for _, row in df_exp.iterrows():
    edges.append({'country': row['country'], 'partner_country': row['partner_country'], 'value': row['export_value_usd']})
for _, row in df_imp.iterrows():
    edges.append({'country': row['partner_country'], 'partner_country': row['country'], 'value': row['import_value_usd']})
edge_df = pd.DataFrame(edges)
edge_df['pair'] = edge_df.apply(lambda r: tuple(sorted([r['country'], r['partner_country']])), axis=1)
edge_df = edge_df.groupby('pair')['value'].sum().reset_index()
edge_df[['country_a', 'country_b']] = pd.DataFrame(edge_df['pair'].tolist(), index=edge_df.index)
edge_df = edge_df.rename(columns={'value': 'total_trade_usd'})

# Calculate mutual benefit
results = []
for _, row in edge_df.iterrows():
    country_a = row['country_a']
    country_b = row['country_b']
    trade_value = row['total_trade_usd']
    gdp_a = df_integrated[df_integrated['country'] == country_a]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == country_a].empty else np.nan
    gdp_b = df_integrated[df_integrated['country'] == country_b]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == country_b].empty else np.nan
    trade_gdp_pct_a = df_integrated[df_integrated['country'] == country_a]['trade_gdp_pct'].iloc[0] if not df_integrated[df_integrated['country'] == country_a].empty else np.nan
    trade_gdp_pct_b = df_integrated[df_integrated['country'] == country_b]['trade_gdp_pct'].iloc[0] if not df_integrated[df_integrated['country'] == country_b].empty else np.nan
    
    if pd.isna(gdp_a) or pd.isna(gdp_b):
        mutual_benefit = np.nan
        gdp_loss_pct_a = np.nan
        gdp_loss_pct_b = np.nan
    else:
        mutual_benefit = trade_value / (gdp_a + gdp_b) * 100  # Trade as % of combined GDP
        trade_dependency_a = trade_value / (gdp_a * (trade_gdp_pct_a / 100))
        trade_dependency_b = trade_value / (gdp_b * (trade_gdp_pct_b / 100))
        gdp_loss_pct_a = trade_dependency_a * 0.5  # 50% of lost trade impacts GDP
        gdp_loss_pct_b = trade_dependency_b * 0.5
    results.append({
        'country_a': country_a,
        'country_b': country_b,
        'total_trade_usd': trade_value,
        'mutual_benefit_pct': mutual_benefit,
        'gdp_loss_pct_a': gdp_loss_pct_a,
        'gdp_loss_pct_b': gdp_loss_pct_b
    })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('mutual_benefit_pct', ascending=False)
top_pair = res_df.iloc[0][['country_a', 'country_b']].to_dict()
print(f"Top trade relationship: {top_pair['country_a']} - {top_pair['country_b']}")

# Save results
res_df.to_csv(OUTFILE, index=False)
print(f"Saved results to {OUTFILE}")

# Visualization: Heatmap of mutual benefits
pivot_df = edge_df.pivot(index='country_a', columns='country_b', values='total_trade_usd').fillna(0)
for country in COUNTRIES:
    if country not in pivot_df.index:
        pivot_df.loc[country] = 0
    if country not in pivot_df.columns:
        pivot_df[country] = 0
pivot_df = pivot_df.loc[COUNTRIES, COUNTRIES]
mutual_benefit_matrix = pivot_df.copy()
for i in mutual_benefit_matrix.index:
    for j in mutual_benefit_matrix.columns:
        if i != j:
            gdp_i = df_integrated[df_integrated['country'] == i]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == i].empty else np.nan
            gdp_j = df_integrated[df_integrated['country'] == j]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == j].empty else np.nan
            if pd.isna(gdp_i) or pd.isna(gdp_j):
                mutual_benefit_matrix.loc[i, j] = np.nan
            else:
                mutual_benefit_matrix.loc[i, j] = mutual_benefit_matrix.loc[i, j] / (gdp_i + gdp_j) * 100

plt.figure(figsize=(12, 10))
sns.heatmap(mutual_benefit_matrix, cmap='viridis', annot=False)
plt.title(f"Mutual Benefit Heatmap ({latest_year}, Trade % of Combined GDP)")
plt.xlabel("Country B")
plt.ylabel("Country A")
plt.savefig(PLOT_FILE, dpi=200)
plt.close()
print(f"Saved heatmap to {PLOT_FILE}")

print("\nTop 5 trade relationships by mutual benefit:")
print(res_df[['country_a', 'country_b', 'mutual_benefit_pct', 'gdp_loss_pct_a', 'gdp_loss_pct_b']].head(5).to_string(index=False))

