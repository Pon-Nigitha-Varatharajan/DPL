import pandas as pd
import numpy as np
import networkx as nx
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
OUTFILE = os.path.join(BASE_PATH, "question7_trade_network.csv")
PLOT_FILE = os.path.join(BASE_PATH, "question7_trade_network.png")
PLOT_DISRUPTED_FILE = os.path.join(BASE_PATH, "question7_trade_network_disrupted.png")

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

# Build trade network
print(f"Building trade network for year {latest_year}...")
edges = []
for _, row in df_exp.iterrows():
    edges.append({'country': row['country'], 'partner_country': row['partner_country'], 'value': row['export_value_usd'], 'direction': 'export'})
for _, row in df_imp.iterrows():
    edges.append({'country': row['country'], 'partner_country': row['partner_country'], 'value': row['import_value_usd'], 'direction': 'import'})
edge_df = pd.DataFrame(edges)
edge_df['pair'] = edge_df.apply(lambda r: tuple(sorted([r['country'], r['partner_country']])), axis=1)
edge_df = edge_df.groupby('pair')['value'].sum().reset_index()
edge_df[['country_a', 'country_b']] = pd.DataFrame(edge_df['pair'].tolist(), index=edge_df.index)
edge_df = edge_df.rename(columns={'value': 'trade_value_usd'})

# Create graph
G = nx.Graph()
for _, row in edge_df.iterrows():
    if row['country_a'] != row['country_b']:
        G.add_edge(row['country_a'], row['country_b'], weight=row['trade_value_usd'])

# Compute centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

# Aggregate centrality scores (normalize and average)
centrality_df = pd.DataFrame({
    'country': list(degree_centrality.keys()),
    'degree_centrality': list(degree_centrality.values()),
    'betweenness_centrality': list(betweenness_centrality.values()),
    'eigenvector_centrality': list(eigenvector_centrality.values())
})
centrality_df['centrality_score'] = (
    (centrality_df['degree_centrality'] / centrality_df['degree_centrality'].max()) +
    (centrality_df['betweenness_centrality'] / centrality_df['betweenness_centrality'].max()) +
    (centrality_df['eigenvector_centrality'] / centrality_df['eigenvector_centrality'].max())
) / 3
most_central = centrality_df.sort_values('centrality_score', ascending=False).iloc[0]['country']
print(f"Most central country: {most_central}")

# Simulate disruption
G_disrupted = G.copy()
G_disrupted.remove_node(most_central)
disrupted_degree = nx.degree_centrality(G_disrupted)
disrupted_betweenness = nx.betweenness_centrality(G_disrupted, weight='weight')
disrupted_eigenvector = nx.eigenvector_centrality(G_disrupted, weight='weight', max_iter=1000)

# Estimate GDP impact
results = []
for country in COUNTRIES:
    if country == most_central:
        continue
    trade_lost = edge_df[(edge_df['country_a'] == country) & (edge_df['country_b'] == most_central) | 
                         (edge_df['country_b'] == country) & (edge_df['country_a'] == most_central)]['trade_value_usd'].sum()
    gdp = df_integrated[df_integrated['country'] == country]['gdp_current_usd'].iloc[0] if not df_integrated[df_integrated['country'] == country].empty else np.nan
    trade_gdp_pct = df_integrated[df_integrated['country'] == country]['trade_gdp_pct'].iloc[0] if not df_integrated[df_integrated['country'] == country].empty else np.nan
    if pd.isna(gdp) or pd.isna(trade_gdp_pct):
        gdp_loss_pct = np.nan
    else:
        trade_dependency = trade_lost / (gdp * (trade_gdp_pct / 100))
        gdp_loss_pct = trade_dependency * 0.5  # Assume 50% of lost trade impacts GDP
    results.append({
        'country': country,
        'trade_lost_usd': trade_lost,
        'gdp_loss_pct': gdp_loss_pct,
        'degree_centrality_disrupted': disrupted_degree.get(country, 0),
        'betweenness_centrality_disrupted': disrupted_betweenness.get(country, 0),
        'eigenvector_centrality_disrupted': disrupted_eigenvector.get(country, 0)
    })

res_df = pd.DataFrame(results)
res_df = res_df.merge(centrality_df, on='country', how='left')
res_df.to_csv(OUTFILE, index=False)
print(f"Saved results to {OUTFILE}")

# Visualization
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.35, seed=42)
node_sizes = [300 + 1700 * degree_centrality[node] for node in G.nodes()]
edge_widths = [0.5 + 4.5 * (G[u][v]['weight'] / edge_df['trade_value_usd'].max()) for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title(f"Trade Network Graph ({latest_year})")
plt.axis("off")
plt.savefig(PLOT_FILE, dpi=250)
plt.close()

# Disrupted network visualization
plt.figure(figsize=(12, 10))
pos_disrupted = nx.spring_layout(G_disrupted, k=0.35, seed=42)
node_sizes_disrupted = [300 + 1700 * disrupted_degree.get(node, 0) for node in G_disrupted.nodes()]
edge_widths_disrupted = [0.5 + 4.5 * (G_disrupted[u][v]['weight'] / edge_df['trade_value_usd'].max()) for u, v in G_disrupted.edges()]
nx.draw_networkx_nodes(G_disrupted, pos_disrupted, node_size=node_sizes_disrupted, alpha=0.9)
nx.draw_networkx_edges(G_disrupted, pos_disrupted, width=edge_widths_disrupted, alpha=0.4)
nx.draw_networkx_labels(G_disrupted, pos_disrupted, font_size=8)
plt.title(f"Trade Network Graph After {most_central} Disruption ({latest_year})")
plt.axis("off")
plt.savefig(PLOT_DISRUPTED_FILE, dpi=250)
plt.close()

print(f"Saved trade network plots to {PLOT_FILE} and {PLOT_DISRUPTED_FILE}")
print("\nTop 5 most central countries:")
print(centrality_df.sort_values('centrality_score', ascending=False)[['country', 'centrality_score']].head(5).to_string(index=False))
print("\nTop 5 countries impacted by disruption:")
print(res_df.sort_values('gdp_loss_pct', ascending=False)[['country', 'trade_lost_usd', 'gdp_loss_pct']].head(5).to_string(index=False))

