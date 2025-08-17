import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ---------------------------
# Config / Inputs
# ---------------------------
INTEGRATED = "feature_engineered_dataset.csv"            # from your feature engineering step
FORECASTS_BASELINE = "forecasts_baseline.csv"            # optional
BILATERAL_EXPORT = "bilateral_export_data.csv"           # optional
BILATERAL_IMPORT = "bilateral_import_data.csv"           # optional
OUTDIR = "viz_outputs"

os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
print("Loading datasets...")
df = pd.read_csv(INTEGRATED)
df = df.sort_values(["country", "year"])

df_fc = None
if os.path.exists(FORECASTS_BASELINE):
    df_fc = pd.read_csv(FORECASTS_BASELINE)

df_exp = pd.read_csv(BILATERAL_EXPORT) if os.path.exists(BILATERAL_EXPORT) else None
df_imp = pd.read_csv(BILATERAL_IMPORT) if os.path.exists(BILATERAL_IMPORT) else None

print(f"Main dataset: {df.shape}")
if df_fc is not None:
    print(f"Forecasts: {df_fc.shape}")
if df_exp is not None:
    print(f"Bilateral export: {df_exp.shape}")
if df_imp is not None:
    print(f"Bilateral import: {df_imp.shape}")

# ---------------------------
# Helper: safe numeric subset
# ---------------------------
def numeric_columns(frame, extra_keep=None):
    extra_keep = extra_keep or []
    cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    for c in extra_keep:
        if c in frame.columns and c not in cols:
            cols.append(c)
    return cols

# ---------------------------
# 1) Correlation Heatmaps
#    - Global
#    - Per country (last 10 years if available)
# ---------------------------
print("Building correlation heatmaps...")

# choose a set of core indicators if present
candidate_cols = [
    "gdp_growth_pct",
    "inflation_consumer_prices_pct",
    "trade_gdp_pct",
    "trade_dependency_index",
    "resilience_score",
    "spending_efficiency",
    "shock_impact_score",
    "unemployment_total_pct_labor_force",
    "poverty_headcount_ratio_3ppp_pct",
    "current_account_balance_gdp_pct",
    "external_debt_stocks_gni_pct",
    "fdi_net_inflows_gdp_pct",
]
use_cols = [c for c in candidate_cols if c in df.columns]

# Global correlation heatmap (over last 10 years to center on recent dynamics)
if "year" in df.columns:
    recent_mask = df["year"] >= max(df["year"].min(), df["year"].max() - 9)
    df_recent = df[recent_mask]
else:
    df_recent = df.copy()

corr = df_recent[use_cols].corr(min_periods=50) if len(use_cols) > 1 else None
if corr is not None and corr.shape[0] > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", annot=False, vmin=-1, vmax=1)
    plt.title("Global Correlation Heatmap (Recent Years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "heatmap_global.png"), dpi=200)
    plt.close()

# Per-country correlation heatmap (use last 10 available years)
countries = sorted(df["country"].dropna().unique().tolist())
for ctry in countries:
    sub = df[df["country"] == ctry]
    if "year" in sub.columns and sub["year"].nunique() > 3:
        sub = sub.sort_values("year").tail(10)
    if len(use_cols) > 1 and sub[use_cols].dropna(how="all").shape[0] > 3:
        cmat = sub[use_cols].corr(min_periods=5)
        if cmat.shape[0] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cmat, cmap="vlag", annot=False, vmin=-1, vmax=1)
            plt.title(f"{ctry} - Correlation Heatmap (Recent Years)")
            plt.tight_layout()
            fn = f"heatmap_{ctry.replace(' ', '_')}.png"
            plt.savefig(os.path.join(OUTDIR, fn), dpi=200)
            plt.close()

# ---------------------------
# 2) Trade Network Graph
#    - Build network with nodes = countries, edges weighted by trade value
#    - Use latest year available
# ---------------------------
def build_trade_network(df_exp, df_imp):
    if df_exp is None and df_imp is None:
        return None, None

    # unify columns
    edges = []
    latest_year = None

    if df_exp is not None and "year" in df_exp.columns:
        latest_year = int(df_exp["year"].max())
    if df_imp is not None and "year" in df_imp.columns:
        latest_year = max(latest_year or -np.inf, int(df_imp["year"].max()))
        if latest_year == -np.inf:
            latest_year = None

    if latest_year is None:
        return None, None

    if df_exp is not None:
        ex = df_exp[df_exp["year"] == latest_year].copy()
        val_col = "export_value_usd" if "export_value_usd" in ex.columns else "primaryValue"
        if val_col in ex.columns:
            ex = ex.rename(columns={val_col: "value"})
            ex = ex.groupby(["country", "partner_country"], as_index=False)["value"].sum()
            ex["direction"] = "export"
            edges.append(ex)

    if df_imp is not None:
        im = df_imp[df_imp["year"] == latest_year].copy()
        val_col = "import_value_usd" if "import_value_usd" in im.columns else "primaryValue"
        if val_col in im.columns:
            im = im.rename(columns={val_col: "value"})
            im = im.groupby(["country", "partner_country"], as_index=False)["value"].sum()
            im["direction"] = "import"
            edges.append(im)

    if not edges:
        return None, None

    all_edges = pd.concat(edges, ignore_index=True)
    # build undirected weight = exports + imports between pairs
    all_edges["pair"] = all_edges.apply(
        lambda r: tuple(sorted([str(r["country"]), str(r["partner_country"])])), axis=1
    )
    w = all_edges.groupby("pair")["value"].sum().reset_index()
    w[["a", "b"]] = pd.DataFrame(w["pair"].tolist(), index=w.index)
    w = w.rename(columns={"value": "trade_value"})
    return w, latest_year

print("Building trade network graph...")
edge_df, net_year = build_trade_network(df_exp, df_imp)
if edge_df is not None:
    G = nx.Graph()
    for _, r in edge_df.iterrows():
        a, b, val = r["a"], r["b"], r["trade_value"]
        if pd.isna(a) or pd.isna(b) or a == b:
            continue
        G.add_edge(a, b, weight=val)

    # degree-weighted sizing
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    if len(weights) > 0:
        wmin, wmax = np.percentile(weights, 10), np.percentile(weights, 90)
        def norm_w(x):
            if wmax == wmin:
                return 1.0
            return (x - wmin) / (wmax - wmin)
        pos = nx.spring_layout(G, k=0.35, seed=42)
        plt.figure(figsize=(12, 10))
        # node size by strength (sum of weights)
        strength = {n: sum([G[n][nbr]["weight"] for nbr in G[n]]) for n in G.nodes()}
        svals = np.array(list(strength.values()))
        if len(svals) > 0:
            smin, smax = np.percentile(svals, 10), np.percentile(svals, 90)
            def norm_s(x):
                if smax == smin:
                    return 1.0
                return (x - smin) / (smax - smin)
            node_sizes = [300 + 1700 * norm_s(strength[n]) for n in G.nodes()]
        else:
            node_sizes = 300

        # edge widths by trade value
        ewidths = [0.5 + 4.5 * norm_w(G[u][v]["weight"]) for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=ewidths, alpha=0.4)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title(f"Trade Network Graph (Total Bilateral Trade, {net_year})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"trade_network_{net_year}.png"), dpi=250)
        plt.close()
    else:
        print("No edges to visualize in trade network.")
else:
    print("Bilateral trade files not found or unusable. Skipping trade network graph.")

# ---------------------------
# 3) Shock Maps (per country)
#    Heatmap across years for shock indicators
# ---------------------------
print("Building shock maps...")
# pick available shock-related columns
shock_candidates = [
    "shock_impact_score",
    "num_disasters",
    "total_deaths",
    "total_affected",
    "total_homeless",
    "total_damage_adjusted_usd",
]
shock_cols = [c for c in shock_candidates if c in df.columns]
if "year" not in df.columns:
    shock_cols = []  # cannot time-index

for ctry in countries:
    sub = df[df["country"] == ctry]
    if sub.empty or not shock_cols:
        continue
    # pivot to (year x metric) heatmap
    pv = sub.pivot_table(index="year", values=shock_cols, aggfunc="mean")
    if pv.shape[0] < 3:
        continue
    # log-scale large monetary/impact cols to make heatmap readable
    pv = pv.copy()
    for col in pv.columns:
        if ("damage" in col) or ("affected" in col) or ("deaths" in col) or ("homeless" in col):
            pv[col] = np.log1p(pv[col])

    plt.figure(figsize=(10, max(4, pv.shape[1] * 0.6)))
    sns.heatmap(pv.T, cmap="rocket", cbar=True)
    plt.title(f"{ctry} - Shock Map (rows=metrics, cols=year; log1p for big values)")
    plt.xlabel("year")
    plt.ylabel("metric")
    plt.tight_layout()
    fn = f"shock_map_{ctry.replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTDIR, fn), dpi=200)
    plt.close()

# ---------------------------
# 4) Top 3 Vulnerabilities per Country
#    - Compute standardized “risk” scores and pick top 3 drivers
# ---------------------------
print("Computing top vulnerabilities...")

# Define risk metrics (higher = worse). If a "good" metric exists, invert it.
risk_defs = {
    "trade_dependency_index": +1.0,
    "shock_impact_score": +1.0,
    "num_disasters": +1.0,
    "total_damage_adjusted_usd": +1.0,
    "external_debt_stocks_gni_pct": +1.0,
    "unemployment_total_pct_labor_force": +1.0,
    "poverty_headcount_ratio_3ppp_pct": +1.0,
    "inflation_consumer_prices_pct": +1.0,
    "resilience_score": -1.0,           # negative sign means higher is better; invert for risk
    "spending_efficiency": -1.0,        # higher efficiency => lower risk
    "current_account_balance_gdp_pct": -1.0,  # surplus is good
}

available_risks = [k for k in risk_defs if k in df.columns]
if not available_risks:
    print("No risk columns available to compute vulnerabilities.")
    vuln_df = pd.DataFrame()
else:
    # use recent window (last 5 years) to compute typical levels
    def recent_window(g):
        if "year" not in g.columns:
            return g
        return g.sort_values("year").tail(5)

    recent = df.groupby("country", group_keys=False).apply(recent_window)

    # aggregate to per-country average in the window
    agg = recent.groupby("country")[available_risks].mean().reset_index()

    # z-score standardization globally
    z = agg.copy()
    for col in available_risks:
        m, s = agg[col].mean(skipna=True), agg[col].std(skipna=True)
        if pd.isna(s) or s == 0:
            z[col] = 0.0
        else:
            z[col] = (agg[col] - m) / s
        # apply direction: positive => risk higher is worse; negative => invert
        z[col] = z[col] * risk_defs[col]

    # pick top 3 per country
    rows = []
    for _, r in z.iterrows():
        ctry = r["country"]
        scores = r[available_risks].to_dict()
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        rows.append({
            "country": ctry,
            "vulnerability_1": top3[0][0] if len(top3) > 0 else None,
            "score_1": round(top3[0][1], 3) if len(top3) > 0 else None,
            "vulnerability_2": top3[1][0] if len(top3) > 1 else None,
            "score_2": round(top3[1][1], 3) if len(top3) > 1 else None,
            "vulnerability_3": top3[2][0] if len(top3) > 2 else None,
            "score_3": round(top3[2][1], 3) if len(top3) > 2 else None,
        })
    vuln_df = pd.DataFrame(rows)
    vuln_df.to_csv(os.path.join(OUTDIR, "top3_vulnerabilities_per_country.csv"), index=False)
    print(f"Saved: {os.path.join(OUTDIR, 'top3_vulnerabilities_per_country.csv')}")

print("All visualizations and insights generated in:", OUTDIR)