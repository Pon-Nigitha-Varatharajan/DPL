import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Files
BASE_PATH = "C:/Users/ADMIN/Desktop/DPL_2025/DPL/Datasets"  # Your provided path
INTEGRATED_FILE = os.path.join(BASE_PATH, "integrated_country_year.csv")
EMPLOYMENT_FILE = os.path.join(BASE_PATH, "Employment_Unemployment.csv")
OUTFILE = os.path.join(BASE_PATH, "question6_ageing_demographics_risk.csv")
PLOT_FILE = os.path.join(BASE_PATH, "question6_ageing_impact_plot.png")

# Countries
COUNTRIES = [
    'India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 'Argentina',
    'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 'Pakistan', 'Afghanistan',
    'Israel', 'Iran', 'Iraq', 'Bangladesh', 'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia'
]

# Hardcoded median ages for 2025 (based on CIA World Factbook 2024 estimates, adjusted)
MEDIAN_AGES_2025 = {
    'India': 29.8, 'USA': 38.9, 'Russia': 41.9, 'France': 42.6, 'Germany': 46.8,
    'Italy': 48.4, 'China': 40.2, 'Japan': 49.9, 'Argentina': 33.3, 'Portugal': 46.4,
    'Spain': 46.8, 'Croatia': 45.1, 'Belgium': 42.0, 'Australia': 38.1, 'Pakistan': 22.9,
    'Afghanistan': 20.0, 'Israel': 30.1, 'Iran': 33.8, 'Iraq': 22.4, 'Bangladesh': 29.6,
    'Sri Lanka': 34.1, 'Canada': 42.6, 'UK': 40.8, 'Sweden': 41.1, 'Saudi Arabia': 32.4
}

# Load data
if not os.path.exists(INTEGRATED_FILE):
    raise FileNotFoundError(f"{INTEGRATED_FILE} not found")
df = pd.read_csv(INTEGRATED_FILE)
df = df[df['country'].isin(COUNTRIES)]

# Load employment data
if not os.path.exists(EMPLOYMENT_FILE):
    print(f"Warning: {EMPLOYMENT_FILE} not found, proceeding without employment data")
    df_emp = pd.DataFrame()
else:
    df_emp = pd.read_csv(EMPLOYMENT_FILE)

# Get latest year data
df['year'] = pd.to_numeric(df['year'], errors='coerce')
latest = df.loc[df.groupby('country')['year'].idxmax()].copy()

# Add median age
latest['median_age'] = latest['country'].map(MEDIAN_AGES_2025)
latest = latest.dropna(subset=['median_age'])

# Ensure required columns
required_cols = ['country', 'exports_goods_services_gdp_pct', 'gdp_current_usd']
for col in required_cols:
    if col not in latest.columns:
        raise KeyError(f"Missing column {col} in {INTEGRATED_FILE}")
latest[required_cols[1:]] = latest[required_cols[1:]].apply(pd.to_numeric, errors='coerce')

# Get labor force data if available
if not df_emp.empty:
    # Debug: Print unique series names
    print("Available Series Names in Employment_Unemployment.csv:")
    print(df_emp['Series Name'].unique())
    
    # Melt employment data
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    value_vars = [col for col in df_emp.columns if re.match(r'.*\d{4}.*', col)]  # Match year columns
    if not value_vars:
        print("Warning: No year columns found in Employment_Unemployment.csv")
        df_emp = pd.DataFrame()
    else:
        emp_melt = pd.melt(df_emp, id_vars=id_vars, value_vars=value_vars,
                           var_name='year_str', value_name='value')
        emp_melt['year'] = emp_melt['year_str'].str.extract(r'(\d{4})').astype(float).astype('Int64', errors='ignore')
        emp_melt['value'] = pd.to_numeric(emp_melt['value'], errors='coerce')
        
        # Flexible regex for employment-to-population ratio
        emp_melt = emp_melt[emp_melt['Series Name'].str.contains('Employment to population ratio.*15.*total', case=False, na=False, regex=True)]
        if emp_melt.empty:
            print("Warning: No 'Employment to population ratio, 15+, total' series found")
            df_emp = pd.DataFrame()
        else:
            emp_pivot = emp_melt.pivot_table(index=['Country Name', 'year'], columns='Series Name', values='value').reset_index()
            emp_pivot = emp_pivot.rename(columns={'Country Name': 'country'})
            latest_emp = emp_pivot.loc[emp_pivot.groupby('country')['year'].idxmax()]
            latest = latest.merge(latest_emp, on='country', how='left')
else:
    latest['employment_ratio'] = np.nan

# Rename employment column if found, or set to NaN
emp_columns = [col for col in latest.columns if 'Employment to population ratio' in col]
if emp_columns:
    latest['employment_ratio'] = latest[emp_columns[0]]
    print(f"Using employment column: {emp_columns[0]}")
else:
    print("Warning: Employment ratio column not found, using default value")
    latest['employment_ratio'] = np.nan

# Productivity impact: 0.5% drop per year of age increase, weighted by labor force participation
def productivity_drop(median_age_increase, labor_ratio):
    base_drop = median_age_increase * 0.005  # 0.5% per year (based on economic literature)
    if not pd.isna(labor_ratio):
        return base_drop * (labor_ratio / 50)  # Normalize around global average ~50%
    return base_drop

# Simulate impacts
results = []
missing_data_countries = []
for _, row in latest.iterrows():
    country = row['country']
    current_median = row['median_age']
    exports_pct = row['exports_goods_services_gdp_pct']
    gdp = row['gdp_current_usd']
    labor_ratio = row.get('employment_ratio', np.nan)
    
    if pd.isna(exports_pct) or pd.isna(gdp):
        missing_data_countries.append(country)
        continue
    
    for inc in [5, 10, 15]:
        new_median = current_median + inc
        drop_frac = productivity_drop(inc, labor_ratio)
        export_impact_pct = drop_frac * exports_pct
        gdp_loss_usd = (export_impact_pct / 100) * gdp
        results.append({
            'country': country,
            'current_median_age': current_median,
            'increased_median_age': new_median,
            'median_age_increase': inc,
            'productivity_drop_frac': drop_frac,
            'export_impact_pct_gdp': export_impact_pct,
            'gdp_loss_usd': gdp_loss_usd,
            'labor_ratio': labor_ratio
        })

if missing_data_countries:
    print(f"Warning: Missing data for countries: {', '.join(missing_data_countries)}")

res_df = pd.DataFrame(results)
res_df = res_df.dropna(subset=['export_impact_pct_gdp'])

# Visualization: Bar plot for +15 years scenario
plt.figure(figsize=(12, 6))
top_15 = res_df[res_df['median_age_increase'] == 15].sort_values('export_impact_pct_gdp', ascending=False).head(10)
sns.barplot(data=top_15, x='export_impact_pct_gdp', y='country', palette='viridis')
plt.title('Top 10 Countries by Export Sector Impact from Ageing (+15 Years)')
plt.xlabel('Export Impact (% of GDP)')
plt.ylabel('Country')
plt.savefig(PLOT_FILE)
plt.close()
print(f"Saved bar plot to {PLOT_FILE}")

# Output results
print("Countries most at risk from ageing demographics (top 5 per scenario):")
for inc in [5, 10, 15]:
    print(f"\nFor +{inc} years median age increase:")
    top = res_df[res_df['median_age_increase'] == inc].sort_values('export_impact_pct_gdp', ascending=False).head(5)
    print(top[['country', 'current_median_age', 'increased_median_age', 'export_impact_pct_gdp', 'gdp_loss_usd', 'labor_ratio']].to_string(index=False))

res_df.to_csv(OUTFILE, index=False)
print(f"\nSaved results to {OUTFILE}")
