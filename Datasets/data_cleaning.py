import pandas as pd
import numpy as np
import os
from uuid import uuid4

# Base directory for datasets
BASE_PATH = '/Users/ponnigithav/Desktop/DPL_3rd_Edition/Datasets'

# Step 1: Setup and Standardization
print("Task 1.1: Defining target countries")
countries = [
    'India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 'Argentina',
    'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 'Pakistan', 'Afghanistan',
    'Israel', 'Iran', 'Iraq', 'Bangladesh', 'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia'
]
print(f"Target countries ({len(countries)}):", countries)

print("\nTask 1.2: Setting up country name standardization")
iso_to_country = {
    'IND': 'India', 'USA': 'USA', 'RUS': 'Russia', 'FRA': 'France', 'DEU': 'Germany',
    'ITA': 'Italy', 'CHN': 'China', 'JPN': 'Japan', 'ARG': 'Argentina', 'PRT': 'Portugal',
    'ESP': 'Spain', 'HRV': 'Croatia', 'BEL': 'Belgium', 'AUS': 'Australia', 'PAK': 'Pakistan',
    'AFG': 'Afghanistan', 'ISR': 'Israel', 'IRN': 'Iran', 'IRQ': 'Iraq', 'BGD': 'Bangladesh',
    'LKA': 'Sri Lanka', 'CAN': 'Canada', 'GBR': 'UK', 'SWE': 'Sweden', 'SAU': 'Saudi Arabia'
}

def standardize_country(name, iso=None):
    if iso and iso in iso_to_country:
        return iso_to_country[iso]
    if isinstance(name, str):
        name = name.strip()
        mapping = {
            'United States': 'USA', 'United States of America': 'USA',
            'Russian Federation': 'Russia', 'United Kingdom': 'UK',
            'United Kingdom of Great Britain and Northern Ireland': 'UK',
            'Iran (Islamic Republic of)': 'Iran', 'Islamic Republic of Iran': 'Iran',
            'People\'s Republic of China': 'China', 'China, People\'s Republic of': 'China',
            'Argentine Republic': 'Argentina'
        }
        return mapping.get(name, name)
    return name

print("\nTask 1.3: Preparing environment")
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

print("\nTask 1.4: Defining year range")
year_range = (2000, 2024)
print(f"Year range for filtering: {year_range[0]}–{year_range[1]}")

def melt_wide_df(df, id_vars, value_name='value'):
    year_cols = [col for col in df.columns if any(y in col for y in ['YR', '19', '20']) or col.isdigit()]
    if not year_cols:
        raise ValueError("No year columns found in DataFrame")
    df_melted = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name='year_str', value_name=value_name)
    df_melted['year'] = df_melted['year_str'].str.extract(r'(\d{4})').astype(float).astype('Int64')
    df_melted[value_name] = pd.to_numeric(df_melted[value_name], errors='coerce')
    df_melted.drop(columns=['year_str'], inplace=True)
    return df_melted

# Step 2: Process Each Dataset
print("\nProcessing Core_economic_indicators.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'Core_economic_indicators.csv'))
    df_core = pd.read_csv(os.path.join(BASE_PATH, 'Core_economic_indicators.csv'))
    df_core['Country Name'] = df_core.apply(lambda x: standardize_country(x['Country Name'], x.get('Country Code')), axis=1)
    df_core = df_core[df_core['Country Name'].isin(countries)]
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    df_core_melt = melt_wide_df(df_core, id_vars=id_vars)
    df_core_pivot = df_core_melt.pivot_table(index=['Country Name', 'year'], columns='Series Name', values='value')
    df_core_pivot.reset_index(inplace=True)
    df_core_pivot.rename(columns={
        'Country Name': 'country',
        'Imports of goods and services (% of GDP)': 'imports_goods_services_gdp_pct',
        'Exports of goods and services (% of GDP)': 'exports_goods_services_gdp_pct',
        'Trade (% of GDP)': 'trade_gdp_pct',
        'Inflation, consumer prices (annual %)': 'inflation_consumer_prices_pct',
        'GDP growth (annual %)': 'gdp_growth_pct',
        'GDP per capita (current US$)': 'gdp_per_capita_current_usd',
        'GDP (current US$)': 'gdp_current_usd'
    }, inplace=True)
    df_core_pivot.set_index(['country', 'year'], inplace=True)
    print(f"Core_economic_indicators shape: {df_core_pivot.shape}")
except Exception as e:
    print(f"Error in Core_economic_indicators: {e}")
    df_core_pivot = pd.DataFrame()

print("\nProcessing Resiliance.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'Resiliance.csv'))
    df_res = pd.read_csv(os.path.join(BASE_PATH, 'Resiliance.csv'))
    df_res['Country Name'] = df_res.apply(lambda x: standardize_country(x['Country Name'], x.get('Country Code')), axis=1)
    df_res = df_res[df_res['Country Name'].isin(countries)]
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    df_res_melt = melt_wide_df(df_res, id_vars=id_vars)
    df_res_pivot = df_res_melt.pivot_table(index=['Country Name', 'year'], columns='Series Name', values='value')
    df_res_pivot.reset_index(inplace=True)
    df_res_pivot.rename(columns={
        'Country Name': 'country',
        'Current account balance (% of GDP)': 'current_account_balance_gdp_pct',
        'External debt stocks (% of GNI)': 'external_debt_stocks_gni_pct',
        'Foreign direct investment, net inflows (% of GDP)': 'fdi_net_inflows_gdp_pct'
    }, inplace=True)
    df_res_pivot.set_index(['country', 'year'], inplace=True)
    print(f"Resiliance shape: {df_res_pivot.shape}")
except Exception as e:
    print(f"Error in Resiliance: {e}")
    df_res_pivot = pd.DataFrame()

print("\nProcessing Social_and_welfare.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'Social_and_welfare.csv'))
    df_social = pd.read_csv(os.path.join(BASE_PATH, 'Social_and_welfare.csv'))
    df_social['Country Name'] = df_social.apply(lambda x: standardize_country(x['Country Name'], x.get('Country Code')), axis=1)
    df_social = df_social[df_social['Country Name'].isin(countries)]
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    df_social_melt = melt_wide_df(df_social, id_vars=id_vars)
    df_social_pivot = df_social_melt.pivot_table(index=['Country Name', 'year'], columns='Series Name', values='value')
    df_social_pivot.reset_index(inplace=True)
    df_social_pivot.rename(columns={
        'Country Name': 'country',
        'Urban population (% of total population)': 'urban_population_pct_total',
        'Population growth (annual %)': 'population_growth_annual_pct',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'unemployment_total_pct_labor_force_social',
        'Life expectancy at birth, total (years)': 'life_expectancy_birth_total_yrs',
        'Life expectancy at birth, female (years)': 'life_expectancy_birth_female_yrs',
        'Life expectancy at birth, male (years)': 'life_expectancy_birth_male_yrs',
        'GINI index (World Bank estimate)': 'Gini index',
        'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)': 'poverty_headcount_ratio_3ppp_pct'
    }, inplace=True)
    df_social_pivot.set_index(['country', 'year'], inplace=True)
    print(f"Social_and_welfare shape: {df_social_pivot.shape}")
except Exception as e:
    print(f"Error in Social_and_welfare: {e}")
    df_social_pivot = pd.DataFrame()

print("\nProcessing Export Data (2000-2012_Export.csv and 2013-2024_Export.csv)...")
df_exp_agg = pd.DataFrame()  # Initialize to avoid NameError
df_exp_bilateral = pd.DataFrame()
try:
    check_file_exists(os.path.join(BASE_PATH, '2000-2012_Export.csv'))
    check_file_exists(os.path.join(BASE_PATH, '2013-2024_Export.csv'))
    df_exp1 = pd.read_csv(os.path.join(BASE_PATH, '2000-2012_Export.csv'), encoding='latin1')
    df_exp2 = pd.read_csv(os.path.join(BASE_PATH, '2013-2024_Export.csv'), encoding='latin1')
    df_exp = pd.concat([df_exp1, df_exp2], ignore_index=True)
    
    print(f"Export data columns: {list(df_exp.columns)}")
    print(f"Unique reporterISO: {sorted(df_exp['reporterISO'].unique())}")
    print("Unique flowCode:", df_exp['flowCode'].unique())
    print("Unique partnerDesc:", df_exp['partnerDesc'].unique())
    print("Unique cmdDesc:", df_exp['cmdDesc'].unique())
    print("Potential year columns:")
    for col in ['refYear', 'period', 'refPeriodId']:
        if col in df_exp.columns:
            print(f"Unique {col}: {sorted(df_exp[col].dropna().unique())}")
    
    # Try different year columns
    year_col = None
    for col in ['period', 'refPeriodId', 'refYear']:
        if col in df_exp.columns:
            df_exp[col] = pd.to_numeric(df_exp[col], errors='coerce')
            unique_years = df_exp[col].dropna().unique()
            if any(2000 <= y <= 2024 for y in unique_years):
                year_col = col
                break
    if year_col is None:
        print("Warning: No valid year column found with values in 2000–2024")
        year_col = 'refYear'  # Fallback to refYear
        df_exp[year_col] = df_exp[year_col].apply(lambda x: x + 1900 if pd.notna(x) and x < 100 else x)
    
    df_exp = df_exp[df_exp[year_col].between(2000, 2024)]
    print(f"Rows after year filter using {year_col}: {len(df_exp)}")
    
    # Convert netWgt and primaryValue to numeric
    df_exp['netWgt'] = pd.to_numeric(df_exp['netWgt'], errors='coerce')
    df_exp['primaryValue'] = pd.to_numeric(df_exp['primaryValue'], errors='coerce')
    
    print("Sample rows (first 5):")
    print(df_exp[['reporterISO', year_col, 'flowCode', 'partnerDesc', 'cmdDesc', 'primaryValue', 'netWgt']].head(5))
    
    # Standardize using reporterISO as country name
    df_exp['country'] = df_exp['reporterISO'].apply(standardize_country)
    df_exp['partner_country'] = df_exp['partnerISO'].apply(standardize_country)
    df_exp = df_exp[df_exp['country'].isin(countries)]
    print(f"Rows after country filter: {len(df_exp)}")
    print(f"Unique countries after standardization: {sorted(df_exp['country'].dropna().unique())}")
    
    # Validate data availability
    print(f"Rows with non-null primaryValue: {len(df_exp[df_exp['primaryValue'].notna()])}")
    print(f"Rows with non-null netWgt: {len(df_exp[df_exp['netWgt'].notna()])}")
    print(f"Rows with positive primaryValue: {len(df_exp[df_exp['primaryValue'] > 0])}")
    print(f"Rows with positive netWgt: {len(df_exp[df_exp['netWgt'] > 0])}")
    
    # Filter for Export flow, World partner, and All Commodities
    df_exp_total = df_exp[(df_exp['flowCode'].str.strip().eq('Export')) &
                      (df_exp['partnerDesc'].str.strip().eq('World')) &
                      (df_exp['cmdDesc'].str.strip().eq('All Commodities'))]
    print(f"Rows after World/All Commodities/Export filter: {len(df_exp_total)}")
    
    if df_exp_total.empty:
        print("Warning: Export DataFrame is empty after filtering. Check filter values or data availability.")
        print("Unique flowCode values:", df_exp['flowCode'].unique())
        print("Unique partnerDesc values:", df_exp['partnerDesc'].unique())
        print("Unique cmdDesc values:", df_exp['cmdDesc'].unique())
    else:
        # Aggregate by country and year
        df_exp_agg = df_exp_total.groupby(['country', year_col]).agg({
            'primaryValue': 'sum',
            'netWgt': 'sum'
        }).reset_index()
        df_exp_agg['netWgt'] = df_exp_agg['netWgt'] / 1000  # kg to tonnes
        df_exp_agg.rename(columns={
            year_col: 'year',
            'primaryValue': 'total_exports_usd',
            'netWgt': 'total_exports_tonnes'
        }, inplace=True)
        df_exp_agg = df_exp_agg[df_exp_agg['year'].between(2000, 2024)]
        print("Aggregated export data sample:")
        print(df_exp_agg.head(5))
        df_exp_agg.set_index(['country', 'year'], inplace=True)
        print(f"Export Data shape: {df_exp_agg.shape}")
    
    # Create bilateral trade dataset
    df_exp_bilateral = df_exp[df_exp['partner_country'].isin(countries) & (df_exp['flowCode'] == 'Export')]
    df_exp_bilateral = df_exp_bilateral.groupby(['country', 'partner_country', year_col]).agg({
        'primaryValue': 'sum',
        'netWgt': 'sum'
    }).reset_index()
    df_exp_bilateral['netWgt'] = df_exp_bilateral['netWgt'] / 1000  # kg to tonnes
    df_exp_bilateral.rename(columns={
        year_col: 'year',
        'primaryValue': 'export_value_usd',
        'netWgt': 'export_weight_tonnes'
    }, inplace=True)
    df_exp_bilateral = df_exp_bilateral[df_exp_bilateral['year'].between(2000, 2024)]
    print("Bilateral export data sample:")
    print(df_exp_bilateral.head(5))
    df_exp_bilateral.to_csv(os.path.join(BASE_PATH, 'bilateral_export_data.csv'), index=False)
    print(f"Bilateral export data saved with shape: {df_exp_bilateral.shape}")
except Exception as e:
    print(f"Error in Export Data: {e}")
    df_exp_agg = pd.DataFrame()
    df_exp_bilateral = pd.DataFrame()

print("\nProcessing Import Data (2000-2012_Import.csv and 2013-2024_Import.csv)...")
df_imp_agg = pd.DataFrame()  # Initialize to avoid NameError
df_imp_bilateral = pd.DataFrame()
try:
    check_file_exists(os.path.join(BASE_PATH, '2000-2012_Import.csv'))
    check_file_exists(os.path.join(BASE_PATH, '2013-2024_Import.csv'))
    df_imp1 = pd.read_csv(os.path.join(BASE_PATH, '2000-2012_Import.csv'), encoding='latin1')
    df_imp2 = pd.read_csv(os.path.join(BASE_PATH, '2013-2024_Import.csv'), encoding='latin1')
    df_imp = pd.concat([df_imp1, df_imp2], ignore_index=True)
    
    print(f"Import data columns: {list(df_imp.columns)}")
    print(f"Unique reporterISO: {sorted(df_imp['reporterISO'].unique())}")
    print("Unique flowCode:", df_imp['flowCode'].unique())
    print("Unique partnerDesc:", df_imp['partnerDesc'].unique())
    print("Unique cmdDesc:", df_imp['cmdDesc'].unique())
    print("Potential year columns:")
    for col in ['refYear', 'period', 'refPeriodId']:
        if col in df_imp.columns:
            print(f"Unique {col}: {sorted(df_imp[col].dropna().unique())}")
    
    # Try different year columns
    year_col = None
    for col in ['period', 'refPeriodId', 'refYear']:
        if col in df_imp.columns:
            df_imp[col] = pd.to_numeric(df_imp[col], errors='coerce')
            unique_years = df_imp[col].dropna().unique()
            if any(2000 <= y <= 2024 for y in unique_years):
                year_col = col
                break
    if year_col is None:
        print("Warning: No valid year column found with values in 2000–2024")
        year_col = 'refYear'  # Fallback to refYear
        df_imp[year_col] = df_imp[year_col].apply(lambda x: x + 1900 if pd.notna(x) and x < 100 else x)
    
    df_imp = df_imp[df_imp[year_col].between(2000, 2024)]
    print(f"Rows after year filter using {year_col}: {len(df_imp)}")
    
    # Convert netWgt and primaryValue to numeric
    df_imp['netWgt'] = pd.to_numeric(df_imp['netWgt'], errors='coerce')
    df_imp['primaryValue'] = pd.to_numeric(df_imp['primaryValue'], errors='coerce')
    
    print("Sample rows (first 5):")
    print(df_imp[['reporterISO', year_col, 'flowCode', 'partnerDesc', 'cmdDesc', 'primaryValue', 'netWgt']].head(5))
    
    # Standardize using reporterISO as country name
    df_imp['country'] = df_imp['reporterISO'].apply(standardize_country)
    df_imp['partner_country'] = df_imp['partnerISO'].apply(standardize_country)
    df_imp = df_imp[df_imp['country'].isin(countries)]
    print(f"Rows after country filter: {len(df_imp)}")
    print(f"Unique countries after standardization: {sorted(df_imp['country'].dropna().unique())}")
    
    # Validate data availability
    print(f"Rows with non-null primaryValue: {len(df_imp[df_imp['primaryValue'].notna()])}")
    print(f"Rows with non-null netWgt: {len(df_imp[df_imp['netWgt'].notna()])}")
    print(f"Rows with positive primaryValue: {len(df_imp[df_imp['primaryValue'] > 0])}")
    print(f"Rows with positive netWgt: {len(df_imp[df_imp['netWgt'] > 0])}")
    
    # Filter for Import flow, World partner, and All Commodities
    df_imp_total = df_imp[(df_imp['flowCode'].str.strip().eq('Import')) &
                      (df_imp['partnerDesc'].str.strip().eq('World')) &
                      (df_imp['cmdDesc'].str.strip().eq('All Commodities'))]
    print(f"Rows after World/All Commodities/Import filter: {len(df_imp_total)}")
    
    if df_imp_total.empty:
        print("Warning: Import DataFrame is empty after filtering. Check filter values or data availability.")
        print("Unique flowCode values:", df_imp['flowCode'].unique())
        print("Unique partnerDesc values:", df_imp['partnerDesc'].unique())
        print("Unique cmdDesc values:", df_imp['cmdDesc'].unique())
    else:
        # Aggregate by country and year
        df_imp_agg = df_imp_total.groupby(['country', year_col]).agg({
            'primaryValue': 'sum',
            'netWgt': 'sum'
        }).reset_index()
        df_imp_agg['netWgt'] = df_imp_agg['netWgt'] / 1000  # kg to tonnes
        df_imp_agg.rename(columns={
            year_col: 'year',
            'primaryValue': 'total_imports_usd',
            'netWgt': 'total_imports_tonnes'
        }, inplace=True)
        df_imp_agg = df_imp_agg[df_imp_agg['year'].between(2000, 2024)]
        print("Aggregated import data sample:")
        print(df_imp_agg.head(5))
        df_imp_agg.set_index(['country', 'year'], inplace=True)
        print(f"Import Data shape: {df_imp_agg.shape}")
    
    # Create bilateral trade dataset
    df_imp_bilateral = df_imp[df_imp['partner_country'].isin(countries) & (df_imp['flowCode'] == 'Import')]
    df_imp_bilateral = df_imp_bilateral.groupby(['country', 'partner_country', year_col]).agg({
        'primaryValue': 'sum',
        'netWgt': 'sum'
    }).reset_index()
    df_imp_bilateral['netWgt'] = df_imp_bilateral['netWgt'] / 1000  # kg to tonnes
    df_imp_bilateral.rename(columns={
        year_col: 'year',
        'primaryValue': 'import_value_usd',
        'netWgt': 'import_weight_tonnes'
    }, inplace=True)
    df_imp_bilateral = df_imp_bilateral[df_imp_bilateral['year'].between(2000, 2024)]
    print("Bilateral import data sample:")
    print(df_imp_bilateral.head(5))
    df_imp_bilateral.to_csv(os.path.join(BASE_PATH, 'bilateral_import_data.csv'), index=False)
    print(f"Bilateral import data saved with shape: {df_imp_bilateral.shape}")
except Exception as e:
    print(f"Error in Import Data: {e}")
    df_imp_agg = pd.DataFrame()
    df_imp_bilateral = pd.DataFrame()

print("\nProcessing crop_and_livestock.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'crop_and_livestock.csv'))
    df_crop = pd.read_csv(os.path.join(BASE_PATH, 'crop_and_livestock.csv'))
    print("Unique Elements in crop_and_livestock:", df_crop['Element'].unique())
    df_crop['Area'] = df_crop.apply(lambda x: standardize_country(x['Area'], x.get('Area Code')), axis=1)
    df_crop = df_crop[df_crop['Area'].isin(countries)]
    df_crop['Year'] = df_crop['Year'].astype('Int64')
    df_crop['Value'] = pd.to_numeric(df_crop['Value'], errors='coerce')
    prod = df_crop[df_crop['Element'] == 'Production'].groupby(['Area', 'Year'])['Value'].sum().reset_index(name='total_production_tonnes')
    yield_avg = df_crop[df_crop['Element'] == 'Yield'].groupby(['Area', 'Year'])['Value'].mean().reset_index(name='avg_yield')
    if 'hg/ha' in df_crop['Unit'].values:
        yield_avg['avg_yield'] = yield_avg['avg_yield'] / 10  # Convert hg/ha to kg/ha
    yield_avg.rename(columns={'avg_yield': 'avg_yield_kg_ha'}, inplace=True)
    livestock = df_crop[df_crop['Element'] == 'Stocks'].groupby(['Area', 'Year'])['Value'].sum().reset_index(name='total_livestock_head')
    df_crop_agg = prod.merge(yield_avg, on=['Area', 'Year'], how='outer').merge(livestock, on=['Area', 'Year'], how='outer')
    df_crop_agg.rename(columns={'Area': 'country', 'Year': 'year'}, inplace=True)
    df_crop_agg.set_index(['country', 'year'], inplace=True)
    print(f"crop_and_livestock shape: {df_crop_agg.shape}")
except Exception as e:
    print(f"Error in crop_and_livestock: {e}")
    df_crop_agg = pd.DataFrame()

print("\nProcessing disasters.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'disasters.csv'))
    df_dis = pd.read_csv(os.path.join(BASE_PATH, 'disasters.csv'))
    df_dis['Country'] = df_dis.apply(lambda x: standardize_country(x['Country'], x.get('ISO')), axis=1)
    df_dis = df_dis[df_dis['Country'].isin(countries)]
    df_dis['Start Year'] = df_dis['Start Year'].astype('Int64')
    df_dis = df_dis[df_dis['Start Year'].between(2000, 2024)]
    for col in ['Total Deaths', 'No. Affected', 'No. Homeless', 'Total Damage, Adjusted (\'000 US$)']:
        df_dis[col] = pd.to_numeric(df_dis[col], errors='coerce')
    df_dis['Total Damage, Adjusted (\'000 US$)'] *= 1000  # Convert to USD
    df_dis_agg = df_dis.groupby(['Country', 'Start Year', 'Disaster Type']).agg({
        'DisNo.': 'count',
        'Total Deaths': 'sum',
        'No. Affected': 'sum',
        'No. Homeless': 'sum',
        'Total Damage, Adjusted (\'000 US$)': 'sum',
        'Magnitude': 'mean'
    }).reset_index()
    df_dis_agg.rename(columns={
        'Country': 'country',
        'Start Year': 'year',
        'Disaster Type': 'disaster_type',
        'DisNo.': 'num_disasters',
        'Total Deaths': 'total_deaths',
        'No. Affected': 'total_affected',
        'No. Homeless': 'total_homeless',
        'Total Damage, Adjusted (\'000 US$)': 'total_damage_adjusted_usd',
        'Magnitude': 'avg_magnitude'
    }, inplace=True)
    df_dis_agg.set_index(['country', 'year', 'disaster_type'], inplace=True)
    print(f"disasters shape: {df_dis_agg.shape}")
    df_dis_agg.to_csv(os.path.join(BASE_PATH, 'disasters_by_type.csv'))
    print(f"Disaster data by type saved with shape: {df_dis_agg.shape}")
except Exception as e:
    print(f"Error in disasters: {e}")
    df_dis_agg = pd.DataFrame()

print("\nProcessing Employment_Unemployment.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'Employment_Unemployment.csv'))
    df_emp = pd.read_csv(os.path.join(BASE_PATH, 'Employment_Unemployment.csv'))
    df_emp['Country Name'] = df_emp.apply(lambda x: standardize_country(x['Country Name'], x.get('Country Code')), axis=1)
    df_emp = df_emp[df_emp['Country Name'].isin(countries)]
    id_vars = ['Country Name', 'Country Code', 'Series Name', 'Series Code']
    df_emp_melt = melt_wide_df(df_emp, id_vars=id_vars)
    df_emp_pivot = df_emp_melt.pivot_table(index=['Country Name', 'year'], columns='Series Name', values='value')
    df_emp_pivot.reset_index(inplace=True)
    df_emp_pivot.rename(columns={
        'Country Name': 'country',
        'Employment to population ratio, 15+, female (%) (modeled ILO estimate)': 'employment_to_pop_ratio_female',
        'Employment to population ratio, 15+, male (%) (modeled ILO estimate)': 'employment_to_pop_ratio_male',
        'Employment to population ratio, ages 15-24, female (%) (national estimate)': 'employment_to_pop_ratio_1524_female',
        'Employment to population ratio, ages 15-24, male (%) (national estimate)': 'employment_to_pop_ratio_1524_male',
        'Employment to population ratio, ages 15-24, total (%) (modeled ILO estimate)': 'employment_to_pop_ratio_1524_total',
        'Unemployment with advanced education, male (% of male labor force with advanced education)': 'unemployment_advanced_education_male',
        'Unemployment with advanced education, female (% of female labor force with advanced education)': 'unemployment_advanced_education_female',
        'Unemployment with basic education (% of total labor force with basic education)': 'unemployment_basic_education',
        'Unemployment with basic education, female (% of female labor force with basic education)': 'unemployment_basic_education_female',
        'Unemployment with basic education, male (% of male labor force with basic education)': 'unemployment_basic_education_male',
        'Unemployment, female (% of female labor force) (modeled ILO estimate)': 'unemployment_female',
        'Unemployment, male (% of male labor force) (modeled ILO estimate)': 'unemployment_male',
        'Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)': 'unemployment_youth_female',
        'Unemployment, youth male (% of male labor force ages 15-24) (modeled ILO estimate)': 'unemployment_youth_male',
        'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)': 'unemployment_youth_total'
    }, inplace=True)
    if 'unemployment_total_pct_labor_force' in df_emp_pivot.columns:
        df_emp_pivot = df_emp_pivot.drop(columns=['unemployment_total_pct_labor_force'])
    df_emp_pivot.set_index(['country', 'year'], inplace=True)
    print(f"Employment_Unemployment shape: {df_emp_pivot.shape}")
except Exception as e:
    print(f"Error in Employment_Unemployment: {e}")
    df_emp_pivot = pd.DataFrame()

print("\nProcessing population_and_demographics.csv...")
try:
    check_file_exists(os.path.join(BASE_PATH, 'population_and_demographics.csv'))
    df_pop = pd.read_csv(os.path.join(BASE_PATH, 'population_and_demographics.csv'))
    df_pop['Area'] = df_pop.apply(lambda x: standardize_country(x['Area'], x.get('Area Code')), axis=1)
    df_pop = df_pop[df_pop['Area'].isin(countries)]
    df_pop['Year'] = df_pop['Year'].astype('Int64')
    df_pop['Value'] = pd.to_numeric(df_pop['Value'], errors='coerce') * 1000
    df_pop_pivot = df_pop.pivot_table(index=['Area', 'Year'], columns='Element', values='Value')
    df_pop_pivot.reset_index(inplace=True)
    df_pop_pivot.rename(columns={
        'Area': 'country',
        'Year': 'year',
        'Total Population - Both sexes': 'total_population',
        'Total Population - Male': 'population_male',
        'Total Population - Female': 'population_female',
        'Rural population': 'rural_population',
        'Urban population': 'urban_population'
    }, inplace=True)
    df_pop_pivot['urbanization_pct'] = (df_pop_pivot['urban_population'] / df_pop_pivot['total_population']) * 100
    # Cap urbanization_pct at 100%
    invalid_urban = df_pop_pivot[df_pop_pivot['urbanization_pct'] > 100][['country', 'year', 'urban_population', 'total_population', 'urbanization_pct']]
    if not invalid_urban.empty:
        print("Warning: Invalid urbanization_pct (>100%) found:")
        print(invalid_urban)
    df_pop_pivot['urbanization_pct'] = df_pop_pivot['urbanization_pct'].clip(upper=100)
    df_pop_pivot.set_index(['country', 'year'], inplace=True)
    print(f"population_and_demographics shape: {df_pop_pivot.shape}")
except Exception as e:
    print(f"Error in population_and_demographics: {e}")
    df_pop_pivot = pd.DataFrame()

print("\nMerging datasets...")
all_dfs = [
    df_core_pivot,
    df_res_pivot,
    df_social_pivot,
    df_exp_agg,
    df_imp_agg,
    df_crop_agg,
    df_dis_agg,
    df_emp_pivot,
    df_pop_pivot
]
df_integrated = df_core_pivot
for df in all_dfs[1:]:
    if not df.empty:
        df_integrated = df_integrated.join(df, how='outer', lsuffix='_left', rsuffix='_right')
print(f"Initial merged shape: {df_integrated.shape}")

# Filter for target countries and years
df_integrated = df_integrated.reset_index()
df_integrated = df_integrated[df_integrated['country'].isin(countries) & df_integrated['year'].between(2000, 2024)]
print(f"Filtered merged shape: {df_integrated.shape}")

if df_integrated.empty:
    print("Error: Integrated DataFrame is empty after filtering. Check country names and year ranges.")
    print("Unique countries in merged DataFrame:", df_integrated['country'].unique())
    raise ValueError("Integrated DataFrame is empty")

print("\nHandling missing data...")
missing_percentages = df_integrated.isna().mean() * 100
print("Missing data percentages before imputation:\n", missing_percentages[missing_percentages > 0])

# Create imputed flag columns in a single step to avoid fragmentation
numeric_cols = df_integrated.select_dtypes(include=['int64', 'float64']).columns
imputed_flags = {f'{col}_imputed': df_integrated[col].isna().astype(int) for col in numeric_cols if col not in ['year']}
df_imputed_flags = pd.DataFrame(imputed_flags, index=df_integrated.index)
df_integrated = pd.concat([df_integrated, df_imputed_flags], axis=1)

# Impute missing values
for col in numeric_cols:
    if col != 'year':
        if any(word in col.lower() for word in ['disaster', 'death', 'affected', 'homeless', 'damage']):
            df_integrated[col] = df_integrated[col].fillna(0)
        else:
            df_integrated[col] = df_integrated.groupby('country')[col].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill()
            )
            if df_integrated[col].isna().sum() > 0:
                df_integrated[col] = df_integrated[col].fillna(df_integrated[col].mean())

print("\nReconciling and normalizing...")
if all(col in df_integrated.columns for col in ['total_exports_usd', 'total_imports_usd', 'gdp_current_usd']):
    computed_trade = ((df_integrated['total_exports_usd'] + df_integrated['total_imports_usd']) / df_integrated['gdp_current_usd']) * 100
    df_integrated['trade_gdp_pct'] = df_integrated['trade_gdp_pct'].combine_first(computed_trade)
numeric_cols = df_integrated.select_dtypes(include=['int64', 'float64']).columns
df_integrated[numeric_cols] = df_integrated[numeric_cols].clip(lower=0)

# Data Integrity Checks
print("\nData Integrity Checks:")
print("Unique countries in final dataset:", sorted(df_integrated['country'].unique()))
print("Descriptive statistics:")
print(df_integrated[numeric_cols].describe())
if os.path.exists(os.path.join(BASE_PATH, 'disasters_by_type.csv')):
    df_dis_type = pd.read_csv(os.path.join(BASE_PATH, 'disasters_by_type.csv'))
    print("Unique disaster types:", df_dis_type['disaster_type'].unique())

print("Final shape:", df_integrated.shape)
print("Missing values:\n", df_integrated.isna().sum())
print("Sample data:\n", df_integrated.head(10))
output_path = os.path.join(BASE_PATH, 'integrated_country_year.csv')
df_integrated.reset_index(drop=True).to_csv(output_path, index=False)
print(f"Saved to '{output_path}'")