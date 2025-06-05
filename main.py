import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D
import factor_analyzer as fact
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

def fill_na(x):
    assert isinstance(x, pd.DataFrame)
    for col in x.columns:
        if is_numeric_dtype(x[col]):
            x[col] = x[col].fillna(x[col].mean())
        else:
            x[col] = x[col].fillna(x[col].mode()[0])
    return x

def tabelare_varianta(varianta, etichete):
    return pd.DataFrame({
        "Varianta": varianta[0],
        "Varianta cumulata": np.cumsum(varianta[0]),
        "Procent varianta": varianta[1]*100,
        "Procent varianta cumulata": varianta[2]*100
    }, index=etichete)

def tabelare(data, linii, coloane):
    return pd.DataFrame(data, index=linii, columns=coloane)

def corelograma(r, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(r, vmin=-1, vmax=1, cmap="RdYlBu", center=0, annot=True, ax=ax)
    plt.title(title)
    return fig

st.set_page_config(page_title="SGI 2022 Explorer", layout="wide")
file_path = "SGI-2022-Final.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()


df = fill_na(df)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns


def normalize_data(data, method='zscore'):
    norm_df = data.copy()

    for col in numeric_columns:
        if method == 'zscore':
            norm_df[col] = stats.zscore(data[col].values)
        elif method == 'log':
            min_val = data[col].min()
            if min_val <= 0:
                offset = abs(min_val) + 1
                norm_df[col] = np.log(data[col] + offset)
            else:
                norm_df[col] = np.log(data[col])
        elif method == 'minmax':
            norm_df[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return norm_df


std_df = normalize_data(df, method='zscore')
zscore_df = normalize_data(df, method='zscore')
log_df = normalize_data(df, method='log')
minmax_df = normalize_data(df, method='minmax')

indicator_categories = {
    "Economic Policies": [
        "Inflation", "Gross Fixed Capital Formation", "Real Interest Rates",
        "Potential Output Growth Rate", "Real GDP Growth Rate", "Unemployment",
        "Long-term Unemployment", "Youth Unemployment", "Low-skilled Unemployment",
        "Employment", "Low Pay Incidence", "Employment Rates by Gender",
        "Involuntary Part-time Employment", "Tax System Complexity", "Structural Balance",
        "Statutory Corporate Tax Rate", "Redistribution Effect",
        "Statutory Maximum Personal Income Tax Rate", "Debt to GDP", "Primary Balance",
        "Gross Interest Payments by General Govt", "Budget Consolidation",
        "Debt per Child", "External Debt to GDP", "Public R&D Spending",
        "Private R&D Spending", "Total Researchers", "Intellectual Property Licenses",
        "PCT Patent Applications", "Quality of Overall Infrastructure",
        "Tier 1 Capital Ratio", "Banks' Nonperforming Loans", "Financial Secrecy Score",
        "External Debt to Exports"
    ],
    "Social Policies": [
        "Upper Secondary Attainment", "Tertiary Attainment", "PISA results",
        "PISA Results According to Socioeconomic Background",
        "Pre-primary Education Expenditure", "PISA Low Achievers in all Subjects",
        "Less Than Upper Secondary Education by Gender", "Poverty Rate", "NEET Rate",
        "Gini Coefficient", "Gender Equality in Parliaments", "Life Satisfaction",
        "Gender Wage Gap", "Spending on Preventive Health Programs",
        "Healthy Life Expectancy", "Infant Mortality", "Perceived Health Status",
        "Household Out Of Pocket Expenses", "Physicians per 1000 Inhabitants",
        "Child Care Enrolment, Age 0-2", "Child Care Enrolment, Age 3-5",
        "Fertility Rate", "Child Poverty Rate", "Female Labor Force Participation Rate",
        "Older Employment", "Old Age Dependency Ratio", "Senior Citizen Poverty",
        "FB-N Upper Secondary Attainment", "FB-N Tertiary Attainment",
        "FB-N Unemployment", "FB-N Employment", "Homicides", "Personal Security",
        "Confidence in Police", "ODA"
    ],
    "Environmental Policies": [
        "Energy Productivity", "Gross Greenhouse Gas Emissions", "Particulate Matter",
        "Biocapacity", "Waste Generation", "Material Recycling", "Material Footprint",
        "Multilateral Environmental Agreements", "Net Greenhouse Gas Emissions",
        "Renewable Energy", "Biodiversity"
    ]
}

all_df_columns = df.columns.tolist()
country_column = all_df_columns[0]
all_indicators = all_df_columns[1:]

categorized_indicators = []
for category_indicators in indicator_categories.values():
    categorized_indicators.extend(category_indicators)

uncategorized_indicators = [ind for ind in all_indicators if ind not in categorized_indicators]

if "Sustainable Policies" in uncategorized_indicators:
    uncategorized_indicators.remove("Sustainable Policies")

if uncategorized_indicators:
    indicator_categories["Other"] = uncategorized_indicators

indicator_to_category = {}
for category, indicators in indicator_categories.items():
    for indicator in indicators:
        indicator_to_category[indicator] = category

st.title("SGI Dataset Explorer")
st.sidebar.header("Main Filters")
country = st.sidebar.selectbox("Select a country", sorted(df.iloc[:, 0].unique()), key="main_country")
st.sidebar.header("Advanced Filters")
with st.sidebar.expander("Column Range Filters"):
    filter_column = st.selectbox("Select column to filter", numeric_columns, key="filter_column")

    min_value = float(df[filter_column].min())
    max_value = float(df[filter_column].max())

    min_filter, max_filter = st.slider(
        f"Range for {filter_column}",
        min_value, max_value,
        (min_value, max_value),
        key="range_filter"
    )

    filtered_df = df[(df[filter_column] >= min_filter) & (df[filter_column] <= max_filter)]

st.sidebar.header("Country Comparison")
country1 = st.sidebar.selectbox("Select first country", sorted(df.iloc[:, 0].unique()), key="country1")
country2 = st.sidebar.selectbox("Select second country", sorted(df.iloc[:, 0].unique()), key="country2")
comparison_indicator = st.sidebar.selectbox("Select an Indicator", df.columns[1:], key="comparison_indicator")
st.sidebar.header("Top/Bottom Rankings")
ranking_indicator = st.sidebar.selectbox("Select indicator for ranking", numeric_columns, key="ranking_indicator")
ranking_count = st.sidebar.slider("Number of countries to show", 3, 15, 5, key="ranking_count")
ranking_order = st.sidebar.radio("Order", ["Top (Highest)", "Bottom (Lowest)"], key="ranking_order")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Dataset Overview", "Data Analysis", "Country Details", "Comparisons", "Factor Analysis", "Geographic Analysis", "Multiple Regression"])

with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Dataset Preview")
        sort_col, sort_order = st.columns(2)
        with sort_col:
            sort_by = st.selectbox("Sort by column", df.columns, key="sort_column")
        with sort_order:
            ascending = st.radio("Sort order", ["Ascending", "Descending"], key="sort_order") == "Ascending"
        st.dataframe(df.sort_values(by=sort_by, ascending=ascending))

    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Dataset Info")
        st.write(f"Total countries: {len(df)}")
        st.write(f"Total indicators: {len(df.columns) - 1}")

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Indicator Analysis")
        indicator = st.selectbox("Select an Indicator", df.columns[1:], key="main_indicator")
        st.write(f"**Statistics for {indicator}:**")
        stats = df[indicator].describe().to_frame().T
        st.dataframe(stats)
        show_top = st.checkbox("Show top 5 countries for this indicator", key="show_top")
        if show_top:
            st.write("**Top 5 countries:**")
            top_df = df.sort_values(by=indicator, ascending=False).head(5)
            st.dataframe(top_df[[df.columns[0], indicator]])

    with col2:
        st.subheader(f"{indicator} Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[indicator], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader(f"{indicator} by Country")
        plt.figure(figsize=(10, 6))
        plot_data = df.sort_values(by=indicator, ascending=False)
        sns.barplot(x=plot_data.iloc[:, 0], y=plot_data[indicator])
        plt.xticks(rotation=90)
        st.pyplot(plt)


    st.subheader("Grouped Analysis")


    df_filled = df.copy()
    df_filled = fill_na(df_filled)


    category_stats = {}
    for category, indicators in indicator_categories.items():
        category_data = df_filled[indicators]
        category_stats[category] = {
            "Mean": category_data.mean().mean(),
            "Median": category_data.median().median(),
            "Std": category_data.std().mean(),
            "Min": category_data.min().min(),
            "Max": category_data.max().max()
        }

    stats_df = pd.DataFrame(category_stats).T
    st.write("**Statistics by Category:**")
    st.dataframe(stats_df)


    plt.figure(figsize=(10, 6))
    sns.barplot(x=stats_df.index, y=stats_df['Mean'])
    plt.xticks(rotation=45)
    plt.title("Mean Values by Category")
    st.pyplot(plt)

with tab3:
    st.subheader(f"Details for {country}")
    data_type = st.radio(
        "Select data visualization type",
        ["Raw Values", "Z-Score Normalization", "Log Transformation", "Min-Max Scaling"],
        horizontal=True
    )

    if data_type == "Raw Values":
        selected_df = df
        value_label = "Value"
    elif data_type == "Z-Score Normalization":
        selected_df = zscore_df
        value_label = "Z-Score"
    elif data_type == "Log Transformation":
        selected_df = log_df
        value_label = "Log Value"
    else:
        selected_df = minmax_df
        value_label = "Scaled Value"

    country_data = selected_df[selected_df.iloc[:, 0] == country].iloc[0, 1:].reset_index()
    country_data.columns = ['Indicator', value_label]
    country_data['Category'] = country_data['Indicator'].map(
        lambda x: indicator_to_category.get(x, "Other")
    )

    if data_type != "Raw Values":
        raw_values = df[df.iloc[:, 0] == country].iloc[0, 1:].reset_index()
        raw_values.columns = ['Indicator', 'Raw Value']
        country_data = pd.merge(country_data, raw_values, on='Indicator')


    if data_type != "Raw Values":
        with st.expander("Compare Normalization Methods Performance"):
            country_raw_values = df[df.iloc[:, 0] == country].iloc[0, 1:].values
            country_zscore = zscore_df[zscore_df.iloc[:, 0] == country].iloc[0, 1:].values
            country_log = log_df[log_df.iloc[:, 0] == country].iloc[0, 1:].values
            country_minmax = minmax_df[minmax_df.iloc[:, 0] == country].iloc[0, 1:].values

            methods_comparison = pd.DataFrame({
                'Raw': country_raw_values,
                'Z-Score': country_zscore,
                'Log': country_log,
                'Min-Max': country_minmax
            }, index=df.columns[1:])


            methods_comparison = methods_comparison.fillna(0)
            skewness = {
                'Raw': float(methods_comparison['Raw'].skew()),
                'Z-Score': float(methods_comparison['Z-Score'].skew()),
                'Log': float(methods_comparison['Log'].skew()),
                'Min-Max': float(methods_comparison['Min-Max'].skew())
            }

            kurtosis = {
                'Raw': float(methods_comparison['Raw'].kurt()),
                'Z-Score': float(methods_comparison['Z-Score'].kurt()),
                'Log': float(methods_comparison['Log'].kurt()),
                'Min-Max': float(methods_comparison['Min-Max'].kurt())
            }

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribution Statistics")
                st.write("**Skewness** (closer to 0 is better):")
                for method, value in skewness.items():
                    st.write(f"{method}: {value:.3f}")

                st.write("**Kurtosis** (closer to 0 is better for normal distribution):")
                for method, value in kurtosis.items():
                    st.write(f"{method}: {value:.3f}")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                for method in ['Raw', 'Z-Score', 'Log', 'Min-Max']:
                    sns.kdeplot(methods_comparison[method].values, label=method, ax=ax)
                plt.legend()
                plt.title("Distribution Comparison of Normalization Methods")
                st.pyplot(fig)


    for category in indicator_categories.keys():
        category_data = country_data[country_data['Category'] == category]

        if len(category_data) > 0:
            with st.expander(f"{category} Indicators for {country}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    chart_data = category_data.sort_values(by=value_label)
                    fig, ax = plt.subplots(figsize=(10, max(4, len(chart_data) * 0.4)))

                    if data_type == "Raw Values":
                        sns.barplot(x=value_label, y='Indicator', data=chart_data, ax=ax)
                        ax.set_title(f"{category} Indicators for {country}")
                        x_max = max(10, chart_data[value_label].max() * 1.1)
                        ax.set_xlim(0, x_max)
                    else:
                        bars = sns.barplot(
                            x=value_label,
                            y='Indicator',
                            data=chart_data,
                            ax=ax,
                            palette="RdBu_r"
                        )

                        for i, bar in enumerate(bars.patches):
                            if chart_data.iloc[i][value_label] < 0:
                                bar.set_facecolor('#d8b365')
                            else:
                                bar.set_facecolor('#5ab4ac')

                        ax.set_title(f"{category} Indicators for {country} (Standardized)")
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        z_max = max(2, abs(chart_data[value_label]).max() * 1.1)
                        ax.set_xlim(-z_max, z_max)

                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    if data_type == "Raw Values":
                        st.dataframe(
                            category_data[['Indicator', value_label]].sort_values(by=value_label, ascending=False))
                    else:
                        display_data = category_data[['Indicator', value_label, 'Raw Value']].sort_values(
                            by=value_label, ascending=False)
                        st.dataframe(display_data)

with tab4:
    st.subheader(f"Comparison: {country1} vs {country2}")
    valid_categories = []
    for category in indicator_categories.keys():
        indicators = indicator_categories[category]
        if len(indicators) > 0:
            category_data = df[df.iloc[:, 0].isin([country1, country2])][
                [df.columns[0]] + indicators
                ]
            if not category_data.empty and len(category_data.columns) > 1:
                valid_categories.append(category)

    col1, col2 = st.columns([3, 1])

    with col1:
        if valid_categories:
            category_tabs = st.tabs(valid_categories)

            for i, category in enumerate(valid_categories):
                with category_tabs[i]:
                    category_indicators = indicator_categories[category]

                    if len(category_indicators) > 0:
                        comparison_data = df[df.iloc[:, 0].isin([country1, country2])][
                            [df.columns[0]] + category_indicators
                            ]
                        comparison_melted = pd.melt(
                            comparison_data,
                            id_vars=[df.columns[0]],
                            var_name='Indicator',
                            value_name='Value'
                        )
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x='Indicator', y='Value', hue=df.columns[0], data=comparison_melted, ax=ax)
                        plt.xticks(rotation=90)
                        plt.legend(title="")
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.dataframe(comparison_data)
        else:
            st.write("No valid categories with data found for comparison")

    with col2:
        st.subheader(f"Selected Indicator: {comparison_indicator}")
        data = df[df.iloc[:, 0].isin([country1, country2])][[df.columns[0], comparison_indicator]]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.barplot(x=data.iloc[:, 0], y=data[comparison_indicator], ax=ax[0])
        ax[0].set_title("Bar Comparison")
        ax[0].set_ylabel(comparison_indicator)
        ax[1].pie(
            data[comparison_indicator],
            labels=data.iloc[:, 0],
            autopct='%1.1f%%',
            startangle=90
        )
        ax[1].set_title("Pie Chart Comparison")

        plt.tight_layout()
        st.pyplot(fig)
        st.write("**Difference:**")
        val1 = float(data[data.iloc[:, 0] == country1][comparison_indicator].values[0])
        val2 = float(data[data.iloc[:, 0] == country2][comparison_indicator].values[0])
        diff = val1 - val2
        diff_percent = (diff / val2) * 100 if val2 != 0 else float('inf')

        st.write(f"{country1}: {val1:.2f}")
        st.write(f"{country2}: {val2:.2f}")
        st.write(f"Absolute difference: {abs(diff):.2f}")
        st.write(f"Relative difference: {abs(diff_percent):.1f}%")

        if diff > 0:
            st.write(f"**{country1}** is higher by {diff:.2f} ({diff_percent:.1f}%)")
        elif diff < 0:
            st.write(f"**{country2}** is higher by {abs(diff):.2f} ({abs(diff_percent):.1f}%)")
        else:
            st.write("Both countries have the same value")

st.header(f"{'Top' if ranking_order == 'Top (Highest)' else 'Bottom'} {ranking_count} Countries by {ranking_indicator}")

sorted_df = df.sort_values(by=ranking_indicator, ascending=ranking_order == "Bottom (Lowest)")
top_countries = sorted_df.head(ranking_count)

col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_countries.iloc[:, 0], y=top_countries[ranking_indicator], ax=ax)
    ax.set_title(f"{'Top' if ranking_order == 'Top (Highest)' else 'Bottom'} {ranking_count} Countries")
    ax.set_ylabel(ranking_indicator)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.dataframe(top_countries[[df.columns[0], ranking_indicator]])

st.header("Indicator Correlations")
with st.expander("Correlation Analysis"):
    corr_norm_method = st.selectbox(
        "Select normalization method for correlation analysis",
        ["Raw Values", "Z-Score Normalization", "Log Transformation", "Min-Max Scaling"],
        key="corr_norm_method"
    )

    if corr_norm_method == "Raw Values":
        corr_df = df
    elif corr_norm_method == "Z-Score Normalization":
        corr_df = zscore_df
    elif corr_norm_method == "Log Transformation":
        corr_df = log_df
    else:
        corr_df = minmax_df

    st.subheader("Correlation Matrices by Category")

    category_tabs = st.tabs(list(indicator_categories.keys()))


    for i, category in enumerate(indicator_categories.keys()):
        with category_tabs[i]:
            category_indicators = indicator_categories[category]

            if len(category_indicators) > 1:

                category_df = corr_df[category_indicators].copy()
                category_df = category_df.fillna(0)

                corr_matrix = category_df.corr()
                fig, ax = plt.subplots(figsize=(max(8, len(category_indicators) * 0.7),
                                                max(6, len(category_indicators) * 0.7)))
                cmap = sns.diverging_palette(230, 20, as_cmap=True)

                sns.heatmap(corr_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
                            square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)

                plt.title(f"Correlation Matrix for {category} Indicators")
                plt.tight_layout()
                st.pyplot(fig)


                corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                corr_pairs = corr_pairs[(corr_pairs < 1.0) & (corr_pairs > -1.0)]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"Strongest Positive Correlations in {category}")
                    positive_corr = corr_pairs[corr_pairs > 0].head(10)
                    if not positive_corr.empty:
                        st.dataframe(
                            positive_corr.reset_index().rename(
                                columns={'level_0': 'Indicator 1', 'level_1': 'Indicator 2', 0: 'Correlation'}
                            )
                        )
                    else:
                        st.write("No positive correlations found.")

                with col2:
                    st.subheader(f"Strongest Negative Correlations in {category}")
                    negative_corr = corr_pairs[corr_pairs < 0].head(10)
                    if not negative_corr.empty:
                        st.dataframe(
                            negative_corr.reset_index().rename(
                                columns={'level_0': 'Indicator 1', 'level_1': 'Indicator 2', 0: 'Correlation'}
                            )
                        )
                    else:
                        st.write("No negative correlations found.")
            else:
                st.write(f"Not enough indicators in the {category} category to compute correlations.")

    st.subheader("Overall Strongest Correlations")
    numeric_df = corr_df.select_dtypes(include=['float64', 'int64'])
    overall_corr_matrix = numeric_df.corr()
    corr_pairs = overall_corr_matrix.unstack().sort_values(ascending=False)
    strong_corr = corr_pairs[(corr_pairs < 1.0) & (corr_pairs > -1.0)].head(15)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strongest Positive Correlations")
        positive_corr = strong_corr[strong_corr > 0]
        st.dataframe(
            positive_corr.reset_index().rename(
                columns={'level_0': 'Indicator 1', 'level_1': 'Indicator 2', 0: 'Correlation'}
            )
        )

    with col2:
        st.subheader("Strongest Negative Correlations")
        negative_corr = corr_pairs[corr_pairs < 0].head(15)
        st.dataframe(
            negative_corr.reset_index().rename(
                columns={'level_0': 'Indicator 1', 'level_1': 'Indicator 2', 0: 'Correlation'}
            )
        )

with tab5:
    st.header("Factor Analysis")

    fa_category = st.selectbox("Select Category for Factor Analysis",
                               list(indicator_categories.keys()),
                               key="fa_category")


    fa_indicators = indicator_categories[fa_category]

    if len(fa_indicators) < 3:
        st.error(f"Not enough indicators in the {fa_category} category. Need at least 3 indicators for factor analysis.")
    else:
        fa_data = df[fa_indicators].copy()
        fa_data = fill_na(fa_data)
        x = fa_data.values
        n, m = x.shape
        if n <= 1:
            st.error("Not enough data for factor analysis.")
        else:
            x_std = StandardScaler().fit_transform(x)
            kmo_all, kmo_model = calculate_kmo(fa_data)
            chi_square_value, p_value = calculate_bartlett_sphericity(fa_data)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Kaiser-Meyer-Olkin (KMO) Test")
                st.write(f"KMO Score: {kmo_model:.3f}")
                if kmo_model < 0.5:
                    st.warning("KMO < 0.5: Sample may not be adequate for factor analysis")
                elif kmo_model < 0.7:
                    st.info("0.5 ≤ KMO < 0.7: Sample is mediocre but acceptable")
                else:
                    st.success("KMO ≥ 0.7: Sample is adequate for factor analysis")

            with col2:
                st.subheader("Bartlett's Test of Sphericity")
                st.write(f"Chi-square: {chi_square_value:.3f}")
                st.write(f"p-value: {p_value:.6f}")
                if p_value < 0.05:
                    st.success("p < 0.05: Correlation matrix is not an identity matrix")
                else:
                    st.warning("p ≥ 0.05: Correlation matrix might be an identity matrix")


            eigenvalues, _ = fact.factor_analyzer.FactorAnalyzer().fit(x_std).get_eigenvalues()

            st.subheader("Scree Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
            ax.axhline(y=1, color='r', linestyle='--')
            ax.set_xlabel('Factor Number')
            ax.set_ylabel('Eigenvalue')
            ax.set_title('Scree Plot')
            ax.grid(True)
            st.pyplot(fig)

            n_factors = st.slider("Select number of factors",
                                 min_value=1,
                                 max_value=min(m, 10),
                                 value=min(3, m))


            model_af = fact.FactorAnalyzer(n_factors=n_factors, rotation="varimax")
            model_af.fit(x_std)


            varianta = model_af.get_factor_variance()
            etichete = [f"F{i+1}" for i in range(n_factors)]
            varianta_df = tabelare_varianta(varianta, etichete)

            st.subheader("Factor Analysis Results")
            st.write("**Factor Variance**")
            st.dataframe(varianta_df)


            loadings = model_af.loadings_
            loadings_df = tabelare(loadings[:, :n_factors], fa_indicators, etichete)


            st.write("**Factor Loadings**")
            st.dataframe(loadings_df)


            st.subheader("Factor Loadings Heatmap")
            fig = corelograma(loadings_df, "Correlations: Variables-Common Factors")
            st.pyplot(fig)


            if n_factors >= 2:
                st.subheader("Circle of Correlations (First 2 Factors)")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.add_patch(plt.Circle((0, 0), 1, linewidth=0.8, color="red", fill=False))
                ax.axhline(0, color="black")
                ax.axvline(0, color="black")
                ax.scatter(loadings[:, 0], loadings[:, 1])

                for i, var in enumerate(fa_indicators):
                    ax.text(loadings[i, 0], loadings[i, 1], var)

                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_title("Circle of Correlations (Factor 1 vs Factor 2)")
                ax.set_xlabel("Factor 1")
                ax.set_ylabel("Factor 2")
                ax.grid(True)
                st.pyplot(fig)

            if st.checkbox("Show factor scores by country"):
                factor_scores = model_af.transform(x_std)
                factor_scores_df = pd.DataFrame(
                    factor_scores,
                    index=df[df.columns[0]],
                    columns=[f"Factor {i+1}" for i in range(n_factors)]
                )

                st.write("**Factor Scores by Country**")
                st.dataframe(factor_scores_df)
                st.subheader(f"Top Countries by Factor 1")
                top_countries = factor_scores_df.sort_values(by="Factor 1", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_countries.index, y=top_countries["Factor 1"], ax=ax)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

with tab6:
    st.header("Geographic Analysis")

    st.info("World maps for easier geographic comparison.")

    @st.cache_data
    def load_world_data():
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        try:
            world = gpd.read_file(url)
            country_name_map = {
                'United States of America': 'United States',
                'Czech Rep.': 'Czechia',
                'Korea': 'South Korea',
                'United Kingdom': 'United Kingdom'
            }
            country_col = 'NAME' if 'NAME' in world.columns else 'ADMIN'
            world[country_col] = world[country_col].replace(country_name_map)

            return world, country_col
        except Exception as e:
            st.error(f"Error loading world data: {str(e)}")
            return None, None

    try:
        world_data = load_world_data()
        if world_data is not None:
            world, country_col = world_data

            col1, col2 = st.columns([3, 1])

            with col2:
                st.subheader("Map Settings")

                geo_indicator = st.selectbox("Select Indicator", df.columns[1:], key="geo_indicator")

                use_binning = st.checkbox("Enable Binning", value=False, key="use_binning")

                if use_binning:
                    bin_method = st.radio(
                        "Binning Method",
                        ["Equal Width", "Equal Frequency", "Custom"],
                        key="bin_method"
                    )

                    if bin_method in ["Equal Width", "Equal Frequency"]:
                        n_bins = st.slider("Number of Bins", 2, 10, 5, key="n_bins")
                    else:
                        custom_bins_text = st.text_area(
                            "Enter bin edges (comma-separated)",
                            value="0,25,50,75,100",
                            key="custom_bins"
                        )
                        try:
                            custom_bins = [float(x.strip()) for x in custom_bins_text.split(",")]
                        except ValueError:
                            st.error("Please enter valid numeric values")
                            custom_bins = [0, 25, 50, 75, 100]

            with col1:
                st.subheader(f"World Map: {geo_indicator}")

                geo_data = df[[df.columns[0], geo_indicator]].copy()

                if use_binning:
                    if bin_method == "Equal Width":
                        geo_data[f"{geo_indicator}_binned"], bin_edges = pd.cut(
                            geo_data[geo_indicator],
                            n_bins,
                            retbins=True,
                            labels=[f"Bin {i+1}" for i in range(n_bins)]
                        )
                        st.write(f"**Bin edges:** {[round(edge, 2) for edge in bin_edges]}")

                    elif bin_method == "Equal Frequency":
                        geo_data[f"{geo_indicator}_binned"], bin_edges = pd.qcut(
                            geo_data[geo_indicator],
                            n_bins,
                            retbins=True,
                            labels=[f"Bin {i+1}" for i in range(n_bins)],
                            duplicates='drop'
                        )
                        st.write(f"**Bin edges:** {[round(edge, 2) for edge in bin_edges]}")

                    elif bin_method == "Custom":
                        geo_data[f"{geo_indicator}_binned"] = pd.cut(
                            geo_data[geo_indicator],
                            custom_bins,
                            labels=[f"Bin {i+1}" for i in range(len(custom_bins)-1)]
                        )
                        st.write(f"**Custom bin edges:** {custom_bins}")

                    bin_counts = geo_data[f"{geo_indicator}_binned"].value_counts().sort_index()
                    bin_stats = pd.DataFrame({
                        'Count': bin_counts,
                        'Percentage': (bin_counts / len(geo_data) * 100).round(1)
                    })
                    st.write("**Binning Results:**")
                    st.dataframe(bin_stats)

                merged = world.merge(geo_data, left_on=country_col, right_on=df.columns[0], how='left')

                merged = merged[~merged[geo_indicator].isna()]

                fig, ax = plt.subplots(1, 1, figsize=(15, 10))

                if not merged.empty:
                    if use_binning:
                        merged.plot(
                            column=f"{geo_indicator}_binned",
                            ax=ax,
                            legend=True,
                            cmap='RdYlBu_r'
                        )
                    else:
                        vmin = merged[geo_indicator].min()
                        vmax = merged[geo_indicator].max()

                        merged.plot(
                            column=geo_indicator,
                            ax=ax,
                            legend=True,
                            cmap='RdYlBu_r',
                            vmin=vmin,
                            vmax=vmax
                        )

                    ax.set_title(f"{geo_indicator} by Country", pad=20, fontsize=14)
                    ax.set_axis_off()
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.subheader(f"{geo_indicator} Data Table")

                    table_data = geo_data.sort_values(by=geo_indicator, ascending=False)

                    table_data = table_data.reset_index(drop=True)
                    table_data.index = table_data.index + 1
                    table_data.index.name = 'Rank'

                    st.dataframe(table_data)

                    st.subheader(f"Summary Statistics: {geo_indicator}")

                    col1, col2 = st.columns(2)

                    with col1:
                        stats = geo_data[geo_indicator].describe()
                        st.write(f"**Mean:** {stats['mean']:.2f}")
                        st.write(f"**Median:** {stats['50%']:.2f}")
                        st.write(f"**Std. Dev.:** {stats['std']:.2f}")
                        st.write(f"**Min:** {stats['min']:.2f} ({geo_data.iloc[geo_data[geo_indicator].idxmin()][df.columns[0]]})")
                        st.write(f"**Max:** {stats['max']:.2f} ({geo_data.iloc[geo_data[geo_indicator].idxmax()][df.columns[0]]})")

                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(geo_data[geo_indicator], kde=True, ax=ax)
                        ax.set_title(f"Distribution of {geo_indicator}")
                        ax.set_xlabel(geo_indicator)
                        st.pyplot(fig)
                else:
                    st.error("No matching data found for the selected countries.")
        else:
            st.error("Could not load world map data.")

    except Exception as e:
        st.error(f"An error occurred in the Geographic Analysis section: {str(e)}")
        st.info("This section requires an internet connection to download geographic data. Please make sure you're connected to the internet.")

with tab7:
    st.header("Multiple Regression with 3D Visualization")



    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Variable Selection")


        dependent_var = st.selectbox(
            "Select Dependent Variable (Y)",
            numeric_columns,
            key="dependent_var"
        )


        available_vars = [col for col in numeric_columns if col != dependent_var]

        independent_var1 = st.selectbox(
            "Select First Independent Variable (X1)",
            available_vars,
            key="independent_var1"
        )

        available_vars2 = [col for col in available_vars if col != independent_var1]
        independent_var2 = st.selectbox(
            "Select Second Independent Variable (X2)",
            available_vars2,
            key="independent_var2"
        )


        normalize_regression = st.checkbox("Normalize data before regression", value=True)

        if normalize_regression:
            regression_norm_method = st.selectbox(
                "Normalization Method",
                ["Z-Score", "Min-Max", "Log"],
                key="regression_norm_method"
            )

    with col2:
        st.subheader("Regression Results")


        regression_data = df[[dependent_var, independent_var1, independent_var2]].copy()
        regression_data = fill_na(regression_data)

        if normalize_regression:
            if regression_norm_method == "Z-Score":
                regression_data = normalize_data(regression_data, method='zscore')
            elif regression_norm_method == "Min-Max":
                regression_data = normalize_data(regression_data, method='minmax')
            else:
                regression_data = normalize_data(regression_data, method='log')


        X = regression_data[[independent_var1, independent_var2]].values
        y = regression_data[dependent_var].values


        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        countries_filtered = df[df.columns[0]][mask].values

        if len(X) > 3:

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)


            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)


            st.write("**Model Performance:**")
            st.write(f"R² Score: {r2:.3f}")
            st.write(f"Mean Squared Error: {mse:.3f}")
            st.write(f"Mean Absolute Error: {mae:.3f}")

            st.write("**Model Coefficients:**")
            st.write(f"Intercept: {model.intercept_:.3f}")
            st.write(f"Coefficient for {independent_var1}: {model.coef_[0]:.3f}")
            st.write(f"Coefficient for {independent_var2}: {model.coef_[1]:.3f}")


            st.write("**Regression Equation:**")
            st.latex(f"{dependent_var} = {model.intercept_:.3f} + {model.coef_[0]:.3f} \\times {independent_var1} + {model.coef_[1]:.3f} \\times {independent_var2}")


    st.subheader("3D Visualization")

    if len(X) > 3:

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')


        scatter = ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', alpha=0.6, s=50)


        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)


        X_surf = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
        y_surf = model.predict(X_surf).reshape(x1_mesh.shape)


        surf = ax.plot_surface(x1_mesh, x2_mesh, y_surf, alpha=0.3, color='red')


        ax.set_xlabel(independent_var1)
        ax.set_ylabel(independent_var2)
        ax.set_zlabel(dependent_var)
        ax.set_title(f'3D Multiple Regression: {dependent_var} vs {independent_var1} & {independent_var2}')


        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

        st.pyplot(fig)


        st.subheader("Residual Analysis")

        residuals = y - y_pred

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(y=0, color='red', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residuals vs Predicted Values')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title('Q-Q Plot of Residuals')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)


        st.subheader("Countries Data and Predictions")

        results_df = pd.DataFrame({
            'Country': countries_filtered,
            f'{independent_var1}': X[:, 0],
            f'{independent_var2}': X[:, 1],
            f'{dependent_var} (Actual)': y,
            f'{dependent_var} (Predicted)': y_pred,
            'Residual': residuals
        })


        results_df['Abs_Residual'] = np.abs(residuals)
        results_df = results_df.sort_values('Abs_Residual', ascending=False)
        results_df = results_df.drop('Abs_Residual', axis=1)

        st.dataframe(results_df)


        st.subheader("Countries with Largest Prediction Errors")
        worst_predictions = results_df.head(5)
        st.dataframe(worst_predictions[['Country', f'{dependent_var} (Actual)', f'{dependent_var} (Predicted)', 'Residual']])
        
    else:
        st.error("Not enough data points for regression analysis. Need at least 4 countries with complete data.")
