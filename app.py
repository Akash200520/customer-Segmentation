import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0e1117;
    }

    /* Main content */
    .main {
        background: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #151922;
        border-right: 1px solid #2a2f3a;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #f5f7fa;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 25px;
    }

    /* KPI cards */
    .metric-card {
        background: linear-gradient(
            135deg,
            #171c27,
            #11151e
        );
        border: 1px solid #2a3140;
        border-radius: 16px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.20);
    }

    .metric-title {
        color: #9ca3af;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #7dd3fc;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Section headers */
    .section-title {
        color: #ffffff;
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Insight cards */
    .insight-card {
        background: #151a24;
        border: 1px solid #2b3340;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .insight-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
    }

    .insight-text {
        color: #aeb7c4;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 7px;
    }

    /* Highlight */
    .highlight {
        background: linear-gradient(
            135deg,
            #172554,
            #111827
        );
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .highlight-title {
        color: #93c5fd;
        font-size: 20px;
        font-weight: 800;
    }

    .highlight-text {
        color: #dbeafe;
        font-size: 15px;
        line-height: 1.7;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Mall_Customers.csv")

    return df


df = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# K-MEANS
# ============================================================

optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ============================================================
# SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)


# ============================================================
# CLUSTER PROFILE
# ============================================================

cluster_profile = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)


# ============================================================
# SEGMENT CLASSIFICATION
# ============================================================

overall_income = df["Annual Income (k$)"].mean()
overall_spending = df["Spending Score (1-100)"].mean()


def identify_segment(row):

    income = row["Average_Income"]
    spending = row["Average_Spending"]

    if income >= overall_income and spending >= overall_spending:
        return "High-Value Customers"

    elif income >= overall_income and spending < overall_spending:
        return "Potential Customers"

    elif income < overall_income and spending >= overall_spending:
        return "Frequent / Budget Customers"

    else:
        return "Low-Priority Customers"


cluster_profile["Customer_Segment"] = cluster_profile.apply(
    identify_segment,
    axis=1
)


# ============================================================
# MAP SEGMENTS TO CUSTOMERS
# ============================================================

segment_map = cluster_profile["Customer_Segment"].to_dict()

df["Customer_Segment"] = df["Cluster"].map(segment_map)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:15px 0 20px 0;
    ">
        <div style="
            font-size:45px;
        ">
            🎯
        </div>

        <div style="
            font-size:22px;
            font-weight:800;
            color:white;
        ">
            Customer Analytics
        </div>

        <div style="
            color:#9ca3af;
            font-size:13px;
            margin-top:5px;
        ">
            K-Means Segmentation
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Executive Dashboard",
            "👥 Customer Segmentation",
            "📊 EDA & Visualizations",
            "📐 Elbow Method",
            "🎯 Cluster Analysis",
            "💎 High-Value Customers",
            "💡 Marketing Insights",
            "📋 Customer Explorer"
        ]
    )

    st.markdown("---")

    st.markdown("### 📌 Project")

    st.write(
        "Customer Segmentation using "
        "K-Means Clustering"
    )

    st.markdown("---")

    st.caption("Built with Python + Streamlit")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Segmentation Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-driven customer intelligence using K-Means clustering'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    # KPI calculations

    total_customers = len(df)

    avg_age = df["Age"].mean()

    avg_income = df["Annual Income (k$)"].mean()

    avg_spending = df["Spending Score (1-100)"].mean()

    high_value_count = len(
        df[df["Customer_Segment"] == "High-Value Customers"]
    )

    # KPI row

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Customers</div>
                <div class="metric-value">{total_customers}</div>
                <div class="metric-description">Total customers</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg Age</div>
                <div class="metric-value">{avg_age:.1f}</div>
                <div class="metric-description">Years</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg Income</div>
                <div class="metric-value">${avg_income:.1f}K</div>
                <div class="metric-description">Annual income</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg Spending</div>
                <div class="metric-value">{avg_spending:.1f}</div>
                <div class="metric-description">Spending score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Value</div>
                <div class="metric-value">{high_value_count}</div>
                <div class="metric-description">Premium customers</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main charts

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Customer Segments</div>',
            unsafe_allow_html=True
        )

        segment_counts = (
            df["Customer_Segment"]
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.pie(
            segment_counts.values,
            labels=segment_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Customer Segment Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.markdown(
            '<div class="section-title">Income vs Spending</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.scatterplot(
            data=df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            hue="Customer_Segment",
            s=90,
            ax=ax
        )

        ax.set_title(
            "Customer Segmentation Map"
        )

        ax.grid(alpha=0.2)

        st.pyplot(fig)

        plt.close(fig)

    # Key insight

    st.markdown("""
    <div class="highlight">

        <div class="highlight-title">
            💎 Key Business Insight
        </div>

        <div class="highlight-text">

        Customer segmentation reveals distinct groups based on
        annual income and spending behavior. High-income,
        high-spending customers represent the most valuable
        segment and should receive premium offers, loyalty
        rewards, and personalized marketing campaigns.

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.markdown(
        '<div class="section-title">Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Customers are grouped using Annual Income and "
        "Spending Score with the K-Means clustering algorithm."
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Customer_Segment",
        style="Customer_Segment",
        s=130,
        ax=ax
    )

    centers = scaler.inverse_transform(
        kmeans.cluster_centers_
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=350,
        marker="X",
        c="black",
        label="Cluster Centers"
    )

    ax.set_title(
        "K-Means Customer Segmentation"
    )

    ax.grid(alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)

    st.markdown(
        '<div class="section-title">Segment Profiles</div>',
        unsafe_allow_html=True
    )

    display_profile = cluster_profile.copy()

    display_profile.columns = [
        "Customers",
        "Avg Age",
        "Avg Income (k$)",
        "Avg Spending",
        "Customer Segment"
    ]

    st.dataframe(
        display_profile,
        use_container_width=True
    )


# ============================================================
# EDA
# ============================================================

elif page == "📊 EDA & Visualizations":

    st.markdown(
        '<div class="section-title">'
        'Exploratory Data Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👤 Demographics",
            "💰 Income",
            "🛍️ Spending",
            "🔥 Correlation"
        ]
    )

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots()

            sns.countplot(
                data=df,
                x="Gender",
                ax=ax
            )

            ax.set_title(
                "Gender Distribution"
            )

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots()

            sns.histplot(
                data=df,
                x="Age",
                bins=15,
                kde=True,
                ax=ax
            )

            ax.set_title(
                "Age Distribution"
            )

            st.pyplot(fig)

            plt.close(fig)

    with tab2:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            data=df,
            x="Annual Income (k$)",
            bins=15,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Annual Income Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    with tab3:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.histplot(
            data=df,
            x="Spending Score (1-100)",
            bins=15,
            kde=True,
            ax=ax
        )

        ax.set_title(
            "Spending Score Distribution"
        )

        st.pyplot(fig)

        plt.close(fig)

    with tab4:

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        sns.heatmap(
            df[
                [
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)"
                ]
            ].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Feature Correlation"
        )

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# ELBOW METHOD
# ============================================================

elif page == "📐 Elbow Method":

    st.markdown(
        '<div class="section-title">'
        'Elbow Method'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Elbow Method helps determine a suitable "
        "number of clusters by analyzing K-Means inertia."
    )

    inertia = []

    K = range(1, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertia.append(model.inertia_)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        K,
        inertia,
        marker="o",
        linewidth=3
    )

    ax.axvline(
        5,
        linestyle="--",
        linewidth=2,
        label="Selected K = 5"
    )

    ax.set_title(
        "Elbow Method for Optimal K"
    )

    ax.set_xlabel(
        "Number of Clusters"
    )

    ax.set_ylabel(
        "Inertia"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    st.pyplot(fig)

    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Selected Clusters",
            "5"
        )

    with col2:

        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}"
        )

    st.success(
        "Five clusters provide meaningful customer "
        "segments based on income and spending behavior."
    )


# ============================================================
# CLUSTER ANALYSIS
# ============================================================

elif page == "🎯 Cluster Analysis":

    st.markdown(
        '<div class="section-title">'
        'Cluster Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["Cluster"].unique())
    )

    cluster_data = df[
        df["Cluster"] == selected_cluster
    ]

    segment_name = cluster_profile.loc[
        selected_cluster,
        "Customer_Segment"
    ]

    st.markdown(
        f"""
        <div class="highlight">

        <div class="highlight-title">
        Cluster {selected_cluster} — {segment_name}
        </div>

        <div class="highlight-text">

        This segment contains
        <b>{len(cluster_data)}</b> customers with an
        average annual income of
        <b>${cluster_data["Annual Income (k$)"].mean():.1f}K</b>
        and an average spending score of
        <b>{cluster_data["Spending Score (1-100)"].mean():.1f}</b>.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Customers",
            len(cluster_data)
        )

    with col2:

        st.metric(
            "Average Income",
            f"${cluster_data['Annual Income (k$)'].mean():.1f}K"
        )

    with col3:

        st.metric(
            "Average Spending",
            f"{cluster_data['Spending Score (1-100)'].mean():.1f}"
        )

    st.markdown(
        '<div class="section-title">'
        'Customer Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        cluster_data,
        use_container_width=True
    )


# ============================================================
# HIGH VALUE CUSTOMERS
# ============================================================

elif page == "💎 High-Value Customers":

    st.markdown(
        '<div class="section-title">'
        'High-Value Customer Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    high_value = df[
        df["Customer_Segment"] ==
        "High-Value Customers"
    ]

    percentage = (
        len(high_value) /
        len(df) *
        100
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "High-Value Customers",
            len(high_value)
        )

    with col2:

        st.metric(
            "Customer Percentage",
            f"{percentage:.1f}%"
        )

    with col3:

        st.metric(
            "Avg Spending Score",
            f"{high_value['Spending Score (1-100)'].mean():.1f}"
        )

    st.markdown("""
    <div class="highlight">

        <div class="highlight-title">
            💎 Premium Customer Segment
        </div>

        <div class="highlight-text">

        These customers have relatively high income and
        high spending behavior. They represent an important
        revenue opportunity and should receive premium
        experiences, loyalty rewards, personalized offers,
        and exclusive product recommendations.

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        high_value[
            [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)",
                "Cluster"
            ]
        ],
        use_container_width=True
    )

    csv = high_value.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download High-Value Customers",
        data=csv,
        file_name="high_value_customers.csv",
        mime="text/csv"
    )


# ============================================================
# MARKETING INSIGHTS
# ============================================================

elif page == "💡 Marketing Insights":

    st.markdown(
        '<div class="section-title">'
        'Marketing Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    strategies = {

        "High-Value Customers": (
            "Focus on premium products, exclusive "
            "offers, loyalty rewards, VIP programs, "
            "and personalized recommendations."
        ),

        "Potential Customers": (
            "Use personalized promotions, product "
            "recommendations, bundles, and incentives "
            "to increase spending."
        ),

        "Frequent / Budget Customers": (
            "Promote affordable products, discounts, "
            "value packs, and loyalty programs."
        ),

        "Low-Priority Customers": (
            "Use low-cost campaigns and occasional "
            "promotions while minimizing marketing costs."
        )
    }

    for segment in cluster_profile[
        "Customer_Segment"
    ].unique():

        count = len(
            df[df["Customer_Segment"] == segment]
        )

        st.markdown(
            f"""
            <div class="insight-card">

                <div class="insight-title">
                    {segment}
                </div>

                <div class="insight-text">

                    <b>Customers:</b> {count}<br><br>

                    <b>Recommended Strategy:</b><br>
                    {strategies.get(segment, "")}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class="highlight">

        <div class="highlight-title">
            🚀 Recommended Business Strategy
        </div>

        <div class="highlight-text">

        Prioritize high-value customers for retention,
        develop targeted campaigns for potential customers,
        use discounts to encourage budget customers, and
        apply cost-efficient campaigns to low-priority
        segments.

        This approach allows marketing resources to be
        allocated according to customer value and behavior.

        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

elif page == "📋 Customer Explorer":

    st.markdown(
        '<div class="section-title">'
        'Customer Data Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        segment_filter = st.multiselect(
            "Customer Segment",
            options=sorted(
                df["Customer_Segment"].unique()
            ),
            default=sorted(
                df["Customer_Segment"].unique()
            )
        )

    with col2:

        gender_filter = st.multiselect(
            "Gender",
            options=sorted(
                df["Gender"].unique()
            ),
            default=sorted(
                df["Gender"].unique()
            )
        )

    filtered_df = df[
        df["Customer_Segment"].isin(
            segment_filter
        )
        &
        df["Gender"].isin(
            gender_filter
        )
    ]

    st.markdown(
        f"""
        <div class="highlight">

        <div class="highlight-title">
            🔎 Filter Results
        </div>

        <div class="highlight-text">

        Showing <b>{len(filtered_df)}</b>
        customers.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Filtered Dataset",
        data=csv,
        file_name="customer_segments_filtered.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    Customer Segmentation Analytics Dashboard
    <br>
    K-Means Clustering • EDA • Customer Intelligence • Marketing Analytics

</div>
""", unsafe_allow_html=True)