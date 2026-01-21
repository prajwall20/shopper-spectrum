import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS for Professional Styling
# --------------------------------------------------
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        text-align: center;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Segment badges */
    .segment-high-value {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .segment-regular {
        background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .segment-occasional {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .segment-at-risk {
        background: linear-gradient(135deg, #f85032 0%, #e73827 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Product recommendation card */
    .product-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .product-name {
        font-weight: bold;
        color: #333;
        font-size: 1.1rem;
    }
    
    .similarity-score {
        color: #667eea;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0 2rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00B4DB;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f8e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #56ab2f;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff4e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f85032;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🛒 Shopper Spectrum</h1>
        <p class="header-subtitle">AI-Powered Customer Intelligence & Product Discovery Platform</p>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Data & Model with Error Handling
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("data/cleaned_online_retail.csv")
    except:
        st.error("⚠️ Data file not found. Please ensure 'data/cleaned_online_retail.csv' exists.")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("models/rfm_kmeans.pkl")
    except:
        st.warning("⚠️ Model file not found. Using demo mode.")
        return None

# Load data with spinner
with st.spinner("🔄 Loading data and models..."):
    df = load_data()
    kmeans = load_model()

# --------------------------------------------------
# Sidebar Configuration
# --------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.title("📊 Dashboard")
    st.markdown("---")
    
    # Dataset Overview
    st.subheader("📈 Dataset Overview")
    total_transactions = len(df)
    total_customers = df['CustomerID'].nunique()
    total_products = df['Description'].nunique()
    total_revenue = df['TotalPrice'].sum()
    
    st.metric("Total Transactions", f"{total_transactions:,}")
    st.metric("Unique Customers", f"{total_customers:,}")
    st.metric("Unique Products", f"{total_products:,}")
    st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    
    st.markdown("---")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Information
    st.subheader("ℹ️ About")
    st.info("""
    This application uses:
    - **RFM Analysis** for customer segmentation
    - **Collaborative Filtering** for product recommendations
    - **Machine Learning** clustering algorithms
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit**")

# --------------------------------------------------
# RFM PREPARATION
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def prepare_rfm_data(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    if kmeans is not None:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        rfm['Cluster'] = kmeans.predict(rfm_scaled)
    else:
        # Demo mode - assign random clusters
        rfm['Cluster'] = np.random.randint(0, 4, size=len(rfm))
    
    return rfm

rfm = prepare_rfm_data(df)

# --------------------------------------------------
# Define Segment Labels and Characteristics
# --------------------------------------------------
def get_segment_info(cluster):
    segment_mapping = {
        0: {
            'label': 'High-Value Customer',
            'icon': '👑',
            'color': 'segment-high-value',
            'description': 'Our VIP customers! Recent buyers with high frequency and spending.',
            'characteristics': [
                '✓ Recent purchases (Low Recency)',
                '✓ Frequent orders (High Frequency)', 
                '✓ High spending (High Monetary)',
                '✓ Strong brand loyalty'
            ],
            'strategies': [
                '💎 Exclusive VIP benefits and early access',
                '🎁 Personalized thank you rewards',
                '👥 Dedicated customer support',
                '📧 Premium product recommendations'
            ]
        },
        1: {
            'label': 'Regular Customer',
            'icon': '⭐',
            'color': 'segment-regular',
            'description': 'Steady and reliable customers with consistent engagement.',
            'characteristics': [
                '✓ Moderate purchase recency',
                '✓ Consistent order frequency',
                '✓ Good lifetime value',
                '✓ Potential for growth'
            ],
            'strategies': [
                '🎯 Loyalty program enrollment',
                '📱 Regular product updates',
                '🎂 Birthday/anniversary offers',
                '💰 Referral incentives'
            ]
        },
        2: {
            'label': 'Occasional Customer',
            'icon': '🌱',
            'color': 'segment-occasional',
            'description': 'Growing customers with good conversion potential.',
            'characteristics': [
                '✓ Occasional purchases',
                '✓ Lower frequency',
                '✓ Moderate spending',
                '✓ Room for engagement'
            ],
            'strategies': [
                '📧 Re-engagement campaigns',
                '⏰ Limited-time offers',
                '🎁 Free shipping incentives',
                '💡 Personalized recommendations'
            ]
        },
        3: {
            'label': 'At-Risk Customer',
            'icon': '⚠️',
            'color': 'segment-at-risk',
            'description': 'Customers who haven\'t purchased recently and need attention.',
            'characteristics': [
                '⚠ High recency (Inactive)',
                '⚠ Low frequency',
                '⚠ Risk of churn',
                '⚠ Requires intervention'
            ],
            'strategies': [
                '🔥 Win-back campaigns with special discounts',
                '📋 Survey to understand needs',
                '💸 Aggressive re-engagement offers',
                '📞 Direct outreach attempts'
            ]
        }
    }
    return segment_mapping.get(cluster, segment_mapping[0])

# --------------------------------------------------
# Main Application Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["👤 Customer Segmentation", "🛍️ Product Recommendations", "📊 Analytics Dashboard"])

# --------------------------------------------------
# TAB 1: CUSTOMER SEGMENTATION
# --------------------------------------------------
with tab1:
    st.markdown("## 🎯 Customer Segmentation Analysis")
    st.markdown("Analyze customer behavior using RFM (Recency, Frequency, Monetary) metrics to identify valuable customer segments.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Customer Lookup")
        
        # Create two input methods
        input_method = st.radio(
            "Choose input method:",
            ["Search by Customer ID", "Input RFM Values Manually"],
            horizontal=True
        )
        
        if input_method == "Search by Customer ID":
            # Customer ID input
            customer_id = st.number_input(
                "Enter Customer ID",
                min_value=int(rfm.index.min()),
                max_value=int(rfm.index.max()),
                step=1,
                help="Enter a valid customer ID to analyze their segment"
            )
            
            if st.button("🔍 Analyze Customer", use_container_width=True):
                if customer_id in rfm.index:
                    customer_data = rfm.loc[customer_id]
                    cluster = int(customer_data['Cluster'])
                    segment_info = get_segment_info(cluster)
                    
                    # Display segment badge
                    st.markdown(f"""
                        <div class="{segment_info['color']}">
                            {segment_info['icon']} {segment_info['label']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # RFM Metrics Display
                    st.markdown("### 📊 Customer RFM Metrics")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            "Recency (Days)",
                            f"{int(customer_data['Recency'])}",
                            delta=None,
                            help="Days since last purchase"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Frequency",
                            f"{int(customer_data['Frequency'])}",
                            delta=None,
                            help="Number of purchases"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Monetary (£)",
                            f"£{customer_data['Monetary']:,.2f}",
                            delta=None,
                            help="Total spending"
                        )
                    
                    # Segment Details
                    st.markdown("### 📝 Segment Description")
                    st.markdown(f"<div class='info-box'><strong>{segment_info['icon']} {segment_info['description']}</strong></div>", unsafe_allow_html=True)
                    
                    # Characteristics
                    st.markdown("### 🎯 Customer Characteristics")
                    for char in segment_info['characteristics']:
                        st.markdown(f"- {char}")
                    
                    # Recommended Strategies
                    st.markdown("### 💡 Recommended Marketing Strategies")
                    for strategy in segment_info['strategies']:
                        st.markdown(f"- {strategy}")
                    
                    # Visual comparison
                    st.markdown("### 📈 Performance vs Segment Average")
                    cluster_avg = rfm[rfm['Cluster'] == cluster].mean()
                    
                    comparison_fig = go.Figure()
                    metrics = ['Recency', 'Frequency', 'Monetary']
                    customer_values = [customer_data['Recency'], customer_data['Frequency'], customer_data['Monetary']]
                    avg_values = [cluster_avg['Recency'], cluster_avg['Frequency'], cluster_avg['Monetary']]
                    
                    comparison_fig.add_trace(go.Bar(
                        name='Customer',
                        x=metrics,
                        y=customer_values,
                        marker_color='#667eea'
                    ))
                    
                    comparison_fig.add_trace(go.Bar(
                        name='Segment Average',
                        x=metrics,
                        y=avg_values,
                        marker_color='#764ba2'
                    ))
                    
                    comparison_fig.update_layout(
                        title='Customer vs Segment Average',
                        barmode='group',
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(comparison_fig, use_container_width=True)
                    
                else:
                    st.error("❌ Customer ID not found. Please enter a valid Customer ID.")
        
        else:
            # Manual RFM input
            st.markdown("### 📝 Enter RFM Values")
            
            recency_input = st.number_input(
                "Recency (Days since last purchase)",
                min_value=0,
                max_value=500,
                value=30,
                help="How many days since the customer's last purchase?"
            )
            
            frequency_input = st.number_input(
                "Frequency (Number of purchases)",
                min_value=1,
                max_value=500,
                value=10,
                help="How many times has the customer made a purchase?"
            )
            
            monetary_input = st.number_input(
                "Monetary (Total spending in £)",
                min_value=0.0,
                value=500.0,
                step=10.0,
                help="What is the total amount the customer has spent?"
            )
            
            if st.button("🎯 Predict Segment", use_container_width=True):
                # Predict cluster
                if kmeans is not None:
                    scaler = StandardScaler()
                    scaler.fit(rfm[['Recency', 'Frequency', 'Monetary']])
                    input_scaled = scaler.transform([[recency_input, frequency_input, monetary_input]])
                    predicted_cluster = kmeans.predict(input_scaled)[0]
                else:
                    # Demo prediction logic
                    if recency_input <= 30 and frequency_input >= 15 and monetary_input >= 1000:
                        predicted_cluster = 0
                    elif recency_input <= 60 and frequency_input >= 8 and monetary_input >= 500:
                        predicted_cluster = 1
                    elif frequency_input >= 3:
                        predicted_cluster = 2
                    else:
                        predicted_cluster = 3
                
                segment_info = get_segment_info(predicted_cluster)
                
                # Display results
                st.markdown(f"""
                    <div class="{segment_info['color']}">
                        {segment_info['icon']} Predicted Segment: {segment_info['label']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Show metrics
                st.markdown("### 📊 Input RFM Metrics")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Recency", f"{recency_input} days")
                
                with metric_col2:
                    st.metric("Frequency", f"{frequency_input} orders")
                
                with metric_col3:
                    st.metric("Monetary", f"£{monetary_input:,.2f}")
                
                # Show segment info
                st.markdown("### 📝 Segment Description")
                st.markdown(f"<div class='info-box'><strong>{segment_info['description']}</strong></div>", unsafe_allow_html=True)
                
                st.markdown("### 💡 Recommended Strategies")
                for strategy in segment_info['strategies']:
                    st.markdown(f"- {strategy}")
    
    with col2:
        st.markdown("### 📚 Understanding RFM")
        
        st.markdown("""
        <div class='info-box'>
            <h4>What is RFM Analysis?</h4>
            <p>RFM is a marketing analysis technique that segments customers based on three key metrics:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🕐 Recency (R)**
        - How recently did the customer make a purchase?
        - Lower is better (recent customers)
        
        **🔄 Frequency (F)**
        - How often does the customer buy?
        - Higher is better (loyal customers)
        
        **💰 Monetary (M)**
        - How much does the customer spend?
        - Higher is better (valuable customers)
        """)
        
        st.markdown("### 📊 Segment Distribution")
        
        # Cluster distribution
        cluster_counts = rfm['Cluster'].value_counts().sort_index()
        segment_names = [get_segment_info(i)['label'] for i in range(len(cluster_counts))]
        
        fig_dist = px.pie(
            values=cluster_counts.values,
            names=segment_names,
            title='Customer Distribution by Segment',
            color_discrete_sequence=['#667eea', '#00B4DB', '#56ab2f', '#f85032']
        )
        fig_dist.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_dist, use_container_width=True)

# --------------------------------------------------
# TAB 2: PRODUCT RECOMMENDATIONS
# --------------------------------------------------
with tab2:
    st.markdown("## 🛍️ Smart Product Recommendations")
    st.markdown("Discover similar products based on collaborative filtering and customer purchase patterns.")
    
    # Prepare recommendation data
    @st.cache_data(show_spinner=False)
    def prepare_recommendations(df):
        pivot_table = df.pivot_table(
            index='CustomerID',
            columns='Description',
            values='Quantity',
            aggfunc='sum',
            fill_value=0
        )
        
        item_similarity = cosine_similarity(pivot_table.T)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=pivot_table.columns,
            columns=pivot_table.columns
        )
        
        return pivot_table, item_similarity_df
    
    pivot_table, item_similarity_df = prepare_recommendations(df)
    product_list = sorted(pivot_table.columns.tolist())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Product Search")
        
        # Search functionality
        search_term = st.text_input(
            "Search for a product",
            placeholder="e.g., WHITE HANGING HEART T-LIGHT HOLDER",
            help="Type to search for products"
        )
        
        # Filter products based on search
        if search_term:
            filtered_products = [p for p in product_list if search_term.upper() in p.upper()]
            if filtered_products:
                st.success(f"✅ Found {len(filtered_products)} matching products")
            else:
                st.warning("⚠️ No products found. Try a different search term.")
                filtered_products = product_list[:20]  # Show top 20 as default
        else:
            filtered_products = product_list[:20]  # Show top 20 as default
        
        selected_product = st.selectbox(
            "Select a product for recommendations",
            filtered_products,
            help="Choose a product to get similar recommendations"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            help="How many similar products would you like to see?"
        )
        
        if st.button("🔍 Get Recommendations", use_container_width=True):
            with st.spinner("🔄 Finding similar products..."):
                # Get recommendations
                scores = item_similarity_df[selected_product].sort_values(ascending=False)
                recommendations = scores.iloc[1:num_recommendations+1]
                
                st.markdown("### 🎯 Recommended Products")
                st.markdown(f"<div class='success-box'><strong>✅ Found {len(recommendations)} recommendations for: {selected_product}</strong></div>", unsafe_allow_html=True)
                
                # Display recommendations
                for idx, (prod, score) in enumerate(recommendations.items(), 1):
                    st.markdown(f"""
                        <div class='product-card'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <div>
                                    <span style='font-size: 1.5rem; margin-right: 0.5rem;'>#{idx}</span>
                                    <span class='product-name'>{prod}</span>
                                </div>
                                <div>
                                    <span class='similarity-score'>Similarity: {score:.2%}</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("### 📊 Recommendation Similarity Scores")
                fig_rec = go.Figure(go.Bar(
                    x=[f"#{i+1}" for i in range(len(recommendations))],
                    y=recommendations.values * 100,
                    text=[f"{score:.1%}" for score in recommendations.values],
                    textposition='auto',
                    marker_color='#667eea',
                    hovertemplate='<b>%{text}</b><br>Product: ' + 
                                 recommendations.index.tolist()[0] + '<extra></extra>'
                ))
                
                fig_rec.update_layout(
                    title='Product Similarity Scores',
                    xaxis_title='Recommendation Rank',
                    yaxis_title='Similarity Score (%)',
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_rec, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 How It Works")
        
        st.markdown("""
        <div class='info-box'>
            <h4>Collaborative Filtering</h4>
            <p>Our recommendation system analyzes customer purchase patterns to find products that are frequently bought together.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **🔍 Process:**
        
        1. **Analyze** customer purchase history
        2. **Calculate** product similarities using cosine similarity
        3. **Identify** products with similar buying patterns
        4. **Recommend** top N similar products
        
        **🎯 Benefits:**
        - Personalized recommendations
        - Increase cross-selling opportunities
        - Improve customer experience
        - Boost average order value
        """)
        
        st.markdown("### 📈 Popular Products")
        
        # Show top products
        top_products = df['Description'].value_counts().head(5)
        
        for idx, (prod, count) in enumerate(top_products.items(), 1):
            st.markdown(f"**{idx}.** {prod}")
            st.progress(count / top_products.max())
            st.caption(f"{count:,} purchases")

# --------------------------------------------------
# TAB 3: ANALYTICS DASHBOARD
# --------------------------------------------------
with tab3:
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("Comprehensive overview of customer segments and business metrics.")
    
    # Overall metrics
    st.markdown("### 🎯 Key Performance Indicators")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    avg_recency = rfm['Recency'].mean()
    avg_frequency = rfm['Frequency'].mean()
    avg_monetary = rfm['Monetary'].mean()
    total_customers = len(rfm)
    
    with kpi_col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Customers</div>
                <div class='metric-value'>{total_customers:,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Avg Recency</div>
                <div class='metric-value'>{avg_recency:.0f}</div>
                <div class='metric-label'>days</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Avg Frequency</div>
                <div class='metric-value'>{avg_frequency:.1f}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Avg Monetary</div>
                <div class='metric-value'>£{avg_monetary:,.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------------------
    # CLUSTER CLASSIFICATION (MANDATORY)
    # --------------------------------------------------
    st.markdown("### 🧠 Cluster Classification Logic")

    cluster_stats = rfm.groupby('Cluster').mean()

    def classify_cluster(row):
        if row['Monetary'] >= cluster_stats['Monetary'].quantile(0.75) and \
           row['Frequency'] >= cluster_stats['Frequency'].quantile(0.75) and \
           row['Recency'] <= cluster_stats['Recency'].quantile(0.25):
            return "High Value Customers"
        elif row['Recency'] >= cluster_stats['Recency'].quantile(0.75):
            return "At Risk Customers"
        elif row['Monetary'] >= cluster_stats['Monetary'].quantile(0.40):
            return "Medium Value Customers"
        else:
            return "Low Value Customers"

    cluster_stats['Customer_Type'] = cluster_stats.apply(classify_cluster, axis=1)

    st.dataframe(
        cluster_stats[['Recency', 'Frequency', 'Monetary', 'Customer_Type']]
        .round(2)
        .rename_axis("Cluster ID"),
        use_container_width=True
    )

    # --------------------------------------------------
    # NUMBER OF CUSTOMERS PER CLUSTER
    # --------------------------------------------------
    st.markdown("### 👥 Number of Customers per Cluster")

    cluster_counts = rfm['Cluster'].value_counts().sort_index()

    cluster_count_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Number of Customers": cluster_counts.values,
        "Customer Type": cluster_stats.loc[cluster_counts.index, 'Customer_Type'].values
    })

    st.dataframe(cluster_count_df, use_container_width=True)

    # --------------------------------------------------
    # CLUSTER DISTRIBUTION VISUALIZATION
    # --------------------------------------------------
    st.markdown("### 📊 Cluster Distribution")

    fig_cluster_bar = px.bar(
        cluster_count_df,
        x="Customer Type",
        y="Number of Customers",
        color="Customer Type",
        text="Number of Customers",
        title="Customer Distribution by Cluster Type",
        color_discrete_sequence=['#667eea', '#00B4DB', '#56ab2f', '#f85032']
    )

    fig_cluster_bar.update_layout(
        showlegend=False,
        height=450
    )

    st.plotly_chart(fig_cluster_bar, use_container_width=True)

    # --------------------------------------------------
    # RFM CLUSTER SCATTER PLOT
    # --------------------------------------------------
    st.markdown("### 📈 RFM Cluster Visualization")

    rfm_plot_df = rfm.copy()
    rfm_plot_df['Customer Type'] = rfm_plot_df['Cluster'].map(
        cluster_stats['Customer_Type']
    )

    fig_scatter = px.scatter(
        rfm_plot_df,
        x="Recency",
        y="Monetary",
        color="Customer Type",
        size="Frequency",
        hover_data=["Frequency"],
        title="Customer Segmentation based on RFM",
        color_discrete_sequence=['#667eea', '#00B4DB', '#56ab2f', '#f85032']
    )

    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --------------------------------------------------
    # FINAL NOTES
    # --------------------------------------------------
    st.markdown("""
    ### ✅ Key Takeaways
    - Customers are **clearly segmented** using RFM-based KMeans clustering.
    - Each cluster is **classified into a business-meaningful category**.
    - The segmentation can be used for **targeted marketing and retention strategies**.
    """)
