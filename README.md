# 🛒 Shopper Spectrum  
### Customer Segmentation & Product Recommendation System

🔗 **Live App**: https://shopper-spectrum-online.streamlit.app/ 
🔗 **GitHub Repository**: https://github.com/prajwall20/shopper-spectrum  

---

## 📌 Project Overview

**Shopper Spectrum** is an end-to-end data science and machine learning project focused on **customer behavior analysis** in an e-commerce setting.  
The project uses **RFM (Recency, Frequency, Monetary) analysis** and **unsupervised learning** to segment customers, and **collaborative filtering** to recommend products.

A fully interactive **Streamlit web application** is built and deployed to demonstrate the results in real time.

---

## 🎯 Objectives

- Segment customers based on purchasing behavior
- Identify high-value, medium-value, low-value, and at-risk customers
- Build a product recommendation system
- Deploy an interactive analytics dashboard
- Make the solution production-ready and shareable

---

## 📊 Dataset

- **Source**: Online Retail Transaction Dataset  
- **Type**: Transactional e-commerce data  
- **Key Fields**:
  - InvoiceNo
  - InvoiceDate
  - CustomerID
  - Description
  - Quantity
  - UnitPrice
  - Country

---

## 🧠 Methodology

### 1. Data Preprocessing
- Removed missing CustomerID records
- Filtered cancelled and invalid transactions
- Removed duplicates
- Created `TotalPrice` feature
- Saved cleaned dataset for reuse

### 2. Exploratory Data Analysis (EDA)
- Country-wise and product-wise analysis
- Transaction value distributions
- Time-based sales trends
- Correlation analysis of numerical features

### 3. RFM Analysis
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending

### 4. Customer Segmentation
- Applied **KMeans clustering**
- Optimal clusters selected using Elbow Method
- Clusters classified into:
  - High Value Customers
  - Medium Value Customers
  - Low Value Customers
  - At-Risk Customers

### 5. Product Recommendation System
- Item-based collaborative filtering
- Cosine similarity on customer–product matrix
- Recommends similar products based on purchase patterns

---

## 🖥️ Streamlit Web Application

The deployed Streamlit app includes:

- 📊 Customer segmentation dashboard
- 🔍 Customer lookup by ID
- 🔮 Segment prediction using custom RFM inputs
- 🛍️ Product recommendation engine
- 📈 Interactive charts and analytics
- 💡 Business-oriented insights for each segment

---

## 📁 Project Structure

```text
shopper_spectrum/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
│
├── data/
│   ├── online_retail.csv
│   └── cleaned_online_retail.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_rfm_clustering.ipynb
│   └── 05_recommendation_system.ipynb
│
├── models/
│   └── rfm_kmeans.pkl
│
└──.gitignore
```

## ⚙️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Plotly**
- **Streamlit**
- **Git & GitHub**

---

## 👤 Author

**Prajwal**  
📧 Email: prajjuprajwal617@gmail.com  
🔗 GitHub: https://github.com/prajwall20  

---

## © Copyright

© 2026 Prajwal. All rights reserved.

This project is created for **educational and portfolio purposes**.  
You are free to **view, fork, and reference** this project with proper attribution.

Unauthorized commercial use or redistribution without permission is prohibited.




