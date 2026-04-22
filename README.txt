📊 Analysis on E-Commerce Market

A complete Data Science & Business Intelligence project focused on analyzing e-commerce transactional data to uncover customer behavior, forecast demand, optimize pricing, and automate customer segmentation using Machine Learning. This project transforms raw retail transaction data into actionable business insights for growth, retention, and operational efficiency.

🚀 Project Highlights

✅ End-to-End Data Analytics Pipeline
✅ Exploratory Data Analysis (EDA)
✅ Customer Segmentation using RFM + K-Means
✅ Demand Forecasting using Prophet
✅ Price Sensitivity Analysis
✅ Predictive Modeling (AutoML)
✅ Business Recommendations & Strategy Insights

📌 Problem Statement

Modern e-commerce businesses generate massive transactional data daily, but without proper analysis, valuable insights remain hidden.

This project solves key business problems such as:

Who are the most valuable customers?
Which products drive the highest revenue?
What seasonal trends affect sales?
How can future demand be predicted?
Which customers are likely to churn?
How should pricing be optimized?
🧠 Objectives
Analyze customer purchasing patterns
Discover top-performing products and markets
Segment customers into meaningful groups
Forecast future sales demand
Understand price elasticity of products
Build ML models for automated customer classification
Support data-driven decision making
📂 Dataset Information

Dataset used: Online Retail Dataset

Features:
Column Name	Description
InvoiceNo	Unique transaction ID
StockCode	Product code
Description	Product name
Quantity	Number of units sold
InvoiceDate	Date & time of transaction
UnitPrice	Product price
CustomerID	Unique customer ID
Country	Customer location
⚙️ Tech Stack
Programming Language
Python 🐍
Libraries Used
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Prophet
XGBoost
CatBoost
Tools
Jupyter Notebook
VS Code
GitHub
📈 Project Workflow
Data Collection
     ↓
Data Cleaning
     ↓
EDA & Visualization
     ↓
Feature Engineering
     ↓
Customer Segmentation
     ↓
Demand Forecasting
     ↓
Price Sensitivity Analysis
     ↓
Machine Learning Models
     ↓
Business Recommendations
🔍 Exploratory Data Analysis (EDA)

Key insights discovered:

🌍 Country-wise Sales
United Kingdom contributed the highest revenue.
Strong dominance of UK market.
🛍️ Top Products
REGENCY CAKESTAND 3 TIER was the top-selling product.
📅 Seasonal Trends
Sales peaked sharply in November due to holiday shopping season.
👥 Customer Segmentation (RFM + K-Means)

Customers were segmented using:

Recency → Last purchase date
Frequency → Number of purchases
Monetary → Total spend
Segments Identified:
Cluster	Segment	Strategy
2	Champions	Retain & Reward
1	Loyal Customers	Upsell & Cross-sell
0	At Risk	Re-activate
📊 Demand Forecasting

Using Prophet, future daily sales were forecasted.

Benefits:
Inventory planning
Better staffing decisions
Financial forecasting
Seasonal demand preparation
💰 Price Sensitivity Analysis

Studied relationship between:

Price ↑  → Demand ↓
Outcome:
Elastic products → Best for discounts/promotions
Inelastic products → Maintain premium pricing
🤖 Predictive Modeling

Built multiple classification models to predict customer segment automatically.

Models Used:
Logistic Regression
Random Forest
XGBoost
CatBoost
Gradient Boosting
Best Model:

🏆 Logistic Regression with excellent performance.

📌 Results
Metric	Outcome
Top Market	United Kingdom
Peak Sales Month	November
Best Product	REGENCY CAKESTAND 3 TIER
Best ML Model	Logistic Regression
Key Features	Recency, Monetary
📷 Sample Visualizations
Sales Trend Charts
Country Revenue Graphs
Top Product Charts
RFM Cluster Plot
Forecast Graph
Confusion Matrix

(Add screenshots in /images folder)

📁 Project Structure
Analysis-on-Ecommerce-Market/
│── data/
│   └── Online Retail.xlsx
│
│── notebooks/
│   └── analysis.ipynb
│
│── images/
│   └── charts.png
│
│── models/
│   └── trained_model.pkl
│
│── requirements.txt
│── README.md
│── app.py
▶️ Installation & Run
Clone Repository
git clone https://github.com/yourusername/Analysis-on-Ecommerce-Market.git
cd Analysis-on-Ecommerce-Market
Install Dependencies
pip install -r requirements.txt
Run Notebook / App
jupyter notebook

or

python app.py
💡 Business Recommendations
🎯 Customer Strategy
Reward loyal customers
Recover at-risk customers
Personalized marketing campaigns
📦 Inventory Strategy
Prepare high stock before November
Prioritize top-selling products
💵 Pricing Strategy
Discounts on elastic products
Premium pricing on inelastic goods
📚 Learning Outcomes

Through this project, I learned:

Real-world Data Cleaning
Business Analytics
Customer Segmentation
Time Series Forecasting
Machine Learning Pipelines
Model Evaluation
Strategic Decision Making
🙋‍♂️ Author

Aditya
🎓 MSc Data Science
📍 India

⭐ Support

If you found this project useful:

⭐ Star this repository
🍴 Fork it
📩 Connect for collaborations

📜 License

This project is for educational and portfolio purposes.
