# SmartCart – Retail Customer Purchase Prediction

## 📌 Project Overview

SmartCart is a Machine Learning based retail customer purchase prediction system. It predicts whether a customer is likely to make a purchase based on customer-related and website interaction features.

The application is built using Python, Pandas, NumPy, Scikit-learn, Pickle, and Streamlit.

## 🎯 Problem Statement

Retail businesses have large amounts of customer data, but it can be difficult to identify which customers are likely to make a purchase.

SmartCart uses Machine Learning to analyze customer information and predict the customer's purchase status.

## 🎯 Objective

- Predict customer purchase status using Machine Learning.
- Provide a simple and interactive web application.
- Help businesses understand customer purchase probability.
- Demonstrate an end-to-end Machine Learning project.

## 📊 Dataset

The project uses the `customer_purchase_data.csv` dataset.

The dataset contains 1500 customer records and 8 input features.

### Input Features

- Age
- Gender
- Annual Income
- Number of Purchases
- Product Category
- Time Spent on Website
- Loyalty Program
- Discounts Availed

### Target Variable

`PurchaseStatus`

## 🤖 Machine Learning

The trained Machine Learning model is saved as:

`model.pkl`

The feature column information used by the model is saved as:

`model_columns.pkl`

The Streamlit application loads these files and uses them to make predictions for new customer inputs.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- GitHub

## ✨ Application Features

- Interactive customer input form
- Customer purchase prediction
- Purchase probability display
- Business recommendation based on prediction probability
- Dataset overview
- Simple and user-friendly Streamlit interface

## 🔄 Project Workflow

```text
Customer Input
      ↓
Data Preprocessing
      ↓
Feature Alignment
      ↓
Trained ML Model
      ↓
Purchase Prediction
      ↓
Purchase Probability
      ↓
Business Recommendation
📁 Project Structure
SmartCart/
│
├── app.py
├── api.py
├── model.pkl
├── model_columns.pkl
├── customer_purchase_data.csv
├── requirements.txt
├── Retail_Purchase_Prediction.ipynb
└── README.md
🚀 How to Run Locally
1. Clone the Repository
git clone https://github.com/itsyogi01/SmartCart.git
2. Open the Project Folder
cd SmartCart
3. Install Dependencies
pip install -r requirements.txt
4. Run the Streamlit App
streamlit run app.py
☁️ Deployment

The project can be deployed using Streamlit Community Cloud.

Basic deployment steps:

Upload the project to GitHub.
Connect the GitHub repository with Streamlit Community Cloud.
Select the main branch.
Select app.py as the main application file.
Deploy the application.
🔮 Future Scope
Use larger real-world retail datasets.
Improve model performance with advanced algorithms.
Add customer segmentation.
Add sales and purchase analytics dashboards.
Add model performance monitoring.
Integrate the application with a real retail database
