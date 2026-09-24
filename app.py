import streamlit as st
import pandas as pd
import requests


# =========================
# FASTAPI URL
# =========================

API_URL = "https://smartcart-u41g.onrender.com"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "customer_purchase_data.csv"
)


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="SmartCart",
    page_icon="🛒",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title("🛒 SmartCart")

st.subheader(
    "Retail Customer Purchase Prediction"
)

st.write(
    "Predict whether a customer is likely "
    "to purchase a product using Machine Learning."
)


# =========================
# DATASET INFO
# =========================

st.write("## 📊 Dataset Information")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Customers",
        len(df)
    )

with c2:
    st.metric(
        "Total Features",
        len(df.columns) - 1
    )

with c3:
    st.metric(
        "Target",
        "PurchaseStatus"
    )


# =========================
# CUSTOMER INPUT
# =========================

st.write("---")
st.header("👤 Customer Details")

features = [
    col for col in df.columns
    if col != "PurchaseStatus"
]

customer_data = {}

cols = st.columns(3)

for i, column in enumerate(features):

    with cols[i % 3]:

        # =========================
        # GENDER
        # =========================

        if column == "Gender":

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            customer_data[column] = (
                1 if gender == "Male" else 0
            )


        # =========================
        # PRODUCT CATEGORY
        # =========================

        elif column == "ProductCategory":

            category_map = {
                "Electronics": 1,
                "Clothing": 2,
                "Groceries": 3,
                "Home & Kitchen": 4,
                "Other": 5
            }

            category = st.selectbox(
                "Product Category",
                list(category_map.keys())
            )

            customer_data[column] = category_map[category]


        # =========================
        # LOYALTY PROGRAM
        # =========================

        elif column == "LoyaltyProgram":

            loyalty = st.selectbox(
                "Loyalty Program",
                ["No", "Yes"]
            )

            customer_data[column] = (
                1 if loyalty == "Yes" else 0
            )


        # =========================
        # DISCOUNTS AVAILED
        # =========================

        elif column == "DiscountsAvailed":

            discount = st.selectbox(
                "Discounts Availed",
                ["No", "Yes"]
            )

            customer_data[column] = (
                1 if discount == "Yes" else 0
            )


        # =========================
        # NUMERIC FEATURES
        # =========================

        else:

            minimum = float(
                df[column].min()
            )

            maximum = float(
                df[column].max()
            )

            default = float(
                df[column].median()
            )

            customer_data[column] = st.number_input(
                column,
                min_value=minimum,
                max_value=maximum,
                value=default
            )


# =========================
# PREDICT
# =========================

if st.button(
    "🚀 Predict Purchase",
    use_container_width=True
):

    try:

        response = requests.post(
            API_URL,
            json={
                "data": customer_data
            },
            timeout=60
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            probability = (
                result["purchase_probability"] / 100
            )


            # =========================
            # RESULT
            # =========================

            st.write("---")

            st.header("🎯 Prediction Result")

            c1, c2 = st.columns(2)

            with c1:

                if prediction == 1:

                    st.success(
                        "🟢 Likely to Purchase"
                    )

                else:

                    st.error(
                        "🔴 Unlikely to Purchase"
                    )

            with c2:

                st.metric(
                    "Purchase Probability",
                    f"{probability * 100:.2f}%"
                )

            st.progress(
                float(probability)
            )


            # =========================
            # BUSINESS RECOMMENDATION
            # =========================

            st.write(
                "### 💡 Business Recommendation"
            )

            if probability >= 0.70:

                st.success(
                    "High purchase intent → "
                    "Show personalized offers "
                    "and product recommendations."
                )

            elif probability >= 0.40:

                st.info(
                    "Medium purchase intent → "
                    "Show product recommendations "
                    "or limited-time offers."
                )

            else:

                st.warning(
                    "Low purchase intent → "
                    "Focus on customer engagement."
                )

        else:

            st.error(
                f"❌ FastAPI Error: {response.status_code}"
            )

            st.write(response.text)

    except requests.exceptions.RequestException as e:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.write(e)
