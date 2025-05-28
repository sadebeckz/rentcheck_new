import streamlit as st
import requests
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # ✅ Needed for unpickling the model

# ----------------------------
# 🔹 Load trained model
# ----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

model_columns = [
    'price', 'beds', 'baths', 'est_rent',
    'zipcode_33304', 'zipcode_33305', 'zipcode_33306',
    'zipcode_33308', 'zipcode_33309', 'zipcode_33311',
    'zipcode_33312', 'zipcode_33315', 'zipcode_33316'
]

# ----------------------------
# 🔹 Streamlit Setup
# ----------------------------
st.set_page_config(page_title="RentCheck Live + ML", layout="wide")
st.title("🏠 RentCheck: Real-Time Rental Investment Analyzer")

zillow_url = st.text_input("Paste a Zillow Property URL")

api_key = "d707ed68-8f48-4607-8de9-36b08277a9f1"  # Replace with your actual key

# ----------------------------
# 🔹 Zillow API Call
# ----------------------------
def get_property_data(zillow_url):
    endpoint = "https://api.hasdata.com/scrape/zillow/property"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }
    payload = {
        "url": zillow_url
    }
    response = requests.get(endpoint, headers=headers, params=payload)
    if response.status_code == 200:
        return response.json()
    return None

# ----------------------------
# 🔹 Analyze Button
# ----------------------------
if st.button("Analyze Property") and zillow_url:
    with st.spinner("Scraping Zillow..."):
        data = get_property_data(zillow_url)

    if data and data.get("property"):
        p = data["property"]
        price = p.get("price", 0)
        beds = p.get("beds", 0)
        baths = p.get("baths", 0)
        zip_code = str(p.get("address", {}).get("zipcode", "00000"))
        address = p.get("addressRaw", "Unknown Address")

        # Napkin Math
        est_rent = price * 0.009
        monthly_expenses = est_rent * 0.5
        loan_amount = price * 0.75
        mortgage_payment = (loan_amount / 100000) * 537
        cash_flow = est_rent - monthly_expenses - mortgage_payment
        cash_invested = price * 0.25
        annual_cash_flow = cash_flow * 12
        coc_return = (annual_cash_flow / cash_invested) * 100 if cash_invested else 0
        deal_rating = "✅ Good Buy" if cash_flow > 0 else "❌ Avoid"

        # ----------------------------
        # 🔹 ML Prediction
        # ----------------------------
        input_data = {col: 0 for col in model_columns}
        input_data['price'] = price
        input_data['beds'] = beds
        input_data['baths'] = baths
        input_data['est_rent'] = est_rent
        zip_col = f"zipcode_{zip_code}"
        if zip_col in input_data:
            input_data[zip_col] = 1

        df_input = pd.DataFrame([input_data])
        ml_prediction = model.predict(df_input)[0]
        ml_rating = "✅ ML Predicts Good Buy" if ml_prediction == 1 else "❌ ML Predicts Avoid"

        # ----------------------------
        # 🔹 Display Results
        # ----------------------------
        st.success("Property analyzed successfully!")
        st.markdown(f"### 📍 {address}")
        st.write(f"**Price:** ${price:,.0f}")
        st.write(f"**Beds/Baths:** {beds} / {baths}")
        st.write(f"**Estimated Rent:** ${est_rent:,.0f}")
        st.write(f"**Cash Flow:** ${cash_flow:,.2f}")
        st.write(f"**CoC Return:** {coc_return:.2f}%")
        st.write(f"**Napkin Math Rating:** {deal_rating}")
        st.write(f"**ML Prediction:** {ml_rating}")

        summary = pd.DataFrame({
            "Address": [address],
            "Price": [price],
            "Est. Rent": [est_rent],
            "Cash Flow": [cash_flow],
            "CoC Return (%)": [coc_return],
            "Napkin Rating": [deal_rating],
            "ML Prediction": [ml_rating]
        })
        st.dataframe(summary)
    else:
        st.warning("❌ Could not retrieve property info. Check URL or try again.")
