import streamlit as st
import numpy as np
import pickle

# Load all files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.title("Crop Recommendation System")

# Inputs
N = st.number_input("Nitrogen", 0, 200, 98)
P = st.number_input("Phosphorus", 0, 200, 79)
K = st.number_input("Potassium", 0, 200, 50)
temp = st.number_input("Temperature", 0.0, 50.0, 25.34)
humidity = st.number_input("Humidity", 0.0, 100.0, 84.47)
ph = st.number_input("pH", 0.0, 14.0, 6.44)
rainfall = st.number_input("Rainfall", 0.0, 500.0, 91.06)


# Top-K function
def top_k_predict(model, X, encoder, k=5):
    probs = model.predict_proba(X)
    top_k_idx = np.argsort(probs, axis=1)[:, -k:][:, ::-1]

    results = []
    for idx in top_k_idx[0]:
        label = encoder.inverse_transform([idx])[0]
        prob = probs[0][idx]
        results.append((label, prob))

    return results


# Predict
if st.button("Recommend"):
    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    results = top_k_predict(model, input_scaled, encoder, 5)

    st.subheader("Top 5 Crop Recommendations:")
    for i, (crop, prob) in enumerate(results, 1):
        st.write(f"{i}. {crop} ({prob:.2f})")