# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

def main():
    st.set_page_config(page_title="Rock Mass Classification Analysis", layout="wide")
    
    pages = {
        "Information": page_information,
        "Analysis": page_analysis
    }

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", list(pages.keys()))

    pages[page]()

def page_information():
    st.title("Rock Mass Classification - Information")
    st.write("Detailed information about RMR, RQD, GSI, in-situ stress, hydrogeology, UCS, and BTS.")

def page_analysis():
    st.title("Interactive Rock Mass Analysis")
    st.sidebar.header("Adjust Parameters")
    
    rmr = st.sidebar.slider("Rock Mass Rating (RMR)", 0, 100, 50)
    rqd = st.sidebar.slider("Rock Quality Designation (RQD) %", 0, 100, 75)
    gsi = st.sidebar.slider("Geological Strength Index (GSI)", 0, 100, 65)
    ucs = st.sidebar.slider("Unconfined Compressive Strength (UCS) MPa", 0, 200, 100)
    bts = st.sidebar.slider("Brazilian Tensile Strength (BTS) MPa", 0, 50, 25)

    # Load the model and the label encoder
    model, label_encoder = joblib.load('tunneling_xgboost_model.pkl')

    input_features = np.array([[rmr, rqd, gsi, ucs, bts]])
    prediction_encoded = model.predict(input_features)
    recommendation = label_encoder.inverse_transform(prediction_encoded)  # Decode prediction
    st.write(f"Recommended Tunneling Method: {recommendation[0]}")

    fig = generate_stress_visualization(rmr, rqd, gsi, ucs, bts)
    st.pyplot(fig)

def generate_stress_visualization(rmr, rqd, gsi, ucs, bts):
    x = np.linspace(0, 10, 100)
    y = rmr * np.sin(rqd * x) + gsi * np.cos(ucs * x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Stress Distribution')
    plt.title('2D Stress Visualization around Tunnel')
    plt.xlabel('Distance (m)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    return plt

if __name__ == "__main__":
    main()
