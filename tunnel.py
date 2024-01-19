import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import builtins
# Set page config
st.set_page_config(page_title="Rock Mass Classification Analysis", layout="wide")
# Custom hash function for dict objects
def hash_dict(obj):
    try:
        return hash(frozenset(obj.items()))
    except TypeError:
        # For unhashable values, hash their string representation
        return hash(frozenset((k, repr(v)) for k, v in obj.items()))

@st.cache_data(hash_funcs={builtins.dict: hash_dict})
def load_model():
    model, label_encoder = joblib.load('tunneling_xgboost_model.pkl')
    return model, label_encoder

model, label_encoder = load_model()



# Custom color scheme
primaryColor = "#E694FF"
backgroundColor = "#f0f2f6"
secondaryBackgroundColor = "#e8eaf6"
textColor = "#262730"
font = "sans serif"

# Custom styles
st.markdown(
    f"""
    <style>
    .reportview-container {{
        font-family: {font};
        background-color: {backgroundColor};
    }}
    .sidebar .sidebar-content {{
        background-color: {secondaryBackgroundColor};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Main app function
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Information": page_information,
        "Analysis": page_analysis,
        "About": page_about
    }
    page = st.sidebar.radio("Select a Page", list(pages.keys()), format_func=lambda x: x + " 📄")
    pages[page]()

def page_information():
    st.title("Rock Mass Classification - Information")
    st.write("Detailed information about RMR, RQD, GSI, in-situ stress, hydrogeology, UCS, and BTS.")

def page_analysis():
    st.title("Interactive Rock Mass Analysis")
    with st.sidebar:
        st.header("Adjust Parameters")
        rmr = st.slider("Rock Mass Rating (RMR)", 0, 100, 50, help="Adjust the RMR value")
        rqd = st.slider("Rock Quality Designation (RQD) %", 0, 100, 75, help="Adjust the RQD percentage")
        gsi = st.slider("Geological Strength Index (GSI)", 0, 100, 65, help="Adjust the GSI value")
        ucs = st.slider("Unconfined Compressive Strength (UCS) MPa", 0, 200, 100, help="Adjust the UCS value in MPa")
        bts = st.slider("Brazilian Tensile Strength (BTS) MPa", 0, 50, 25, help="Adjust the BTS value in MPa")

    # Prediction and visualization
    if st.sidebar.button("Analyze"):
        with st.spinner('Analyzing...'):
            prediction_encoded = model.predict(np.array([[rmr, rqd, gsi, ucs, bts]]))
            recommendation = label_encoder.inverse_transform(prediction_encoded)
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

def page_about():
    st.title("About this App")
    st.write("This app provides a comprehensive analysis of rock mass classifications using advanced machine learning techniques.")

if __name__ == "__main__":
    main()
