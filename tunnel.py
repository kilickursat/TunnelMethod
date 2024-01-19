import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import builtins


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

# Set page config
st.set_page_config(page_title="Rock Mass Classification Analysis", layout="wide")

# Custom styles
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        padding-top: 2rem;
    }}
    .sidebar .sidebar-content {{
        background-color: white;
    }}
    .creator-name {{
        font-size: 20px;
        font-style: italic;
        text-align: center;
        margin-top: 20px;
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
    st.image("path_to_your_image.jpg", width=700)  # Add your image path
    st.markdown("### Features Description")
    st.write("Detailed information about the features of this application...")
    st.markdown('<p class="creator-name">Created by Kursat Kilic</p>', unsafe_allow_html=True)

def page_analysis():
    st.title("Interactive Rock Mass Analysis")
    with st.sidebar:
        st.header("Adjust Parameters")
        rmr = st.slider("Rock Mass Rating (RMR)", 0, 100, 50, help="Adjust the RMR value")
        rqd = st.slider("Rock Quality Designation (RQD) %", 0, 100, 75, help="Adjust the RQD percentage")
        gsi = st.slider("Geological Strength Index (GSI)", 0, 100, 65, help="Adjust the GSI value")
        ucs = st.slider("Unconfined Compressive Strength (UCS) MPa", 0, 200, 100, help="Adjust the UCS value in MPa")
        bts = st.slider("Brazilian Tensile Strength (BTS) MPa", 0, 50, 25, help="Adjust the BTS value in MPa")

    if st.sidebar.button("Analyze"):
        with st.spinner('Analyzing...'):
            prediction_encoded = model.predict(np.array([[rmr, rqd, gsi, ucs, bts]]))
            recommendation = label_encoder.inverse_transform(prediction_encoded)
            st.markdown(f"<h1 style='text-align: center; font-size: 40px;'><b>{recommendation[0]}</b></h1>", unsafe_allow_html=True)
            fig, ax = plt.subplots()
            # Add interactive 2D scene logic here
            # Example: ax.plot(sample_data_x, sample_data_y)
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
