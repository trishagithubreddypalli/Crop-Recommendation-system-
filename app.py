import streamlit as st
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# Set page configuration
st.set_page_config(
    page_title="SmartCrop",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def load_model(modelfile):
    if not os.path.exists(modelfile):
        st.error(f"Model file {modelfile} does not exist.")
        return None
    try:
        with open(modelfile, "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except (EOFError, pickle.UnpicklingError) as e:
        st.error(f"Error loading the model: {e}")
        return None


def main():
    # Title and description
    st.markdown(
        """
        <div style="background-color: mediumseagreen; padding: 10px; border-radius: 10px">
            <h1 style="color: white; text-align: center;">SmartCrop: Intelligent Crop Recommendation 🌱</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Find out the most suitable crop to grow in your farm 👨‍🌾")

    # Input fields
    N = st.number_input("Nitrogen", 1, 10000)
    P = st.number_input("Phosphorus", 1, 10000)
    K = st.number_input("Potassium", 1, 10000)
    temp = st.number_input("Temperature (°C)", 0.0, 100.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("pH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0)

    # Prepare features for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Prediction button
    if st.button("Predict"):
        loaded_model = load_model("model.pkl")
        if loaded_model is not None:
            prediction = loaded_model.predict(single_pred)
            # Display prediction result
            st.write("## Results 🔍")
            st.success(
                f"{prediction.item().title()} are recommended by the A.I for your farm."
            )

    # Hide Streamlit menu
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
