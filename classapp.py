import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Classification App", page_icon="🏋️", layout="centered")

# Load trained k-NN model safely
try:
    with open("class.pkl", "rb") as file:
        knn_model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Dark Theme Styling
st.markdown("""
    <style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    .title { font-size: 28px; font-weight: bold; text-align: center; color: #61dafb; }
    .subtext { font-size: 18px; text-align: center; color: #CCCCCC; }
    .input-box { font-size: 20px; font-weight: bold; }
    .result-box { 
        padding: 15px; 
        border-radius: 8px; 
        background-color: #008000; /* Dark green for success */
        text-align: center; 
        font-size: 22px; 
        color: white;
        font-weight: bold;
    }
    .error-box {
        padding: 15px; 
        border-radius: 8px; 
        background-color: #FF0000; /* Dark red for errors */
        text-align: center; 
        font-size: 20px; 
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown('<p class="title">🏋️ Health Classification App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Enter your weight and height to determine your health status.</p>', unsafe_allow_html=True)

# User Input Fields (Side-by-side Layout)
col1, col2 = st.columns(2)
weight = col1.text_input("🔢 Enter Weight (kg):", placeholder="e.g., 60")
height = col2.text_input("📏 Enter Height (cm):", placeholder="e.g., 170")

# Prediction Button
if st.button("🔍 Predict Health Status"):
    if weight and height:
        try:
            # Convert inputs to numeric values
            weight = float(weight)
            height = float(height)

            # Fix: Ensure input uses Pandas DataFrame to match training data format
            input_data = pd.DataFrame([[weight, height]], columns=['Weight', 'Height'])
            prediction = knn_model.predict(input_data)[0]

            # Display result using Streamlit message styling
            st.markdown(f'<div class="result-box">🌟 Prediction: {prediction}</div>', unsafe_allow_html=True)
        except ValueError:
            st.markdown('<div class="error-box">❌ Please enter valid numeric values!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠️ Error in prediction: {e}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter both weight and height!")

# Footer
st.markdown("---")
st.markdown("📌 *Developed with ❤️ using Streamlit*")