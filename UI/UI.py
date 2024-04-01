import streamlit as st
from simpletransformers.model import ClassificationModel
import numpy as np


st.title('Intimacy predictor')
st.markdown("""
This tool is designed to analyze the text you provide and offers an intimacy rating on a scale from 1 to 5.
Any information you enter, personal or otherwise, will not be saved or stored in any form.
This task trains a RoBERTa model, a type of machine learning model designed to understand English text.
The preciseness of predictions made by this tool reflects how well the RoBERTa model has been trained to identify different levels of intimacy in text. 
Through this, we can better understand the complexity of emotional communication and expressions in written communication and the model's ability to accurately capture them""")
st.caption('1 being red, which means the lowest level of intimacy, and 5 being green, which means the textual input is highly intimate.')

user_input = st.text_input("Input Data:", placeholder="Type here...")

@st.cache_resource
def load_model(): 
    try:
        model = ClassificationModel('roberta', './model_output', use_cuda=False, num_labels=1, args={'regression': True})
        return model
    except Exception as e:
        st.error('Failed to load the model.check the model path and files.')
        st.stop()
        
model = load_model()

def intimacy_prediction(text):
    try:
        predictions, _ = model.predict([text])
        prediction_value = predictions[0] if predictions.ndim > 0 else predictions
        adjusted_prediction = max(1, round(float(prediction_value), 1))
        return adjusted_prediction
    except Exception as e:
        st.error('Prediction failed. Please try again.')
        st.stop()

def color_picker(value):
    score_percentage = (value - 1) / 4.0  
    red = 255 * (1 - score_percentage)
    green = 255 * score_percentage
    return f"rgb({int(red)}, {int(green)}, 0)"


if st.button('Enter'):
    if user_input.strip():
        with st.spinner('Processing...'):
            prediction = intimacy_prediction(user_input)
            color = color_picker(prediction)
            st.markdown(f'<div style="color: black; background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Predicted intimacy level: {prediction}</div>', unsafe_allow_html=True)
    else:
        st.error('Please enter some text.')
