import streamlit as st
from simpletransformers.classification import ClassificationModel
import numpy as np


st.title('Intimacy predictor')
st.markdown("""
This is a software tool created to read anything you input and predict the intimacy on a scale from 1 to 5. 
This task trains a RoBERTa model, a type of advanced tool designed to understand English text, on a dataset that's all in English. 
...
""")

user_input = st.text_input("Input Data:", "")

@st.cache(allow_output_mutation=True)
def load_model():
    model = ClassificationModel('roberta', './model_output', use_cuda=False)
    return model

model = load_model()

def predict_intimacy(text):
    predictions, _ = model.predict([text])
    prediction_value = predictions[0] if predictions.ndim > 0 else predictions
    adjusted_prediction = max(1, round(float(prediction_value), 1))
    return adjusted_prediction
    
def color_picker(value):
    score_percentage = value / 5.0
    red = 255 * (1 - score_percentage)
    green = 255 * score_percentage
    return f"rgb({int(red)}, {int(green)}, 0)"
    
if st.button('Enter'):

    if user_input.strip():  
        with st.spinner('Processing...'):
            prediction = predict_intimacy(user_input)
            color = color_picker(prediction)
            st.markdown(f'<div style="color: black; background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Predicted intimacy level: {prediction}</div>', unsafe_allow_html=True)
    else:
        st.error('Please enter valid text for analysis.')
# to run this file please run the following code in the terminal "streamlit run path/to/file"
