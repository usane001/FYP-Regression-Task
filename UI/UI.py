
st.title('Intimacy predictor')
st.markdown("""This is a software tool created to read anything you input and predict the intimacy on a scale from 0 to 5. 

This task trains a RoBERTa model, a type of advanced tool designed to understand English text, on a dataset that's all in English. 
The RoBERTa model learns from the training part by recognizing patterns and figuring out how to make accurate predictions. 
After training, the model is then tested to see how well it can predict on new, unseen data. 
The results help us understand how good the model is. Once the training is done, the model is saved so it can be used later for more predictions or to help analyze other sets of English text.""")

st.caption('0 being red, which also means no intimacy, and 5 being green, which means the textual input is intimate.')


user_input = st.text_input("Input Data:", "Type here...")
def predict_intimacy(text):
    predictions, _ = model.predict([text])
    prediction_value = predictions[0] if predictions.ndim > 0 else predictions
    rounded_prediction = round(float(prediction_value), 1)
    return rounded_prediction

def get_color(value):
    score_percentage = value / 5.0
    red = 255 * (1 - score_percentage)
    green = 255 * score_percentage
    return f"rgb({int(red)}, {int(green)}, 0)"
        color = get_color(prediction)
    st.markdown(f'<div style="color: black; background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Predicted intimacy level: {prediction}</div>', unsafe_allow_html=True)
# to run this file please run the following code in the terminal "streamlit run path/to/file"