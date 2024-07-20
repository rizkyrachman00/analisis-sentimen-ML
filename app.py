import streamlit as st
import joblib
import os

# Path ke file model dan vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

# Muat model dan vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Streamlit app
st.title("Sentiment Analysis")

message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message:
        data = [message]
        transformed_data = vectorizer.transform(data)
        prediction = model.predict(transformed_data)

        # Konversi prediksi menjadi label sentimen
        sentiment_label = ""
        if prediction[0] == 1:
            sentiment_label = "positif"
        elif prediction[0] == 2:
            sentiment_label = "netral"
        elif prediction[0] == 3:
            sentiment_label = "negatif"
        else:
            sentiment_label = "tidak dikenali"

        st.write(f"Predicted Sentiment: {sentiment_label}")
    else:
        st.write("Please enter a message.")
