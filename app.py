from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Path ke file model dan vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

# Muat model dan vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
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

        return render_template('index.html', prediction=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
