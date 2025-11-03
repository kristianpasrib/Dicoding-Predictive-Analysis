import streamlit as st
import joblib, re, string

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict whether the sentiment is **positive** or **negative**.")

review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if review.strip():
        clean_text = preprocess(review)
        tfidf = vectorizer.transform([clean_text])
        prediction = model.predict(tfidf)[0]
        sentiment_label = label_encoder.inverse_transform([prediction])[0]
        if sentiment_label == 'positive':
            st.success("Prediction Result: **Positive**")
        else:
            st.error("Prediction Result: **Negative**")
    else:
        st.warning("Please enter a review before making a prediction.")