import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack

# Load artifacts
model = joblib.load(model.pkl)
vectorizer = joblib.load(tfidf_vectorizer.pkl)
sender_stats = joblib.load(sender_stats.pkl)

st.title(Phishing Email Detector)

st.write(Enter email content and sender address.)

email_text = st.text_area(Email Text)
sender_input = st.text_input(Sender Email)

def prepare_sender_features(sender)
    row = sender_stats[sender_stats[sender_norm] == sender]

    if len(row) == 0
        return pd.DataFrame([{
            sender_total_prev 0,
            sender_phish_prev 0,
            sender_phish_rate 0
        }])
    
    return row[[
        sender_total_prev,
        sender_phish_prev,
        sender_phish_rate
    ]]

if st.button(Predict)
    if email_text.strip() == 
        st.warning(Please enter email text.)
    else
        text_features = vectorizer.transform([email_text])
        sender_features = prepare_sender_features(sender_input).values
        
        combined = hstack([text_features, sender_features])
        
        prob = model.predict_proba(combined)[0][1]
        pred = model.predict(combined)[0]
        
        if pred == 1
            st.error(fPrediction Phishing ({prob.4f}))
        else
            st.success(fPrediction Legitimate ({1-prob.4f}))
