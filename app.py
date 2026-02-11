import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# ==============================
# Load Saved Artifacts
# ==============================

model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
sender_stats = joblib.load("sender_stats.pkl")

sender_cols = [
    "sender_total_prev",
    "sender_phish_prev",
    "sender_phish_rate"
]

# ==============================
# Prediction Function
# ==============================

def predict_email(text, sender):

    # Clean inputs
    text = text.lower().strip()
    sender = sender.lower().strip()

    # Transform text
    text_features = vectorizer.transform([text])

    # Get sender features
    sender_row = sender_stats[
        sender_stats["sender_norm"] == sender
    ]

    if len(sender_row) == 0:
        sender_features = np.array([[0, 0, 0]])
    else:
        sender_features = sender_row[sender_cols].values

    # Combine features
    combined_features = hstack([text_features, sender_features])

    # Predict
    prob = model.predict_proba(combined_features)[0][1]
    pred = model.predict(combined_features)[0]

    return pred, prob


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="Phishing Email Detector", layout="centered")

st.title("📧 Phishing Email Detection System")
st.markdown("Detect whether an email is phishing or legitimate using ML.")

email_text = st.text_area("Enter Email Content")

sender_input = st.text_input("Enter Sender Email (optional)")

if st.button("Analyze Email"):

    if email_text.strip() == "":
        st.warning("Please enter email content.")
    else:

        prediction, probability = predict_email(
            email_text,
            sender_input if sender_input else "unknown_sender"
        )

        if prediction == 1:
            st.error(f"⚠️ Prediction: Phishing ({probability:.4f})")
        else:
            st.success(f"✅ Prediction: Legitimate ({probability:.4f})")

        st.write("Probability of phishing:", round(probability, 4))
