
import streamlit as st
import joblib
import numpy as np

# Load the saved components
model = joblib.load('best_drug_rating_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le_drug = joblib.load('label_encoder_drug.pkl')
le_condition = joblib.load('label_encoder_condition.pkl')

# Streamlit UI
st.title("💊 Drug Rating Predictor")

drug_name = st.text_input("Drug Name", "Zoloft")
condition = st.text_input("Condition", "Depression")
review = st.text_area("Patient Review", "This drug worked really well for me and I felt better within days.")

if st.button("Predict Rating"):
    try:
        drug_enc = le_drug.transform([drug_name])[0]
    except:
        drug_enc = 0
    try:
        condition_enc = le_condition.transform([condition])[0]
    except:
        condition_enc = 0

    review_tfidf = tfidf.transform([review]).toarray()
    X_input = np.hstack((review_tfidf, [[drug_enc, condition_enc]]))
    predicted_rating = model.predict(X_input)[0]
    st.success(f"⭐ Predicted Rating: {predicted_rating:.2f}")
