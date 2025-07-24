import streamlit as st
import pandas as pd
import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Paths (consistent with train_bert_model.py)
model_dir = 'bert_model_disease_prediction'
label_encoder_path = 'label_encoder_disease.pkl'

# Streamlit app title and description
st.title("Disease Prediction from Symptoms")
st.write("""
Enter a list of symptoms (e.g., 'palpitations, sweating, fatigue') to predict the most likely disease.
The model provides the top-5 predicted diseases with their probabilities.
**Note**: The model has limited accuracy (~10%) due to a small dataset and many disease classes.
Predictions are for informational purposes only and should not be used for medical diagnosis.
""")

# Load model, tokenizer, and label encoder
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        label_encoder = joblib.load(label_encoder_path)
        return tokenizer, model, label_encoder
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None, None

tokenizer, model, label_encoder = load_model_and_tokenizer()

# Check if model loaded successfully
if tokenizer is None or model is None or label_encoder is None:
    st.stop()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Input form
with st.form(key='symptom_form'):
    symptoms = st.text_input("Enter symptoms (comma-separated, e.g., palpitations, sweating, fatigue):")
    submit_button = st.form_submit_button(label='Predict')

# Prediction function
def predict_disease(symptoms):
    if not symptoms.strip():
        return None, "Please enter at least one symptom."

    # Tokenize input
    MAX_LENGTH = 128
    encoded = tokenizer(
        symptoms,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_k_indices = np.argsort(probabilities)[-5:][::-1]  # Top-5 indices
        top_k_probs = probabilities[top_k_indices]
        top_k_labels = label_encoder.inverse_transform(top_k_indices)

    # Format results
    results = pd.DataFrame({
        'Disease': top_k_labels,
        'Probability': [f"{prob*100:.2f}%" for prob in top_k_probs]
    })
    return results, None

# Handle form submission
if submit_button:
    if symptoms:
        results, error = predict_disease(symptoms)
        if error:
            st.error(error)
        else:
            st.write("**Top-5 Predicted Diseases**")
            st.table(results)
    else:
        st.error("Please enter symptoms before predicting.")

# Footer
st.markdown("""
---
*Built with Streamlit and BioBERT. Model trained on a dataset of 800 samples with ~383 disease classes.*
""")