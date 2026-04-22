import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Set page configuration
st.set_page_config(page_title="Toxicity Detector", layout="centered")

@st.cache_resource
def load_model_and_tokenizer():
    """Loads the model and tokenizer only once to prevent UI freezing."""
    try:
        model = tf.keras.models.load_model('toxicity_model.h5')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Critical Failure loading assets: {e}")
        return None, None

def clean_input_text(text):
    """Must mirror the exact cleaning process from training."""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load assets
model, tokenizer = load_model_and_tokenizer()
labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

st.title("Comment Toxicity Detection")
st.markdown("Enter a comment below to analyze its toxicity levels.")

user_input = st.text_area("User Comment:", height=150)

if st.button("Analyze Content"):
    if not user_input.strip():
        st.warning("You cannot analyze empty text.")
    elif model is not None and tokenizer is not None:
        # 1. Clean
        cleaned = clean_input_text(user_input)
        
        # 2. Tokenize and Pad (Ensure maxlen matches training: 200)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        
        # 3. Predict
        predictions = model.predict(padded)[0]
        
        # 4. Display Results
        st.subheader("Analysis Results:")
        for label, prob in zip(labels, predictions):
            # Convert probability to percentage
            percentage = prob * 100
            
            # Visual feedback
            if percentage > 50:
                st.error(f"{label}: {percentage:.2f}% (High Risk)")
            elif percentage > 20:
                st.warning(f"{label}: {percentage:.2f}% (Warning)")
            else:
                st.success(f"{label}: {percentage:.2f}% (Clean)")
