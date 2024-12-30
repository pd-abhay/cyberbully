import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the trained model
@st.cache_resource
def load_pickle_model(file_path):
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load the Sentence Transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-mpnet-base-v2', device='cpu')

# Function to dynamically calculate Peerness
def calculate_peerness(age1, age2, grade1, grade2):
    # Avoid division by zero
    age_similarity = 1 - abs(age1 - age2) / max(age1, age2, 1)
    grade_similarity = 1 - abs(grade1 - grade2) / max(grade1, grade2, 1)
    # Combine similarities (simple average)
    peerness = (age_similarity + grade_similarity) / 2
    return peerness

# Paths
pickle_file_path = "C:/Users/Acer/Downloads/best_model.pkl"  # Update with your actual path
trained_model = load_pickle_model(pickle_file_path)
sentence_model = load_sentence_transformer()

# Streamlit App Title
st.title("Is it a Bully?")
st.write("Predict if the input message indicates bullying behavior based on user attributes and the message content.")

# Sidebar Inputs for Optional User Features
st.sidebar.header("Optional User Information")

# User Ages
age_user_1 = st.sidebar.text_input("Age of User 1 (Optional)", value="")
age_user_2 = st.sidebar.text_input("Age of User 2 (Optional)", value="")

# Validate Ages
try:
    age_user_1 = float(age_user_1) if age_user_1.strip() else 0
    age_user_2 = float(age_user_2) if age_user_2.strip() else 0
except ValueError:
    st.sidebar.error("Please enter valid numbers for ages.")

# User Genders
gender_user_1 = st.sidebar.radio("Gender of User 1 (Optional)", options=["Male", "Female", "Other"], index=0)
gender_user_2 = st.sidebar.radio("Gender of User 2 (Optional)", options=["Male", "Female", "Other"], index=0)

# Gender Encoding
enc_gender_1 = {"Male": 1, "Female": 2, "Other": 3}[gender_user_1]
enc_gender_2 = {"Male": 1, "Female": 2, "Other": 3}[gender_user_2]

# User Grades
grade_user_1 = st.sidebar.slider("Grade of User 1 (Optional, 1-10)", min_value=1, max_value=10, value=5)
grade_user_2 = st.sidebar.slider("Grade of User 2 (Optional, 1-10)", min_value=1, max_value=10, value=5)

# Main Input: Message to Classify
message = st.text_area("Enter a message to classify", placeholder="Type your message here...")

# Button to Trigger Classification
if st.button("Classify"):
    if not message.strip():
        st.error("Please enter a valid message.")
    else:
        # Calculate Peerness Dynamically
        peerness = calculate_peerness(age_user_1, age_user_2, grade_user_1, grade_user_2)

        # Generate Embeddings for the Message
        message_embedding = sentence_model.encode([message], batch_size=1)

        # Combine Features for Prediction
        input_features = np.array([[peerness, age_user_1, enc_gender_1, grade_user_1, age_user_2, enc_gender_2, grade_user_2]])
        combined_input = np.concatenate((message_embedding, input_features), axis=1).astype("float32")

        # Make Prediction
        prediction = trained_model.predict(combined_input)
        tag = "Bully" if prediction[0] == 1 else "Not a Bully"

        # Display the Prediction Result
        st.subheader("Prediction Result")
        st.write(f"The model predicts: **{tag}**")

        # Optional Debugging Info
        st.write("Peerness:", peerness)
        st.write("Message Embedding Shape:", message_embedding.shape)
        st.write("Combined Input Shape:", combined_input.shape)
