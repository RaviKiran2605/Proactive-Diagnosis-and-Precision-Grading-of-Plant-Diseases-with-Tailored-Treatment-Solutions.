import streamlit as st
import pandas as pd
import base64
from googletrans import Translator
from PIL import Image
from io import BytesIO
from pymongo import MongoClient

# Initialize the translator
translator = Translator()

# Connect to MongoDB
def get_database():
    """Establish a connection to the MongoDB database."""
    client = MongoClient("mongodb+srv://sathwikhs235:mlZmzYDhLsvkaMOe@cluster0.hyp6k.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
    return client["PlantDiseaseDB"]  # Replace with your database name

def fetch_predictions_from_mongodb():
    """Fetch prediction results from MongoDB without the timestamp."""
    db = get_database()
    collection = db["DiseasePredictions"]

    # Retrieve all documents from the collection, excluding the timestamp field
    documents = collection.find({}, {"timestamp": 0})
    return list(documents)

def translate_text(text, target_language):
    """Translate text using googletrans."""
    try:
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation API request failed: {e}")
        return text

def display_dashboard_page():
    st.title("📊 Dashboard: Plant Disease Predictions")

    # Fetch prediction data from MongoDB
    try:
        data = fetch_predictions_from_mongodb()

        if data:
            # Create a DataFrame for tabular display
            df = pd.DataFrame(data)
            df = df.drop(columns=["_id", "image", "treatment_plan"])  # Drop unnecessary fields for the table view

            st.markdown("### All Prediction Results:")
            st.dataframe(df, use_container_width=True)  # Display the table

            # Allow user to select a record from the table
            selected_prediction = st.selectbox(
                "Select a prediction to view details:",
                options=[f"{d['plant_name']} (ID: {d['_id']})" for d in data]
            )

            # Extract the selected record's _id
            selected_prediction_id = selected_prediction.split(" (ID: ")[-1][:-1]  # Extract _id from the string

            # Find the selected record
            selected_record = next(record for record in data if str(record["_id"]) == selected_prediction_id)

            # Show details if a row is selected
            if selected_record:
                st.markdown("---")
                st.markdown("### 🖼️ Detailed Prediction:")

                st.markdown(f"**Plant: {selected_record['plant_name']}**")
                st.markdown(f"**Disease: {selected_record['disease_name']}**")
                st.markdown(f"**Severity Level: {selected_record['severity_level']}**")
                st.markdown(f"**Severity Percentage: {selected_record['severity_percentage']}%**")
                
                # Decode and display the image
                image_binary = base64.b64decode(selected_record["image"])
                image = Image.open(BytesIO(image_binary))
                st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
                
                # Display treatment plan
                st.markdown(f"**Treatment Plan:** {selected_record['treatment_plan']}")

                # Translation option
                st.markdown("### 🌐 Translate Treatment Plan:")
                languages = {
                    "Kannada": "kn",
                    "Hindi": "hi",
                    "Tamil": "ta",
                    "Telugu": "te",
                    "Malayalam": "ml"
                }
                selected_language = st.selectbox("Select a language:", options=languages.keys())
                if selected_language:
                    if st.button("Translate"):
                        translated_plan = translate_text(selected_record['treatment_plan'], languages[selected_language])
                        st.markdown(f"**Translated Treatment Plan ({selected_language}):** {translated_plan}")
        else:
            st.info("No prediction data available.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Run the app
if __name__ == "__main__":
    display_dashboard_page()
