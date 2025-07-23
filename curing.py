import os
import streamlit as st
import pymongo
import google.generativeai as genai

# Set up the GEMINI API key from the environment variable
os.environ["GEMINI_API_KEY"] = "AIzaSyA6gYO5o9WfvN00_BiQNIGv7-7xzQURLWk"  # Replace with your actual API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model with appropriate generation configurations
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000,  # Adjust based on response length
    "response_mime_type": "text/plain",
}

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",  # Ensure this model is accessible with your API key
    generation_config=generation_config,
)

# Start a new chat session
chat_session = model.start_chat(history=[])

def get_treatment_plan(plant_name, disease_name, severity_level):
    """
    Generate a treatment plan for the given plant disease and severity level.
    
    Args:
    - plant_name (str): Name of the plant.
    - disease_name (str): Name of the detected disease.
    - severity_level (str): Severity level of the disease (Low, Moderate, High).
    
    Returns:
    - str: Generated treatment plan.
    """
    prompt = f"""
    You are an agricultural expert providing plant disease treatment advice. Based on the detected plant disease and severity level, generate a detailed response that includes:

    1. Chemical treatment methods (specific fungicides/pesticides, dosage, application frequency).
    2. Organic treatment methods (natural remedies, bio-controls, cultural practices).
    3. The estimated time required for recovery for each method.
    4. Preventive measures to avoid recurrence.

    Detected Plant: {plant_name}
    Disease: {disease_name}
    Severity Level: {severity_level} (Low, Moderate, High)

    Provide the response in the following format:
    - **Chemical Treatment**: [Details]
    - **Organic Treatment**: [Details]
    - **Estimated Recovery Time**: [Details for both methods]
    - **Preventive Measures**: [Details]
    """
    
    # Send the prompt to the chat session
    response = chat_session.send_message(prompt)
    return response.text

def display_curing_page():
    """
    Display the curing page in the Streamlit app.
    """
    # MongoDB connection setup
    mongo_client = pymongo.MongoClient("e2wrmongodb+srv://sathwikhs235:mlZmzYDhLsvkaMOe@cluster0.hyp6k.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB connection string
    db = mongo_client["PlantDiseaseDB"]  # Replace with your database name
    collection = db["DiseasePredictions"]  # Replace with your collection name

    # Streamlit interface setup for the curing page
    st.title("🌱 Plant Disease Treatment Plan Generator")
    st.markdown(
        """
        Welcome to the **Curing Page**! Here, you can generate treatment plans for plant diseases based on severity levels.
        Click the button below to fetch prediction data and generate a detailed treatment plan.
        """
    )

    # Button to retrieve data and generate treatment plan
    if st.button("Generate Treatment Plan"):
        try:
            with st.spinner("Fetching prediction data and generating treatment plan..."):
                # Fetch the latest entry from MongoDB
                latest_entry = collection.find_one(sort=[("_id", pymongo.DESCENDING)])
                
                if latest_entry:
                    plant_name = latest_entry.get("plant_name", "Unknown Plant")
                    disease_name = latest_entry.get("disease_name", "Unknown Disease")
                    severity_level = latest_entry.get("severity_level", "Unknown Severity")

                    # Generate treatment plan using the AI model
                    treatment_plan = get_treatment_plan(plant_name, disease_name, severity_level)
                    
                    # Update the document with the new treatment plan
                    result = collection.update_one(
                        {"_id": latest_entry["_id"]},  # Match by ID
                        {"$push": {"treatment_plans": treatment_plan}}  # Append to the treatment_plans array
                    )

                    if result.modified_count > 0:
                        # Display the results
                        st.success("Treatment Plan Generated and Stored Successfully!")
                        st.markdown("### **Generated Treatment Plan**")
                        st.markdown(f"""
                        <div class="treatment-plan">
                            {treatment_plan}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display all treatment plans for this record
                        st.markdown("### **All Treatment Plans for this Record**")
                        updated_entry = collection.find_one({"_id": latest_entry["_id"]})
                        if "treatment_plans" in updated_entry:
                            for idx, plan in enumerate(updated_entry["treatment_plans"], start=1):
                                st.markdown(f"#### Plan {idx}:")
                                st.markdown(f"""
                                <div class="treatment-plan">
                                    {plan}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to update the database with the treatment plan.")
                else:
                    st.warning("No prediction data found in the database.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add custom CSS for styling
    st.markdown(
        """
        <link rel="stylesheet" href="curing.css">
        """,
        unsafe_allow_html=True,
    )
