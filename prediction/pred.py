import streamlit as st
import torch
from torchvision.models import densenet201
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pymongo
from io import BytesIO
import base64
import datetime
import os
import google.generativeai as genai
from curing import get_treatment_plan  # Import treatment plan function

# MongoDB connection
def get_database():
    client = pymongo.MongoClient("mongodb+srv://sathwikhs235:mlZmzYDhLsvkaMOe@cluster0.hyp6k.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["PlantDiseaseDB"]
    return db

# Store prediction and treatment data in MongoDB
def store_prediction_and_treatment(plant_name, disease_name, severity_level, severity_percentage, treatment_plan, image):
    db = get_database()
    collection = db["DiseasePredictions"]

    # Convert image to binary
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_binary = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Prepare the document
    document = {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "severity_level": severity_level,
        "severity_percentage": severity_percentage,
        "treatment_plan": treatment_plan,
        "image": image_binary,
        "timestamp": datetime.datetime.now()
    }
    result = collection.insert_one(document)
    return result.inserted_id

# Load the DenseNet201 model
@st.cache_resource
def load_model(model_path, num_classes):
    model = densenet201(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(size=232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Severity calculation function
def calculate_severity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = image.shape[0] * image.shape[1]
    affected_area = sum([cv2.contourArea(cnt) for cnt in contours])
    severity_percentage = (affected_area / total_area) * 100
    if severity_percentage < 25:
        severity = 'Mild'
    elif severity_percentage < 50:
        severity = 'Moderate'
    elif severity_percentage < 75:
        severity = 'Severe'
    else:
        severity = 'Very Severe'
    return severity, severity_percentage

# Prediction function
def predict_disease(image, model, selected_plant):
    selected_classes = plant_classes[selected_plant]
    all_indices = [all_diseases.index(cls) for cls in selected_classes]

    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    valid_confidences = {selected_classes[i]: confidences[idx].item() for i, idx in enumerate(all_indices) if idx < len(confidences)}
    if valid_confidences:
        predicted_label = max(valid_confidences, key=valid_confidences.get)
        confidence = valid_confidences[predicted_label]
        return predicted_label, confidence
    else:
        return None, None

# Plant and disease mapping
all_diseases = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

plants = list(set(cls.split("___")[0] for cls in all_diseases))
plant_classes = {plant: [cls for cls in all_diseases if cls.startswith(plant)] for plant in plants}

# Prediction page display
def display_prediction_page():
    st.title("Plant Disease Detection")
    st.sidebar.header("Options")

    # List of all plants
    plants = [
        "Apple", "Blueberry", "Cherry_(including_sour)", "Corn_(maize)", "Grape",
        "Orange", "Peach", "Pepper,_bell", "Potato", "Raspberry", "Soybean",
        "Squash", "Strawberry", "Tomato"
    ]
    
    # Dropdown for plant selection
    selected_plant = st.sidebar.selectbox("Select a Plant", plants)

    # Display the selected plant
    st.write(f"**Selected Plant:** {selected_plant}")

    # Model path and number of classes
    MODEL_PATH = "model_weights_densenet.pth"
    NUM_CLASSES = 38

    # Load model
    model = load_model(MODEL_PATH, NUM_CLASSES)

    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict disease
        predicted_label, confidence = predict_disease(image, model, selected_plant)

        # Calculate severity
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        severity, severity_percentage = calculate_severity(image_cv)

        # Generate treatment plan
        treatment_plan = get_treatment_plan(selected_plant, predicted_label, severity)

        # Display results
        if predicted_label:
            st.success(f"Disease: {predicted_label}")
            st.warning(f"Severity: {severity} ({severity_percentage:.2f}%)")
            st.info(f"Treatment Plan: {treatment_plan}")

            # Store prediction and treatment plan in MongoDB
            record_id = store_prediction_and_treatment(
                plant_name=selected_plant,
                disease_name=predicted_label,
                severity_level=severity,
                severity_percentage=severity_percentage,
                treatment_plan=treatment_plan,
                image=image
            )
            st.info(f"Prediction and treatment plan stored in the database with ID: {record_id}")
        else:
            st.error("Could not predict the disease. Please try again.")

# Add custom CSS
st.markdown(
    """
    <link rel="stylesheet" href="pred.css">
    """,
    unsafe_allow_html=True,
)

# Run the app
if __name__ == "__main__":
    display_prediction_page()
