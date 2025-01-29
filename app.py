import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load dataset
csv_file = 'dataset.csv'
df = pd.read_csv(csv_file)

# Initialize model pipeline (Ensure you have the model)
model = models.resnet18(pretrained=True)

# If you have a custom trained model, you can load it as follows
try:
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    print("Loaded custom model weights.")
except FileNotFoundError:
    print("Custom model weights not found, using pre-trained ResNet18 model.")

model.fc = nn.Linear(model.fc.in_features, len(df['Diagnosis'].unique()))  # Adjust for medical classes
model.eval()

def process_text(text):
    """Process text (symptoms) to generate a diagnosis."""
    # Define a basic mapping of symptoms to diagnoses and medications
    symptom_to_diagnosis = {
        "fever": ("Fever", "Paracetamol"),
        "headache": ("Migraine", "Ibuprofen"),
        "cough": ("Common Cold", "Cough Syrup"),
        "cold": ("Common Cold", "Cough Syrup"),
        # Add more mappings as needed
    }

    # Check if the symptom matches any known condition
    diagnosis, medication = "Unknown Condition", "No medication available"

    for symptom, (diagn, med) in symptom_to_diagnosis.items():
        if symptom.lower() in text.lower():
            diagnosis, medication = diagn, med
            break
    
    response = f"Based on the symptoms '{text}', we suggest a possible diagnosis: {diagnosis} and medication: {medication}."
    return response

def process_image(image_path):
    """Classify medical images using the CNN model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, 1).item()
    diagnosis = df.iloc[prediction]['Diagnosis']
    medicine = df.iloc[prediction]['Medicines']
    return diagnosis, medicine

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chat functionality (text input)
@app.route('/chat', methods=['POST'])
def chat():
    text = request.form.get('user_input')
    if not text:
        return jsonify({"error": "No symptoms provided"}), 400
    
    response = process_text(text)
    return jsonify({"message": response})

# Route to handle image upload and medical diagnosis (image input)
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({"error": "Invalid file format. Please upload a valid image."}), 400
    
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    diagnosis, medicine = process_image(file_path)
    return jsonify({"diagnosis": diagnosis, "medicine": medicine})

# Main entry point to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
