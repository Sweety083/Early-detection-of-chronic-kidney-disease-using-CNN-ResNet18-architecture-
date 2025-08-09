import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fpdf import FPDF

# --------------------
# Load the Model
# --------------------
@st.cache_resource
def load_model():
    # Create ResNet18 and modify the last layer for 4 classes
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 4)  # 4 output classes

    # Load state_dict
    state_dict = torch.load("ckd_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_model()

# --------------------
# Class Labels
# --------------------
class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# --------------------
# Image Transform
# --------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # same as training
])

# --------------------
# PDF Generation
# --------------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Chronic Kidney Disease Detection Report', ln=True, align='C')

    def chapter_body(self, patient_name, patient_age, patient_gender, patient_notes, prediction):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
        self.cell(0, 10, f"Age: {patient_age}", ln=True)
        self.cell(0, 10, f"Gender: {patient_gender}", ln=True)
        self.multi_cell(0, 10, f"Notes: {patient_notes}")
        self.cell(0, 10, f"Prediction: {prediction}", ln=True)

def generate_pdf(patient_name, patient_age, patient_gender, patient_notes, prediction):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(patient_name, patient_age, patient_gender, patient_notes, prediction)
    pdf_output = "report.pdf"
    pdf.output(pdf_output)
    return pdf_output

# --------------------
# Streamlit App
# --------------------
st.title("🩺 CKD Detection from CT Scan Images")

# Patient details
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=0, max_value=120)
patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
patient_notes = st.text_area("Additional Notes")

# Image upload
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_container_width=True)

    if st.button("Predict"):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_labels[predicted.item()]

        st.success(f"Prediction: **{prediction}**")

        # Generate PDF
        pdf_path = generate_pdf(patient_name, patient_age, patient_gender, patient_notes, prediction)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("Download Report", data=pdf_file, file_name=pdf_path, mime="application/pdf")









