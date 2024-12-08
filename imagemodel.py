from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
from flask_cors import CORS 
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)  
# For Getting scan information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_scan(model_path_scan, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path_scan, map_location=device))
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image_scan(image, model, class_names_scan):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    return class_names_scan[predicted_class.item()]


model_path_scan = "./multimodal resnet/resnet_multimodal.pth"
class_names_scan = ['Pneumonia', 'Retinopathy']

model_for_scan = load_model_scan(model_path_scan, len(class_names_scan))
image_path_scan = "multimodal resnet/rpt.png"  

############################################################################################################
# Retinopathy detection

model_retino = load_model('./Retinopathy resnet 50/resnet retinopathy.keras')
class_labels_retino = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
def predict_image_retino(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    predictions = model_retino.predict(img_array)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels_retino[predicted_class_index]
    # confidence = predictions[0][predicted_class_index] * 100
    
    return predicted_class


# Chest detection

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

loaded_model_chest = load_model("./Chest Dataset trained model/final_trained_model.keras")
def predict_image_chest(img):
    # print("Entered function")
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    # print("Got image ", img.size)
    img_array = image.img_to_array(img) / 255.0
    # print("Made array")
    img_array = np.expand_dims(img_array, axis=0)
    # print("img array created")
    prediction = loaded_model_chest.predict(img_array)
    class_label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    # print(class_label)
    return class_label

############################################################################################################

llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = "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"
)


def generate_response_text(prediction_scan, prediction_dis=None, query_asked=None):
    prompt_extract = PromptTemplate.from_template(
            """
            ### Give the Following Question asked by user:
            {query_asked}
            ### INSTRUCTION:The user have detected an image, and the detection is came as follows: {prediction_scan} is detected scan of image and the disease is: {prediction_dis} of this scan
            based on this information, provide the valid response to the user's question in html inline css all in white text with no html tag , based on the context of the document and the instruction given.
            ### (NO PREAMBLE):
            """
    )

    chain_extract = prompt_extract | llm 
    res = chain_extract.invoke(input={'prediction_scan':prediction_scan, 'query_asked':query_asked, 'prediction_dis':prediction_dis})
    return res.content

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        description = request.form
        print(description["query"])
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image_file = request.files['image']
        image = Image.open(BytesIO(image_file.read()))

        prediction_scan = predict_image_scan(image, model_for_scan, class_names_scan)
        
        if(prediction_scan == "Retinopathy"):
            prediction_retino = predict_image_retino(image)
            valid_res = generate_response_text(prediction_scan, prediction_dis=prediction_retino, query_asked=description["query"])
            return jsonify({'success': True, 'prediction_scan': prediction_scan, "response": valid_res})
        else:
            prediction_chest = predict_image_chest(image)
            valid_res = generate_response_text(prediction_scan, prediction_dis=prediction_chest, query_asked=description["query"])
            return jsonify({'success': True, 'prediction_scan': prediction_scan, "response": valid_res})
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
