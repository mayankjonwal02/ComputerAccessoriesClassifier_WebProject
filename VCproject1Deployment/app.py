import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from flask import Flask, jsonify
import requests
from flask import Flask, request, jsonify
import pygame
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify, render_template, send_file
app = Flask(__name__)

classdict = {0:'cable',1:'keyboard',2:'monitor',3:'mouse',4:'pendrive'}
class myCNN(nn.Module):
    def __init__(self, num_classes):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def transform_image(image_path):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    ])

    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return transform(image).unsqueeze(0)  # Add a batch dimension

def get_prediction(image_tensor, model):
    outputs = model(image_tensor)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

def predict_image(image_path):
    PATH = "large_model.pth"  # Path to your model
    totalclasses = 5
    model = myCNN(totalclasses)
    state_dict = torch.load(PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    try:
        image_tensor = transform_image(image_path)
        prediction = get_prediction(image_tensor, model)
        return classdict[prediction]
    except Exception as e:
        return str(e)

# Example usage:


@app.route("/process-image", methods=["POST", "GET"])
def process_image():
    # Retrieve the image data from the request
    image_data = request.json.get("image_data")

    # Decode the base64 image data and convert it to a NumPy array
    _, encoded_data = image_data.split(",", 1)
    decoded_data = base64.b64decode(encoded_data)
    nparr = np.frombuffer(decoded_data, np.uint8)

    # Read the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform any processing on the image
    # ...
    cv2.imwrite("input.jpeg", img)
    image_path = 'input.jpeg'  
    predictedclass = predict_image(image_path)
    return jsonify({"PredictedClass":predictedclass})

    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 
