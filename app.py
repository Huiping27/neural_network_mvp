from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Define your neural network (same architecture as before)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = Net()
model.load_state_dict(torch.load('mnist_net.pth'))
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/')
def home():
    return "Welcome to the MNIST digit classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    # Read the image
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img = transform(img).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)

    # Return the prediction as a JSON response
    return jsonify({'predicted_digit': predicted.item()}), 200

if __name__ == '__main__':
    app.run(debug=True)
