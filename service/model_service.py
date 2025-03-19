from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import io
import base64
from io import BytesIO
import numpy as np
from torch import nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.conv2d(x)

# Load the model
model = NeuralNetwork()
model.load_state_dict(torch.load("digit_model_cv2d-2.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()

# FastAPI app
app = FastAPI()

# Preprocessing for MNIST
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')

    image = ImageOps.invert(image)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(60.0)  # You can adjust this factor if needed

    image = transform(image)

    image = 1.0 - image

    print(f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean()}")

    # Convert tensor back to PIL image for debugging
    debug_image = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype(np.uint8))

    # Encode the image to base64
    buffered = BytesIO()
    debug_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()


    print(image.shape)

    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        pred_probab = nn.Softmax(dim=1)(pred)
        confidence = torch.max(pred_probab).item()
        predicted = classes[pred[0].argmax(0)]
        print(f'Predicted: "{predicted}", Confidence: "{confidence}"')

    return {"label": predicted, "confidence": confidence, "debug_image": img_str}
