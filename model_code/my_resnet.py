from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T, models
from collections import OrderedDict
import segmentation_models_pytorch as smp
from fastapi.responses import StreamingResponse
import pydicom as dicom
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import base64
from fastapi.responses import JSONResponse
import traceback
from torchvision import transforms

# ResNetPNImageData 클래스 정의
class ResNetPNImageData:
    def __init__(self):
        print("image_transform")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_image(self, image):
        print("image_transform_process")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        transformed_image = self.transform(image)
        print(f"Transformed Image: {transformed_image.shape}")
        return transformed_image

    def process_dicom(self, dicom_data):
        dicom_image = dicom.dcmread(io.BytesIO(dicom_data))
        image_array = dicom_image.pixel_array
        image = Image.fromarray(image_array).convert('RGB')
        return self.process_image(image)

# resnet_pn 클래스 정의
class resnet_pn:
    def __init__(self):
        self.cuda_available()
        self.load_model()
        self.lung_class = {0: "Normal", 1: "Pneumonia"}
        self.image_data_processor = ResNetPNImageData()  # ResNetPNImageData 클래스 인스턴스 생성

    def load_model(self):
        print("loadmodel")
        self.resnet_model = models.resnet101()
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        model_path = 'C:/Users/HKIT/Desktop/Flask_Server0520/model/complete_model_resnet101bk3.pth'
        self.resnet_model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.resnet_model.to(self.device).eval()

    def cuda_available(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_image_judgePN(self, image: Image.Image):
        print("analyze_image")
        transform_img = self.image_data_processor.process_image(image)
        with torch.no_grad():
            pred = self.resnet_model(transform_img.unsqueeze(0).to(self.device))
            probabilities = torch.nn.functional.softmax(pred, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            class_name = self.lung_class[preds.item()]
        return {"result": class_name, "confidence": round(confidence.item(), 4)}
