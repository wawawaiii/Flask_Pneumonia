import torch
import torch.nn as nn
from torchvision import transforms as T, models
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class vgg16_pn():
    def __init__(self):
        self.cuda_available()
        self.load_model()
        self.lung_class = {0: "Normal", 1: "Pneumonia"}

    def cuda_available(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        self.pn_judge_model = models.vgg16()
        self.pn_judge_model.load_state_dict(
            torch.load("C:/Users/HKIT/Desktop/Flask_Server0520/model/vgg16-397923af.pth", map_location=self.device))
        for param in self.pn_judge_model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(4096, 4096)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(0.3)),
            ('fc3', nn.Linear(4096, 2)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.pn_judge_model.classifier = classifier
        self.pn_judge_model.to(self.device)
        pn_judge_model_save_path = 'C:/Users/HKIT/Desktop/Flask_Server0520/model/Pneumonia_model.pt'
        self.pn_judge_model.load_state_dict(torch.load(pn_judge_model_save_path, map_location=self.device))

    def softmax(self, x):
        exp_x = np.exp(x)  # 오버플로우 방지를 위해 입력값 중 최댓값을 빼줌
        return exp_x / exp_x.sum(axis=0)

    def analyze_image_judgePN(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        data = MyPNImageData()
        transform_img = data.process_image(image)
        with torch.no_grad():
            self.pn_judge_model.eval()
            pred = self.pn_judge_model(transform_img.to(self.device).view(1, 3, 224, 224))

            probabilities = torch.nn.functional.softmax(pred, dim=1)
            confidence, preds = torch.max(probabilities, 1)

        res = self.lung_class[preds.item()]

        return {"result": res, "confidence": round(confidence.item(), 4)}


class MyPNImageData(Dataset):
    def __init__(self):
        self.image_transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def process_image(self, img):
        return self.image_transform(img)
