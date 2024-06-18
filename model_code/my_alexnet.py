import torch
import torch.nn as nn
from torchvision import transforms as T, models
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

print("alexnet_load")


class alexnet_pn():
    def __init__(self):
        self.cuda_available()
        self.load_model()
        self.lung_class = {0: "Normal", 1: "Pneumonia"}

    def cuda_available(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        self.pn_judge_model = AlexNet()
        self.pn_judge_model.to(self.device)
        self.pn_judge_model.load_state_dict(
            torch.load("C:/Users/HKIT/Desktop/Flask_Server0520/model/model_weights_AlexNet.pth", map_location=self.device),
            strict=False)
        print("alexnet_model_load")

    def softmax(self, x):
        exp_x = np.exp(x)  # 오버플로우 방지를 위해 입력값 중 최댓값을 빼줌
        return exp_x / exp_x.sum(axis=0)

    def analyze_image_judgePN(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        transform = T.Compose([
            T.Resize(225),
            T.CenterCrop(225),
            T.ToTensor(),
        ])
        image = transform(image)
        print("alexnet_model_image_processing........")
        image = torch.tensor(image, dtype=torch.float32)
        image = image.to(self.device).unsqueeze(0)
        with torch.no_grad():
            self.pn_judge_model.eval()
            pred = self.pn_judge_model(image)
            probabilities = torch.nn.functional.softmax(pred, dim=1)
            confidence, preds = torch.max(probabilities, 1)

        res = self.lung_class[preds.item()]
        return {"result": res, "confidence": round(confidence.item(), 4)}


class AlexNet(nn.Module):
    def __init__(self, number_classes=2):
        super(AlexNet, self).__init__()
        self.model_name = 'alexnet'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.Linear(1000, number_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
