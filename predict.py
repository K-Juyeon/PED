import os

import torch
from torchvision import transforms

from PIL import Image

from sklearn.metrics import accuracy_score

import data_loader
import utils

class Predict :
    def __init__(self, test_path, modelss, model_path, batch_size):
        self.device = torch.device("cuda:0")

        self.test_path = test_path
        self.test_data = data_loader.ImportImageData(test_path, batch_size)
        self.test_loader= self.test_data.load_data()

        self.model_path = model_path

        self.num_classes = len(os.listdir(test_path))
        self.class_names = os.listdir(test_path)

        self.load_model = utils.SelectModel(self.num_classes, modelss)
        self.model = self.load_model.select_model()
        self.model.load_state_dict(torch.load(model_path)['net'])
        self.model = self.model.to(self.device)

    def test(self):
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader :
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)

        print(f"Test - Accuracy: {100 * accuracy:.4f}")

    def top(self):
        self.model.eval()

        top1_sucess = 0
        top5_sucess = 0
        filelist_sum = 0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        for root, dir, files in os.walk(self.test_path) :
            filelist = os.listdir(root)
            filelist_jpg = [file for file in filelist if file.endswith(".jpg")]
            filelist_sum += len(filelist_jpg)
            for file in filelist_jpg :
                image_path = os.path.join(root, file)
                label = image_path.split('\\')[-2]

                image = Image.open(image_path)

                input_tensor = transform(image)
                input_batch = input_tensor.unsqueeze(0)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(device)
                input_batch = input_batch.to(device)

                with torch.no_grad():
                    input_batch = input_batch.to(device)
                    outputs = self.model(input_batch)

                # Top-1 예측
                _, predicted_idx = torch.max(outputs, 1)
                top1_accuracy = self.class_names[predicted_idx.item()]

                # Top-5 예측
                _, top5_indices = torch.topk(outputs, 5)
                top5_accuracy = [self.class_names[idx.item()] for idx in top5_indices[0]]

                if label in top1_accuracy :
                    top1_sucess += 1
                if label in top5_accuracy :
                    top5_sucess += 1

        print(self.model_path, str(round(top1_sucess/filelist_sum*100, 4)) + "%")
        print(self.model_path, str(round(top5_sucess/filelist_sum*100, 4)) + "%")