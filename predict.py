import os

import torch

from sklearn.metrics import accuracy_score

import data_loader
import utils

class Predict :
    def __init__(self, test_path, modelss, model_path, batch_size):
        self.device = torch.device("cuda:0")

        self.test_data = data_loader.ImportImageData(test_path, batch_size)
        self.test_loader= self.test_data.load_data()

        self.model_path = model_path

        self.num_classes = len(os.listdir(test_path))

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