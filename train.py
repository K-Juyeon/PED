import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import data_loader
import utils

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

writer = SummaryWriter()

class FineTuning:
    def __init__(self, train_path, val_path, modelss, epoch, optimizer, batch_size):
        self.device = torch.device("cuda:0")

        self.train_data = data_loader.ImportImageData(train_path, batch_size)
        self.train_loader= self.train_data.load_data()
        self.val_data = data_loader.ImportImageData(val_path, batch_size)
        self.val_loader= self.val_data.load_data()

        self.num_classes = len(os.listdir(train_path))

        self.load_model = utils.SelectModel(self.num_classes, modelss)
        self.model = self.load_model.select_model()
        self.model = self.model.to(self.device)

        self.optimizer = utils.SelectOptimizer(optimizer, self.model)
        self.optimizer = self.optimizer.select_optimizer()

        self.epoch = epoch
        self.criterion = nn.CrossEntropyLoss()

        self.txt_model = modelss
        self.txt_optimizer = optimizer

    def training(self):
        self.model.train()
        training_loss = 0.0
        total = 0
        correct = 0
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        for batch_i, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            inputs, targets = sample
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step() scalar 미사용시 사용

            training_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        writer.add_scalar('Loss/train', training_loss / total, self.epoch)
        writer.add_scalar('Accuracy/train', (100. * correct / total), self.epoch)

        print(f"Epoch: {self.epoch}")
        print(f"Training - Loss: {training_loss / total: .4f}, Accuracy: {100 * correct / total: .4f}")

        return self.model

    def validation(self, model, early_stop, epoch, best_acc):
        self.model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader) :
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100 * correct / total

        writer.add_scalar('Loss/valid', valid_loss / total, epoch)
        writer.add_scalar('Accuracy/valid', accuracy, epoch)
        
        print(f"Validation - Loss: {valid_loss / total: .4f}, Accuracy: {accuracy: .4f}")

        if accuracy > best_acc :
            state = {
                'net': model.state_dict(),
                'acc': accuracy,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'checkpoint/{self.txt_model}_{self.txt_optimizer}.pt')
            best_acc = accuracy
            print(f'Best Acc : {best_acc: .3f}')
            early_stop = 0
        else:
            early_stop += 1

        return best_acc, early_stop, all_targets, all_predictions
