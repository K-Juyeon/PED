import torch.optim as optim

import model as pmodel

class SelectModel:
    def __init__(self, num_classes, smodel):
        self.num_classes = num_classes
        self.smodel = smodel

    def select_model(self):
        if self.smodel == "vgg16" :
            model = pmodel.VGG16_BN(num_classes=self.num_classes)
        elif self.smodel == "cnn" :
            model = pmodel.CNN(num_classes=self.num_classes)
        else :
            model = pmodel.ResNet50(num_classes=self.num_classes)

        return model
    
class SelectOptimizer:
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model

    def select_optimizer(self):
        if self.optimizer == "sgd" :
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.999, weight_decay=1e-5, nesterov=True)
        else :
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001) #0.001
            
        return optimizer