
import torch
from torchvision import datasets, transforms

class ImportImageData:
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_data(self):
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        data_dataset = datasets.ImageFolder(self.data_path, transform=transform)

        data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=self.batch_size, shuffle=True)
    
        return data_loader