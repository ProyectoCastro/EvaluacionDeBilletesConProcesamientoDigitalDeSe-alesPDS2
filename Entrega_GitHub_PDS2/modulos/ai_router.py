import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class HybridRouter:
    def __init__(self, model_path='../models/best_model_EfficientNet.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = [
            'billete100_anverso_antiguo', 'billete100_anverso_nuevo',
            'billete100_reverso_antiguo', 'billete100_reverso_nuevo',
            'billete10_anverso_antiguo', 'billete10_anverso_nuevo',
            'billete10_reverso_antiguo', 'billete10_reverso_nuevo',
            'billete20_anverso_antiguo', 'billete20_anverso_nuevo',
            'billete20_reverso_antiguo', 'billete20_reverso_nuevo',
            'billete50_anverso_antiguo', 'billete50_anverso_nuevo',
            'billete50_reverso_antiguo', 'billete50_reverso_nuevo'
        ]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = models.efficientnet_b0(weights=None)
        if hasattr(self.model, 'classifier'):
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 16)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Modelo PyTorch cargado correctamente en {self.device}")
        except Exception as e:
            print(f"ATENCIÓN: No se pudo cargar el modelo CNN. Error: {e}")
            
    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(self.model(tensor), dim=1)
                conf, idx = torch.max(probs, 1)
            return self.classes[idx.item()], conf.item() * 100
        except:
            return None, 0.0

def map_pytorch_class_to_pds_folder(class_name):
    return class_name.replace('_nuevo', 'nuevo').replace('_antiguo', 'antiguo')
