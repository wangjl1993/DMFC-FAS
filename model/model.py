import torch
import torch.nn as nn
from torchvision import models
from model.resnext import resnext50_32x4d, resnext101_32x8d

class EfficientNetB7Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetB7Classifier, self).__init__()
        self.model = models.efficientnet_b7(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)


class EfficientNetB6Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetB6Classifier, self).__init__()
        self.model = models.efficientnet_b6(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)
    

class EfficientNetB5Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetB5Classifier, self).__init__()
        self.model = models.efficientnet_b5(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)
    

class EfficientNetB4Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetB4Classifier, self).__init__()
        self.model = models.efficientnet_b4(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)
    

class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetB3Classifier, self).__init__()
        self.model = models.efficientnet_b3(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)


class ResNet101Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet101Classifier, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        # Replace the final fully connected layer for 4-class classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)
    

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        # Replace the final fully connected layer for 4-class classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)
    
    
class ResNet34Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet34Classifier, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        # Replace the final fully connected layer for 4-class classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer for 4-class classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self,):
        """
        Freeze or unfreeze the parameters of the backbone (all layers except the final fc layer)
        
        Args:
            freeze (bool): If True, freeze backbone parameters; if False, make them trainable
        """
        for name, param in self.model.named_parameters():
            if "fc" not in name:  # Skip the final fully connected layer
                param.requires_grad = False

    def reinit_fc(self, num_classes):
        """
        Reinitialize the fully connected layer with default initialization
        """
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def extract_features(self, x):
        features = torch.nn.Sequential(*list(self.model.children())[:-1])(x)
        return torch.flatten(features, 1)

 
MODEL_DICT = {
    'resnet18': ResNet18Classifier,
    'resnet34': ResNet34Classifier,
    'resnet50': ResNet50Classifier,
    'resnet101': ResNet101Classifier,
    'resnext50': resnext50_32x4d,
    'resnext101': resnext101_32x8d,
    'efficientnetb7': EfficientNetB7Classifier,
    'efficientnetb6': EfficientNetB6Classifier,
    'efficientnetb5': EfficientNetB5Classifier,
    'efficientnetb4': EfficientNetB4Classifier,
    'efficientnetb3': EfficientNetB3Classifier
}

def build_model(model_name, num_classes=6, **args):
    """
    Factory function to create and initialize the ResNet101 model
    """
    model = MODEL_DICT[model_name](num_classes=num_classes, **args)
    return model 

