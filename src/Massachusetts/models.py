import os
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU, Fscore

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "CPU")
METRICS = [
    IoU(threshold=0.5),
    Fscore()
]
CLASSES = ["background", "road"]

class Unet():
    def __init__(self):
        encoder = 'resnet50'
        encoder_weights = 'imagenet'
        activation = 'sigmoid'
        
        self.model = smp.Unet(
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=len(CLASSES), 
            activation=activation,
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        self.loss = DiceLoss()
        
        self.optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=0.00008),
        ])
        
    def load_model(self, path):
        self.model = torch.load(path, map_location=DEVICE)

    def save_model(self, path):
        torch.save(self.model, path)

    def get_train_valid_epoch(self):
        train_epoch = TrainEpoch(
            self.model, 
            loss=self.loss, 
            metrics=METRICS, 
            optimizer=self.optimizer,
            device=DEVICE,
            verbose=True,
        )
        
        valid_epoch = ValidEpoch(
            self.model, 
            loss=self.loss, 
            metrics=METRICS, 
            device=DEVICE,
            verbose=True,
        )

        return train_epoch, valid_epoch



