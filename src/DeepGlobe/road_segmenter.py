import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics import Accuracy, JaccardIndex

class RoadSegmenter(pl.LightningModule):
    def __init__(self, model_name = "segformer", encoder_name = "resnet34", encoder_weights = "imagenet"):
        '''
        Args: I preemptively entered the default version, but didn't want to "lock it", just in case we need to test more models.
            model_name      : Name of desired model from segmentation_models_pytorch library
            encoder_name    : Name of desired encoder from segmentation_models_pytorch library
            encoder_weights : Name of desired weights for encoder from segmentation_models_pytorch library
        '''
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            model_name,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )        
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)    # Loss function
        self.acc = Accuracy(task="binary")              # Pixel accuracy compared to whole image
        self.pixel_acc = JaccardIndex(task="binary")    # Pixel accuracy compared to TP + TF + FN (IoU score)
        self.imgid = 0


    def forward(self, x):
        return self.model(x)

    '''
    These are methods for the lightning module, it makes it more readable and it automatically calls them in the step given by the name.
    '''
    def shared_step(self, batch, stage):
        images, masks = batch
        raw_predictions = self(images)
        loss = self.loss_fn(raw_predictions, masks)
        
        prob_masks = raw_predictions.sigmoid()
        pred_masks = (prob_masks > 0.5).type(torch.uint8)
        
        self.acc(pred_masks, masks)
        self.pixel_acc(pred_masks, masks)
        
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_pixel_acc", self.pixel_acc, prog_bar=True)
        del prob_masks
        del pred_masks
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        del batch
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "val")
        return loss

    def on_train_epoch_end(self):
        print(f"Memory:     {torch.cuda.memory_allocated()}")
        self.log("train_epoch_acc", self.acc.compute(), prog_bar=True)
        self.log("train_epoch_pixel_acc", self.pixel_acc.compute(), prog_bar=True)
        if self.pixel_acc >= 0.95 and self.acc >= 0.95:
            self.model.stop_training = True
        self.acc.reset()
        self.pixel_acc.reset()


    def on_validation_epoch_end(self):
        self.log("val_epoch_acc", self.acc.compute(), prog_bar=True)
        self.log("val_epoch_pixel_acc", self.pixel_acc.compute(), prog_bar=True)
        self.acc.reset()
        self.pixel_acc.reset()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [scheduler]