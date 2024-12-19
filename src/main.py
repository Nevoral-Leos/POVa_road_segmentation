import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data_preparation import DeepGlobe
from road_segmenter import RoadSegmenter

torch.set_float32_matmul_precision('medium')
torch.cuda.memory._record_memory_history()

dataset_dir = "./data/deepglobe"
train = DeepGlobe(dataset_dir, ['train'])                       #loads only training images and masks
trainLoader = DataLoader(train, batch_size=2, num_workers=2, shuffle=False)
#test = DeepGlobe(dataset_dir, ['test','valid'])                 #loads only test / valid images from the dataset, not sure if necessary, since there are no masks...
#testLoader = DataLoader(train, batch_size=2, num_workers=8, shuffle=False)

model = RoadSegmenter()
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1, accelerator='gpu', devices=1 if torch.cuda.is_available() else 0)
trainer.fit(model, train_dataloaders=trainLoader, val_dataloaders=trainLoader)
torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

torch.save(model.state_dict(), "model.pth")