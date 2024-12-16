# POVa_road_segmentation
Team project, completed during studies at BUT FIT.

## Our proposal

### Road segmentation 
Our goal is to build a CNN (Convolutional neural network) and compare it to ViT (Vision Transformer) for road segmentation in satellite images. We will primarily use PyTorch and NumPy libraries. 

#### Data 

We are going to use the DeepGlobe dataset, containing somewhere about 6000 training and 500 testing images and their masks. If necessary, we will acquire more datasets, as there are many available online. To name a few: Massachusetts Roads Dataset, SpaceNet. 

#### Design of networks  
we will draw inspiration from these models and papers: 
- [Graph Reasoned Multi-Scale Road Segmentation in Remote Sensing Imagery (git)](https://ieeexplore.ieee.org/document/10281660)
- [Road maps from Aerial Images](https://www.kaggle.com/code/vanvalkenberg/road-maps-from-aerial-images)
- [Road Segmentation in SAR Satellite Images With Deep Fully Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8447237)
- [Road Extraction from Satellite Images (DeepLabV3+)](https://www.kaggle.com/code/balraj98/road-extraction-from-satellite-images-deeplabv3)
- [RoadFormer: Pyramidal deformable vision transformers for road network extraction with remote sensing images](https://www.sciencedirect.com/science/article/pii/S1569843222001789)

#### Accuracy evaluation 

For the measurement of accuracy of our models we plan to compare predicated masks with reference masks using known metrics such as IoU (Intersection over Union) or mIoU (mean IoU). 

#### Division of work 

xnevor03 
- Image preparation - necessary scaling and augmentation 
- Training data forwarding for the learning algorithm 
- Accuracy analysis and output formatting 

xkovac61 
- Implementation of the CNN. 

xkrizd03 
- Evaluation and comparison of CNN and ViT (from mentioned paper). 

This division is only preliminary, and it is a subject of change. It is probable (and desirable) that all members of the team are going to actively contribute to the design and implementation of networks and to the final writing of documentation. 


## Response

Dear students,

your proposal is mostly OK. I just have some remarks and pointers:
* Keep a good track which code you write and which is taken from existing sources. Write specific and detailed comments in your source code.
* It is a good idea to use some library specificaly for semantic segmentation - architecture of models, loss functions, ...  --- I tend to use segmentation_models.pytorch https://github.com/qubvel-org/segmentation_models.pytorch
* For this topic, it is a good idea to run larger number of experiemnts. Where will you run the experiments? Will you have enoug GPU time for the experiments?
* You have to foolow the exact experimental protocol and have your results "compatible" with existring results on the dataset / challenge.
* Is the datasets "good"? When I've looked at it, I saw many mistakes and inacuracies in the annotations and no ground truth for validataion and testing sets. Can you find large number of papers using this dataset?

Regards,
Michal Hradiš 
