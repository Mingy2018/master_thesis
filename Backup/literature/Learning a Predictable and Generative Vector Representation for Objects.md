## Learning a Predictable and Generative Vector Representation for Objects - Aug 2016

### Abstract

The paper introduces a novel framework to learn the good representation of objects, the representation is **generative in 3D** and **predictable from 2D**. And also the representation can be inferred from both voxels and images. Extensive experiment shows the representation demonstrates the usefulness and versatility.

### Research Objective

joint object representation(feature vector)

### Background and Problems

* features generated from **discriminative CNN**: 
  * not incorporate 3D information
  * no strong generative capabilities
* learn 3D model to images: no offer to the real world information (for interacting)
* current 3D CAD model learning & generation:
  * no guarantees that their representations can be inferred from images
* previously no mapping from images to 3D structure

### Approach

* novel TL framework to join the features from voxels and images
  * autoencoder: learn the 3D voxels
  * discriminatively trained ConvNet: **learn the 2D images into 64D embedding space**
  * Euclidean loss function: Euclidean distance?
* a series of experiments on the nature of the representation

### Evaluation

- Dataset:
  - ShapeNet: Training  with rendered images and voxels
  - IKEA Dataset: quanlify the performance of the model on natural indoor images
- Embedding Analysis
  - Autoencoder vs PCA: AE outperforms with **higher confidence** as well as **fine details**, gets the extent correct
  - The learned space is smooth: linear interpolation; explore latent space
  - Understand the embedding space: scaling one feature while keeping others constant (固定变量法) -> consistent effect, semantically meaningful
  - Reconstruction accuracy: the auto-encoder is worthwhile
  - class-discriminative despite not being trained or design so
- Voxel Prediction
  - model automatically predicts occluded region of the object
  - directly predict the voxels(not doing joint training) does worse
- CAD Retrieval
  - 



### Conclusion



### Notes

1. how to ensure the representations from voxels and images are consistent?

   how to joining them?

   这两个标准常常是互相冲突的，用于重建3D体素的信息很难从2D图片学习到，之前的方法，常常是基于其中的一个标准进行优化学习。本文提供了一种新的框架，可以基于这两种标准，同时进行优化学习。

2. other concepts

- [ ] freezing the layers
- [x] discriminatively trained CNN
- [ ] cosine distance













