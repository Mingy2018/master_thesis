## 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction

### Abstract

The paper proposed a novel framework to achieve robust 3D reconstruction, the networks learns a mapping from images of objects to their underlaying 3D shapes from a large collection of synthetic data. The network does not require any image annotations or object class labels for training or testing(**unsupervised**). Extensive experiments shows that the network outperforms the state-of-the-art methods for single view reconstruction.

### Research Objective

Single and Multi-view 3D Object Reconstruction

### Background and Problems

* most of the state-of-the-art methods for 3D reconstruction have **restrictions:**
  * requires **dense** number of views: **feature correspondences**
  * object appearances(Lambertian, albedos requirements) : feature matching
* shape prior-based methods:
  * work with fewer images and fewer assumptions on object reflectance

### Approach

- **3D-R2N2(3D-Recurrent Reconstruction Neural Network)**
  - inspired from LSTM
  - selectively update hidden representation using *input* & *output* gates
- 3D-R2N2 **Structure**
  - Encoder(2D-CNN): encodes input images into low dimensional features
  - 3D-LSTM: given encodes features, update cell states or retain states
    - effectively handle object self-occlusions when multiple views are fed to the network
  - Decoder(3D-DCNN): decodes hidden states and generate 3D probabilistic voxel reconstruction
- Data augmentation
  - PASCAL VOC 2012
- Training
  - number of views for each training example within a single mini-batch was kept constant, but **different** across mini-batched
    - A unified approach - 见题目

### Evaluation

- 3D-R2N2 Variations
  - Res3D-GRU-3 works best according to loss and IoU 
- single real-world image reconstruction: **3D-R2N2 has better performance and requires no data labels**
  - 直观的图片对比
  - IoU quantitative comparison
- Multi-view Reconstruction
  - reconstruction quality improves as the number of views increases
  - highest reconstruction performance on **bulky shaped class**(less shape variance), worse on higher shape variation class
- Reconstruction Real World Image
  - capable
- 3D-R2N2 vs. MVS
  - comparison configurations: various texture level and different number of views
  - MVS result
    - failed to reconstruct when the number of views less than 20
    - failed to reconstruct low texture level objects
  - paper result
    - good generalization performance
    - limitations
      - not as many details in some situations
      - performs worse in reconstructing object with high texture levels

### Conclusion

The paper proposed a unified framework for reconstructing 3D object, it can take variable input length. It also outperforms other single-view reconstruction method. Furthermore, the reconstruction quality increases when more views of object are given. It can also reconstruct object when other methods failed, while do not require a minimum number of input images.

### Notes

- [ ] IoU
- [ ] Lambertian (non-reflective)











