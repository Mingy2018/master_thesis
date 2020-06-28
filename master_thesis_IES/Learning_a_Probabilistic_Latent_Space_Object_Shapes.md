## Learning a Probabilistic Latent of Object Shapes via 3D Generative-Adversarial Modeling

### Abstract

The paper introduce a novel way (3D-GAN)for learning latent feature from 3D objects and then reconstruct the 3D objects, the method is unsupervised and outperforms among many the-state-of-the-art method. 

The reconstructed objects are novel and relative highly realistic, by analyzing the latent space by three ways, it shows that the network gives semantic information, which can be used as informative feature representation of 3D object.

### Research Objective

varied and realistic 3D modeling

### Background and Problems

* traditional 3D modeling: synthesis of existing parts from database, not conceptual novel
* many deep learning networks for 3D data is supervised
* GANs are previously used to generate 2D images

### Method

* 3D Generative Adversarial Network (3D-GAN)
  * adaptive training strategy: update the discriminator selectively:
    make training stable and produce better results
* 3D VAE GAN
  * adding a encoder of 2D images to extract the latent representation vectors

### Evaluation

* 3D object generation
  
  * qualitative comparison of 3D generations by three methods: 
  
    the 3D-GAN, volumetric Autoencoder, from literature
* 3D object classification
  * outperforms among unsupervised method by large margin
  * best among normal supervised methods
  * close to the state-of-the-art voxel-based supervised learning approach
* single image 3D reconstruction
  * outperforms previous methods
  * separately trained network more precise than jointly trained network

### Analyze & Conclusion

* three methods for understanding the **generative representation**
  * visualize the object vector: some vectors carry the **semantic information**
  * interpolation between two object vectors: smooth transitions between objects
  * arithmetic in latent space:  semantic information
* discriminative representation
  * for single neuron: strongest activation = similar shape
  * parts activate neuron: semantic information

#### New Unknown Concepts

- [ ] traditional heuristic criteria
- [ ] binary cross entropy











