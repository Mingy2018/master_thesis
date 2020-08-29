### Variational Autoencoders

VAE is a neural network made up of two parts:

- An *encoder* network that compresses high-dimensional input data into a lower-dimensional representation vector
- An *decoder* network that de-compress a given representation vector back to the origin domain

The network is trained to find weights for the encoder and decoder that minimize the loss between the original input and the reconstruction of the input after it has passed through the encoder and decoder.



#### The Loss Function

*Kullback-Leibler (KL) divergence(KL散度):*  $D_{KL}(p||q)=\sum_{i=1}^np(x_i)log(\frac{p_{x_i}}{q_{x_i}})$

- 定量地确定哪种已有分布能更好地解释真实分布: ask which distribution preserves the most information from our original data source
- help to measure how much information we lose when we choose an approximation
- $p$ is our probability distribution, $q$ is the approximating distribution
- KL Divergence is **not a distance metric**, because KL Divergence is not *symmetric*
- 



*Information Entropy:* $H=-\sum_{i=1}^Np(x_i)\cdot logp(x_i)$



- [ ] the minimum number of bits it would take us to encode our distribution

- theoretical lower bound of needed bits

  

### Generative Adversarial Networks





