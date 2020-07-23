![img](https://raw.githubusercontent.com/Mingy2018/master_thesis/master/pics/paper/1.PNG)

关键词: Probalistic latent space;  3D modeling

**摘要：**

本文提出了一个新框架用于**无监督式**地学习3D物体特征和重建3D物体，该框架做了三点微小的工作：1. 通过对抗学习，生成器可以隐式地学习到3D物体的结构并重建；2. 生成器可以直接利用*特征空间(latent space)* 生成3D物体，并不需要数据库中的模块或参考图片，这为探索构建3D物体流形(manifold) 提供了可能；3. 对抗网络中的判别器提供了3D物体很好的描述，并且这些描述可以通过无监督学习获得，这对3D物体识别 (obeject recognition) 也会有广泛应用。

该框架不仅可以**高质量地**重建的3D物体，在3D物体识别中，即使同**监督式**的方法相比，也能获得优秀的表现。

**一、简介**

作者认为，一个好的3D生成模型需具备的能力是：可以多变化(highly varied)和高保真(realistic)的构建3D物体。

过去关于3D重建的方法中，很多基于骨架和网格；许多传统方法则是从数据库中已有的例如CAD模型来构建3D物体，它们看起来很真实，但并不是概念上新颖(not conceptually novel)。作者应该认为，将已有的部分重新组合不是好的重构。-> not highly varied

近来，随着深度学习的发展，有了许多基于体素(voxels) 3D重建尝试，这是一个挑战性的问题，3D数据维数很多，当前的效果很有前景，但是重建的3D物体经常会缺失、失真。-> not realistic

由于存在的这些问题，作者提出了新的框架，它结合了对抗学习和卷积网络的优点。针对不同的应用任务，学习不同的特征表示(representation=latent space)。**生成式表示**可以很好地重建3D物体，**判别式表示**可以用来进行3D识别。除此之外，通过探索实验，这两种表示都含有了丰富的**语义信息**。

**二、相关工作**

三维重建：再次声明属于**无监督式**的模型

基于三维数据的深度学习：以往许多方式也是**监督式**的；

对抗网络学习：在2D数据中已经有实现

**三、模型**

本章将介绍3D-GAN模型和VAE模型，其中3D-GAN用于生成三维物体，VAE模型用于利用2D图片生成三维物体。

**3D-GAN:**  生成对抗网络由一个生成器和一个判别器组成:



<img src="https://raw.githubusercontent.com/Mingy2018/master_thesis/master/pics/paper/3.PNG?token=AJPFENSO6DQLZMP5NFAMJJ27CI6CE" alt="img" style="zoom: 50%;" />

其中生成器的具体结构如下：判别器的结构和生成器是镜像的，除了其最终输出为一个值(判断input真假情况)。

![img](https://pic1.zhimg.com/v2-22d19bbd6e7b0a0a0a540d442d317714_b.png)

**训练技巧:** 文章发现，可能由于生成三维物体要比判断三维物体真假要困难，故判别器学习得比生成器要快很多。如果判别器过早的学习得太好，生成器很难从判别器得到有效的提升反馈，所以设定一个训练策略，学习一个batch后，只有判别器正确率小于80%，我们才更新判别器的参数。通过这个方法，可以观察到训练过程变得更稳定，结果也更好了。

**3D-VAE-GAN:**

上面的GAN网络展示了由特征空间 **z** 生成三维物体，下面介绍学习**特征空间**(latent space)的网络。在3D-GAN的基础上，添加一个Encoder结构，对其输入2D图片，输出特征空间 **z** 。



<img src="https://raw.githubusercontent.com/Mingy2018/master_thesis/master/pics/paper/4.PNG?token=AJPFENRFVRGPHM4LRWMFXRK7CI6RC" alt="img" style="zoom:50%;" />



**Loss** 函数分析：

$L_{3D-GAN}=logD(x)+log(1-D(G(z)))$, 这部分设计基于交叉熵。$D(x)$是判别器基于一个3D输入$x$ 输出的值，介于0到1之间，越高说明该3D物体为真；由于$log()$是增函数，则该Loss的目标是使得$D(x)$尽可能的小，$D(G(z))$尽可能地大。

$L_{KL}=D_{KL}(q(z|y)||p(z))$, 该Loss使$q(z|y)$ 的分布尽可能的接近已有分布$p(z)$。

$L_{recon}=||G(E(y))-x||_2$, $x $代表训练集中的三维物体表示(体素张量)，$y$代表2D图片, $E(y)$为Encoder提取的特征，$G(E(y))$则是基于提取特征生成的三维物体表示(体素张量)。该Loss代表重建三维模型(estimate)和真实三维模型(ground-truth)的$l_2$距离。

**四、评估**

文章从不同的角度来评估了框架，首先，展示了定性地展示了生成的3D物体；定性和定量地然后评价了由判别器进行无监督式学习得到的特征空间，该特征向量可用于3D物体分类；最后，定性和定量地评估了基于2D图片利用3D-VAE-GAN生成3D物体。



**五、分析**

**六、结论**