`.binvox` files:

data is either a three-dimensional numpy boolean array (dense representation) or a two-dimensional numpy float array (coordinate representation).

`.mat` files

`.obj` file (Wavefront):

The OBJ file format is a simple data-format that represents 3D geometry alone.



## 3D Dataset

### [ShapeNet](https://www.shapenet.org/)

**ShapeNetCore:** A densely annotated **subset of ShapeNet** covering 55 common object categories with ~51,300 unique 3D models.

fotmat:  .obj + .mtl +

**ShapeNetCore v1 (July 2015):**

*note:* some models are mirrored and have incorrect normals.

data format: .obj + .mtl



**ShapeNetCore v2 (Fall 2016):** An update to ShapeNetCore with improved quality of model geometry and fixed issues relating to materials and textures.

data format:  .obj + .mtl  + .binvox

Introduction of `.binvox` format: https://www.patrickmin.com/binvox/

There is the script which transforms `.obj`  to `.binvox`.

因为整个数据集有25GB，

**可以在按照编号下载各子类：**

- http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2/

- 压缩包：http://shapenet.cs.stanford.edu/shapenet/obj-zip/

编号-类别索引：https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e



### [ModelNet](https://modelnet.cs.princeton.edu/)



## Paper data format

1. [Probabilistic Reconstruction Networks for 3D Shape Inference from a Single Image](https://arxiv.org/pdf/1908.07475v1.pdf) - Aug 2019
   数据集：ShapeNet

   处理方式：直接读取[.binvox](https://github.com/Regenerator/prns/blob/master/process_ShapeNetAll.py) line 70，然后预处理成[ .h5 ](https://github.com/Regenerator/prns/blob/master/process_ShapeNetAll.py)

2. 偶然找到的转换脚本`.mat`和`.binvox`

3. 





