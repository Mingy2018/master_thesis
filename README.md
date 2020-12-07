# master_thesis

### Dataset

- ShapeNet

- Pix3D

- ModelNet



### Files

**vae_vo_Andrew_oridata.py:** 

Reimplement of paper ["Generative and Discriminative Voxel Modeling with Convolutional Neural Networks"](https://arxiv.org/abs/1608.04236) 

Input voxel size [32,32,32]

no zero padding

no data augmentation

voxel value [-1, 2]



**vae_vo_Andrew_noidata.py:**

Reimplement of paper ["Generative and Discriminative Voxel Modeling with Convolutional Neural Networks"](https://arxiv.org/abs/1608.04236) 

Input voxel size [32,32,32]: [24, 24, 24] subvolume with object, 4 voxels of padding at each side

data augmentation(rotations and translations)

voxel value [-1, 2]



### Training

#### Set training data path

Set the `dataset_model_path` in **dataIO.py** with your dataset directory:

```python
dataset_img_path = '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/'
dataset_model_path = '/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetVox32/'
```

Set the path for saving training data in 

#### Start Training

```shell
python vae_vo_Andrew_oridata.py 0 0
```

Or

```
python vae_vo_Andrew_noidata.py 0 0
```



- While training, script will create a `train_data` folder and create a new folder(named by Time)  under it to save training data. **Move them to another place before pushing to GitHub**