# Keras Implementation of 3D-VAE

### Packages

```
python 3.6
tensorflow-gpu 1.13.1
matplotlib 3.3.2
scikit-image 0.17.2 
```

### Dataset

We use the [ShapeNetCore v2 (Fall 2016)](https://www.shapenet.org/download/shapenetcore) dataset, in the Github project we only provide the **chair class(03001627)** in the dataset for training and testing. If you want to train with many other different type of objects, the complete dataset is available in link.

There is 6778 elements in the chair class, the full object is available under `/dataset/03001627`. 

We also divide it into a train set and a test set, the `/dataset/03001627_train` folder consists of 5778 elements and the`/dataset/03001627_test` folder consists of 1000 elements.

 If you want to generate object 3d images at the same time, **it is recommended** to use this `/dataset/03001627_test_img`, which consists of 100 objects from the 1000 test objects, because we use CPU to generate 3d images, it takes about 10 minutes to test and generate images on 100 objects.

### Training

Set your configuration in the `train.py` includes hyper parameters, train dataset, **save path** etc, and start train:

```python
python train.py
```

### Test

After training the weights of models is saved as `.h5` file and used for testing, your need to **set the weights file path** and test dataset path. You should also set the path where save the reconstructed objects in `pathname` , and choose if generate the images (`1`) or not (`0`) for **visualization**.  Then run the test script.

While testing, the script will save both the original data and generated data.

```python
python test.py pathname 1
```







