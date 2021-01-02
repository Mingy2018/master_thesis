# Keras Implementation of 3D-VAE

### Dataset

We use the [ShapeNetCore v2 (Fall 2016)](https://www.shapenet.org/download/shapenetcore) dataset, in the Github project we only provide the **chair class(03001627)** in the dataset for training and testing. If you want to train with many other different type of objects, the complete dataset is available in link.

There is 6778 elements in the chair class, the full object is available under `/dataset/03001627`. We also divide it into a train set and a test set, the `/dataset/03001627_train` folder consists of 5778 elements and the`/dataset/03001627_test` folder consists of 1000 elements. The `/dataset/03001627_test_1` consists of 100 objects from the 1000 test objects, **it is recommended** to use it while testing, if you want to generate images at the same time. If you need other configurations of the train/test set, you could create by yourself.

### Training

Set your configuration in the `train.py` includes hyper parameters, train dataset, save path etc, and start train:

```python
python train.py
```

### Test

After training the weights of models is saved as `.h5` file and used for testing, your need to set the weights file path and test dataset path and then run it. You should set the path  where save the reconstructed objects in `pathname` , and choose if generate the images (`1`) or not (`0`) for **visualization**.

```
python test.py pathname 1
```







