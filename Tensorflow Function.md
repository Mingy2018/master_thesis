## Tensorflow Function



### 数据读入

`tf.data.Dataset.from_tensor_slices(x)` :  将数据导入数据集

- **从 Numpy 导入**

  ```shell
  # create a random vector of shape (100,2)
  x = np.random.sample((100,2))
  # make a dataset from a numpy array
  dataset = tf.data.Dataset.from_tensor_slices(x)
  ```

- **从占位符导入**: 当我们希望动态地修改 Dataset 中的数据时

  ```shell
  x = tf.placeholder(tf.float32, shape=[None,2])
  dataset = tf.data.Dataset.from_tensor_slices(x)
  ```



`tf.train.string_input_producer`: 这个函数需要传入一个文件名list，系统会自动将它转为一个文件名队列。还有两个重要的参数，一个是num_epochs，它就是我们上文中提到的epoch数。另外一个就是shuffle，shuffle是指在一个epoch内文件的顺序是否被打乱。



`tf.data.Dataset`: **Dataset可以看作是相同类型“元素”的有序列表**。在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。Dataset是存储Tensor结构的类，它可以保存一批Tensor结构，以供模型来训练或者测试。



`tf.data.Dataset.map(map_func,num_parallel_calls=None)` : 

This transformation applies `map_func` to each element of this dataset, and returns a new dataset containing the transformed elements, in the same order as they appeared in the input. 简单的可以理解为，map的参数是一个函数，而这个函数的参数是固定的。就是和所使用dataset的内容是完全一致的。



`tf.stack(values,axis=0,name='stack')`: 将秩为 R 的张量列表堆叠成一个秩为 (R+1) 的张量.

```python
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
```



`dataset.make_one_shot_iterator()` : 从Dataset中示例化一个Iterator，然后对Iterator进行迭代。

```python
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
# create the iterator
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    print(sess.run(el)) # output: [ 0.42116176  0.40666069]
```







`tf.Session.run(self, fetches, feed_dict=None, options=None, run_metadata=None)`:  启动对所需要的数据流图的计算

- `fetches` ”取得之物“，表示数据流图中能接收的任意数据流图元素，各类Op/Tensor对象。Op.run()将返回None；Tensor.run()将返回Numpy数组

```python
import tensorflow as tf
from collections import namedtuple

a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
session = tf.Session()

v1 = session.run(a) #fetches参数为单个张量值，返回值为Numpy数组
print(v1)
>>> [10 20]
v2 = session.run([a, b]) #fetches参数为python类表，包括两个numpy的1维矩阵
print(v2)
>>> [array([10, 20], dtype=int32), array([ 1.,  2.], dtype=float32)]
v3 = session.run(tf.global_variables_initializer()) #fetches 为Op类型
print(v3)
>>> None
session.close()
```

- `feed_dict`: 给数据流图提供运行时数据。`feed_dict`的数据结构为python中的字典，其元素为各种键值对。"key"为各种Tensor对象的句柄；"value"很广泛，但必须和“键”的类型相匹配，或能转换为同一类型。

```python
import tensorflow as tf

a = tf.add(1, 2)
b = tf.multiply(a, 2)
session = tf.Session()
v1 = session.run(b)
print(v1)

replace_dict = {a:20}
v2 = session.run(b, feed_dict = replace_dict)
print(v2)
```

输出：

```shell
6
40
```





`tf.Variable()`: 用于生成一个初始值为initial-value的变量；必须指定初始化值。

`tf.get_variable()`: 获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。**可以用各种初始化方法，不用明确指定值。**

`tf.variable_scope()`: 可以让不同命名空间中的变量取相同的名字，无论`tf.get_variable`或者`tf.Variable`生成的变量

`tf.name_scope`: 具有类似的功能，但只限于`tf.Variable`生成的变量



 