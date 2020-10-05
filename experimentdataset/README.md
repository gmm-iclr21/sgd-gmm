# Experiment Datasets Project

This project provide a semi automatic construction method for different datasets that can easily be included as submodule in other projects.

* [CIFAR-10](cifar10.py)
* [CUB200](cub200.py)
* [Devnagari](devnagari.py)
* [EMNIST](emnist.py)
* [FashionMNIST](fashionmnist.py)
* [FRUITS](fruits.py)
* [MADBase](madbase.py)
* [MNIST](mnist.py)
* [notMNIST](notmnist.py)
* [SIMPSONS](simpsons.py)
* [SVHN](svhn.py)

## Add a new Dataset
Each dataset class inherit from the abstract base class `Dataset` which provide abstract properties, abstract methods and default implementations for the creation of datasets.

The abstract base class `Dataset` provide a template pattern to create a dataset step by step (if the `load_dataset` method is invoked by a model).

1. if the dataset was already prepared (existing pickle-file), it will be loaded.

Otherwise (you need 16 GB of RAM for the construction of some dataset):
2. the `download_data` method is invoked
3. the `extract_data` method is invoked
4. the `prepare_data` method is invoked, all data from self.data_train, self.labels_train, self.data_test and self.labels_test were used for further steps
5. if no explicit test data exists, a percent value (`SPLIT_TRAINING_TEST` default is 10%) of the training data were extracted and stored as test data.
6. training and test data were shuffled (seeded)
7. data_train and data_test were flattened
8. normalization is applied to data_train and data_test. The VALUE_RANGE is used for normalization
9. the labels for training and testing were converted to one-hot-vector
10. the converted data were stored in a pickle file as a dictionary:
        'data_train':   data_train,
        'labels_train': labels_train,
        'data_test':    data_test,
        'labels_test':  labels_test,
        'properties':   properties
    }
11. the `clean_up` method is invoked
12. if the inherited dataset class do not overwrite the validate_shape property, the validation mechanism is invoked after the dataset is restored from disk.
13. the loaded data get prepared by the given parameter in the FLAG variable, e.g., permute data, exclude classes.
Further the dataset is converted to the DataSet class provided by TensorFlow.

##### An example is given for fashionmnist:

Following abstract properties have to be overwritten by the dataset class:
 * `name` (name of the dataset, should be the same name as the class)
 * `shape` (shape of the input data, will be used for reshaping)
 * `pickle_file` (the pickle filename were the data are stored after the transformation)
 * `num_train` (number of training elements, should match for verification)
 * `num_test` (number of test elements, should match for verification)
 * `num_classes` (number of classes, should match for verification)

Following abstract methods must be implemented (could be empty if not needed)
 * `download_data` (download the needed files for further preparation of the data, default download mechanisms get provided by the Downloader class)
 * `extract_data` (decompress and load data into memory, default decompress mechanism is provided by Data_Loader class)
 * `prepare_data` convert method for data in memory, must store data in self.data_train, self.labels_train, self.data_test and self.labels_test for further steps
 * `clean_up` (remove not needed files for dataset creation)

## Usage

### Simple Dataset (_deprecated_)

1. Add this project by cloning it `git clone` or adding as submodule `git submodule add`.
2. Import of the `get_dataset` method and optional the `Dataset_Type` e.g.:

```
from experimentdataset import load_dataset, Dataset_Type
```

3. To load the dataset in form of multiple `tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet` (TF_DataSet) objects and a properties dictionary, call the `load_dataset` method with an "filled" `argparse.Namespace` object as parameter.
Needed parameter of the Namespace object:
    + `dataset_file`:
        * `MNIST.pkl.gz       `  :  17MB [209MB extracted]
        * `FashionMNIST.pkl.gz`  :  42MB [222MB extracted]
        * `CIFAR10.pkl.gz     `  : 186MB [739MB extracted]
        * `NotMNIST.pkl.gz    `  : 108MB [1,7GB extracted]
        * `MADBase.pkl.gz     `  :  17MB [220MB extracted]
        * `CUB200.pkl.gz      `  :  20MB [ 69MB extracted] (only 10 classes)
        * `Devnagari.pkl.gz   `  :  56MB [393MB extracted] (only 10 classes)
        * `EMNIST.pkl.gz      `  : 309MB [2,8GB extracted] (62 classes)
        * `Fruits.pkl.gz      `  : 176MB [780MB extracted] (only 10 classes)
        * `Simpsons.pkl.gz    `  : 122MB [577MB extracted] (only 10 classes)
        * `SVHN.pkl.gz        `  : 357MB [1.2GB extracted] (only 10 classes)
    + `dataset_dir`: Directory for storing input data (pkl.gz files)
    + `permuteTrain`: Provide random seed for permutation train
    + `permuteTrain2`: Provide random seed for permutation train2
    + `permuteTest`: Provide random seed for permutation test
    + `permuteTest2`: Provide random seed for permutation test2
    + `permuteTest3`: Provide random seed for permutation test3
    + `mergeTrainInto`: merge train set arg into train set arg2
    + `mergeTestInto`: merge test set arg into train set arg2
    + `mergeTest12`: merge sets test and test2 to form test3
    + `mergeTrainWithPermutation`: merge train set and permuted train set
4. Return structure: train1 (`TF_DataSet`), train2 (`TF_DataSet`), test1 (`TF_DataSet`), test2 (`TF_DataSet`), test3(`TF_DataSet`), properties (`dict`)

### Incremental Dataset Tasks
1. Add this project by cloning it `git clone` or adding as submodule `git submodule add`, `git submodule init` and `git submodule update`.
2. Import of the `Dataset_Wrapper` class with:

```
from experimentdataset import Dataset_Wrapper, Dataset_Type as DT
```

3. To load the dataset in form of multiple `tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet` (TF_DataSet) objects and a properties dictionary, call the `load_dataset` method with an "filled" `argparse.Namespace` object as parameter.
Needed parameter of the Namespace object:
    + `dataset_file`: load a compressed pickle file. If not present, a download attempt is made (select one of Dataset_Type)
       * SVHN
       * MNIST
       * CUB200
       * EMNIST
       * Fruits
       * CIFAR10
       * MADBase
       * NotMNIST
       * Devanagari
       * FashionMNIST
4. Use the `Namespace`-object as parameter to create an object of the `Dataset_Wrapper` class.
5. TODO: use `Namespace`-object to configure slicing e.g.: [-1, -1] full,  [6, 6] only the center $6\times6$ pixel patch
5. Use `get_properties()` to get a properties-dictionary of the dataset loaded dataset
6. You get a training and a test dataset (`tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet`) by calling `get_dataset(list)`, whereby the `list` specifies the containing data elements e.g. `[0, 1, 2]`
7. To get the number of training iterations (for one epoch, based on Namespace-object `batch_size` respectively `test_batch_size`) for a specific task call `get_iter(list)` , whereby the `list` specifies the containing data elements e.g. `[0, 1, 2]`
