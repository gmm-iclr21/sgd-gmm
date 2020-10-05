Code for Gradient-Based Training of Gaussian Mixture Models for High-Dimensional Streaming Data (NIPS2020)

### Dependencies
Tested with Python 3.7+ and tensorflow-gpu 1.14 (Linux and Windows)

Additional packages besides Python3 standard lib: numpy, matplotlib, scipy
Can all be installed via pip, e.g. `python3 -m pip install numpy`

### Getting Started
0. Install python3
1. Install tensorflow (with pip)
2. Install other dependencies (with pip)
3. Start run with default parameters `python GMM.py` or `python3 emAlgos.py --taskEpochs 1 --nrTasks 1`

### Datasets
The module 'experimentdataset' provides download facilities for all datasets used in the paper. 
This is controlled by the cmd line parameter '--dataset_file' for all programs in this archive. Valid arguments include:
MNIST, FashionMNIST, NotMNIST, Devanagari, SVHN, Fruits, ISOLET (see experimentdataset doc for the full list). 
If a dataset is not already cached locally, it is downloaded automatically and cached (which can take a while for some datasets liek Fruits or SVHN). 
For some datasets, a kaggle account is required. 
MNIST and Devanagari are included/cached in this archive already.

### Run SGD
The main file is GMM.py. To change default parameters, you can specify command line parameters. Use `python3 GMM.py --help` to get an overview.
For example, to change the number of used Gaussian components: `python3 GMM.py --K 100`
Or work with FashionMNIST: `python3 GMM.py --K 100 --dataset_file FashionMNIST`

### Run sEM
The main file is emAlgos.py which can perform EM and sEM with a variety of options, see `python3 emAlgos.py --help`
For example, to change the number of used Gaussian components: `python3 emAlgos.py --mode sEM --taskEpochs 1 --nrTasks 1  --n 100`
Or work with FashionMNIST: `python3 emAlgos.py --mode sEM --n 100 --taskEpochs 1 --nrTasks 1 --dataset_file FashionMNIST`

### Visualization
GMM.py and emAlgos.py write the current GMM weights, centroids and precision matrices to mus/pis/sigmas.npy. 
Visualize centroids and pis with `python3 vis.py` which generates a file mus.png that can be displayed. 
Standard incovation to visualize centroids: `python3 vis.py` 
See `python3 vis.py --help` for options. 

### Logging
Both GMM.py and emAlgos.py generate .json files which contain all relevant information about an experiment. Most notably, they 
contain the log-likelihoods at various points in time, measured on all relevant datasets. 
The Name of the log file is derived from the arguments '--tmp_dir' and '--exp_id'

### Streaming experiments
The files sem.bash and gmm.bash contain some sample invocations both for sEM and GMM from the streaming experiments in the paper.
Under Linux, you can execute these files by typing, e.g., `source sem.bash`
Log files from these experiments will be plaed in the subdirectory 'ExpDist'. 

### Concept drift experiments
The files sem_inc.bash and gmm_inc.bash contain some sample invocations both for sEM and GMM from the concept drift experiments in the paper.
Under Linux, you can execute these files by typing, e.g.,  `source sem_inc.bash`
Log files from these experiments will be placed in the subdirectory 'ExpDist'. 
