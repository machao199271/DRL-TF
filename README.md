# DRL-TF

This is the official clone for the implementation of Distributed Representation Learning for Trained Forests (DRL-TF) by Skip-Gram Model[1]. The implementation is flexible enough to modify the model or fit your datasets.

Reference: [1] Ma, Chao and Wang, Tianjun and Zhang, Le and Cao, Zhiguang and Huang, Yue and Ding, Xinghao, Distributed Representation Learning for Trained Forests by Skip-Gram Model. Available at SSRN: https://ssrn.com/abstract=4223514

## How to use

Before running this code, you should replace your Sklearn package with Sklearn we offered in 'package'. 'package' contains both the Win and Linux code of Sklearn. The version of Sklearn we recommend is scikit-learn=0.23.2=py37h47e9c7a_0. For more versions of python packages, you can find them in 'environment.yml'.

'forest2vec.py' is the method we proposed called DRL-TF. And 'utils.py' is the utils for experiments. 'circle.py' is the experiment for the circle dataset. The results are summarized in 'results'.
