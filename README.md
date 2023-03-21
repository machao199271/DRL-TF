# DRL-TF

This is the official clone for the implementation of Distributed Representation Learning for Trained Forests (DRL-TF) by Skip-Gram Model[1]. The implementation is flexible enough for modifying the model or fit your own datasets.

Reference: [1] Ma, Chao and Wang, Tianjun and Zhang, Le and Cao, Zhiguang and Huang, Yue and Ding, Xinghao, Distributed Representation Learning for Trained Forests by Skip-Gram Model. Available at SSRN: https://ssrn.com/abstract=4223514

## Files description
'data.csv' is cleaned data of P2P car sharing, note that, not all features are used in the demo. And other datasets can be found in Kaggle.

'costsensitive' is a necessary package for cost-sentive learning.

'run_on_server.py' contains the modified K-means we proposed and some other functions which will be used in the following step.

'cost_sensitive_deep_forest.py' is the method we proposed and traditional deep forest.

'random_forest.py' and 'rotation_forest.py' are the traditional methods. Note that, the code of SVM and MLP is not offered because of their bad performance.
