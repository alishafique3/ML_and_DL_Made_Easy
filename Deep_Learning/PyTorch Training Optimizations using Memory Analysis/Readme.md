# PyTorch Training Optimizations using Memory Analysis

Training deep learning models, especially large ones, can be a costly expenditure. One of the main methods we have at our disposal for managing these costs is performance optimization. Performance optimization is an iterative process in which we consistently search for opportunities to increase the performance of our application and then take advantage of those opportunities. In previous posts (e.g., here) we have stressed the importance of having appropriate tools for conducting this analysis. The tools of choice will likely depend on a number of factors including the type of training accelerator (e.g., GPU, HPU, or other) and the training framework.

The focus in this post will be on training in PyTorch on GPU. More specifically, we will focus on the PyTorch’s built-in performance analyzer, PyTorch Profiler, and on one of the ways to view its results, the PyTorch Profiler TensorBoard plugin.

## Baseline Model
For a while, I have been intrigued by one portion in particular of the TensorBoard-plugin tutorial. The tutorial introduces a classification model (based on the Resnet architecture) that is trained on the popular Cifar10 dataset. It proceeds to demonstrate how PyTorch Profiler and the TensorBoard plugin can be used to identify and fix a bottleneck in the data loader. 

![1_baseline_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3c871ec6-9aac-45c3-a306-7e43f5f65fe7)


