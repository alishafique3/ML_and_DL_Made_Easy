# PyTorch Training Optimizations using Memory Analysis

Training deep learning models, especially large ones, can be a costly expenditure. One of the main methods we have at our disposal for managing these costs is performance optimization. Performance optimization is an iterative process in which we consistently search for opportunities to increase the performance of our application and then take advantage of those opportunities. In previous posts (e.g., here) we have stressed the importance of having appropriate tools for conducting this analysis. The tools of choice will likely depend on a number of factors including the type of training accelerator (e.g., GPU, HPU, or other) and the training framework.

The focus in this post will be on training in PyTorch on GPU. More specifically, we will focus on the PyTorch’s built-in performance analyzer, PyTorch Profiler, and on one of the ways to view its results, the PyTorch Profiler TensorBoard plugin.

## Result
Android Device use for this project is Xiaomi Mi A2 with octacore processor and Adreno512 GPU. During benchmarking, 4 CPU threads are used. Runtime memory and model size are in MB while inference time is an average time in microseconds. 
| Optimization Technique        | Batch Size           | GPU Memory (MB)  | Avg. Step Time (ms)  | Samples per sec  | Optimization (rel. to base)  | 
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Base_Model      | 8.5 | 6719.84 | 11.4      | 5175.1 | 49.4 |
| Automatic Mixed Precision      | 2.14 | 8375 | 5.7      | 5075 | 43.34 |
| Increase Batch Size      | 4.26 | 7563 | 15.69      | 5173 | 45.34 |
| Reduce H2D Copy      | 2.14 | 8897 | 5.98      | 5214 | 43.35 |
| Multi-process Data Loading     | 2.14 | 8032 | 5.88      | 5057.1 | 43.32 |
| Memory Pinning     | 2.14 | 8032 | 5.88      | 5057.1 | 43.32 |

## Baseline Model
For a while, I have been intrigued by one portion in particular of the TensorBoard-plugin tutorial. The tutorial introduces a classification model (based on the Resnet architecture) that is trained on the popular Cifar10 dataset. It proceeds to demonstrate how PyTorch Profiler and the TensorBoard plugin can be used to identify and fix a bottleneck in the data loader. 

![1_baseline_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3c871ec6-9aac-45c3-a306-7e43f5f65fe7)

## Optimization #1: Automatic Mixed Precision

## Optimization #2: Increase Batch Size

## Optimization #3: Reduce Host to Device Copy

## Optimization #4: Multi-process Data Loading

## Optimization #5: Memory Pinning

## Conclusion
In this project, different optimized models have been compared on android device. Dynamic quantization plays remarkably well among these optimized models. This project can be extended on different datasets, models and hardware to see the performance of optimization techniques.

## References
1.	Pyimagesearch Website: Fire and smoke detection with Keras and deep learning link: https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/
2.	Youtube Website: tinyML Talks: A Practical guide to neural network quantization link: https://www.youtube.com/watch?v=KASuxB3XoYQ 


