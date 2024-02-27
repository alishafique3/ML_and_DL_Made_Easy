# PyTorch Training Optimizations using Memory Analysis

Training deep learning models, especially large ones, can be a costly expenditure. One of the main methods we have at our disposal for managing these costs is performance optimization. Performance optimization is an iterative process in which we consistently search for opportunities to increase the performance of our application and then take advantage of those opportunities. In previous posts (e.g., here) we have stressed the importance of having appropriate tools for conducting this analysis. The tools of choice will likely depend on a number of factors including the type of training accelerator (e.g., GPU, HPU, or other) and the training framework.

The focus in this post will be on training in PyTorch on GPU. More specifically, we will focus on the PyTorch’s built-in performance analyzer, PyTorch Profiler, and on one of the ways to view its results, the PyTorch Profiler TensorBoard plugin.

## Result
Android Device use for this project is Xiaomi Mi A2 with octacore processor and Adreno512 GPU. During benchmarking, 4 CPU threads are used. Runtime memory and model size are in MB while inference time is an average time in microseconds. 
| Optimization Technique        | Batch Size           | GPU Memory (GB)  | Avg. Step Time (ms)  | Samples per sec  | Optimization (rel. to base)  | 
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Base_Model      | 32 | 2.51 | 117.1      | 273.5 | 100% |
| Automatic Mixed Precision      | 32 | 1.32 | 81.1      | 395.1 | 144% |
| Increase Batch Size      | 128 | 4.95 | 232.1      | 551.5 | 201% |
| Reduce H2D Copy      | 128 | 4.95 | 202.6      | 631.63 | 230% |
| Multi-process Data Loading     | 128 | 4.95 | 145.8      | 877.84 | 320% |
| Memory Pinning     | 128 | 4.95 | 92.6      | 1381.6 | 505% |

## Baseline Model
For a while, I have been intrigued by one portion in particular of the TensorBoard-plugin tutorial. The tutorial introduces a classification model (based on the Resnet architecture) that is trained on the popular Cifar10 dataset. It proceeds to demonstrate how PyTorch Profiler and the TensorBoard plugin can be used to identify and fix a bottleneck in the data loader. 

![1_baseline_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3c871ec6-9aac-45c3-a306-7e43f5f65fe7)

![1_baseline_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3e1a7661-e364-4987-8b4a-1953b3081aa1)


## Optimization #1: Automatic Mixed Precision
The GPU Kernel View displays the amount of time that the GPU kernels were active and can be a helpful resource for improving GPU utilization:
![1_baseline_kernel_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/074249c0-cd1d-4002-8a9f-6a5522bcc151)

One of the most glaring details in this report is the lack of use of the GPU Tensor Cores. Available on relatively newer GPU architectures, Tensor Cores are dedicated processing units for matrix multiplication that can boost AI application performance significantly. Their lack of use may represent a major opportunity for optimization.

Being that Tensor Cores are specifically designed for mixed-precision computing, one straight-forward way to increase their utilization is to modify our model to use Automatic Mixed Precision (AMP). In AMP mode portions of the model are automatically cast to lower-precision 16-bit floats and run on the GPU TensorCores.

Importantly, note that a full implementation of AMP may require gradient scaling which we do not include in our demonstration. Be sure to see the documentation on mixed precision training before adapting it.

The modification to the training step required to enable AMP is demonstrated in the code block below.

```python
def train(data):
    inputs, labels = data[0].to(device=device, non_blocking=True), \
                     data[1].to(device=device, non_blocking=True)
    inputs = (inputs.to(torch.float32) / 255. - 0.5) / 0.5
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    # Note - torch.cuda.amp.GradScaler() may be required  
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
The impact to the Tensor Core utilization is displayed in the image below. Although it continues to indicate opportunity for further improvement, with just one line of code the utilization jumped from 0% to 26.3%.
![2_AMP_Kernel_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/183527fb-ec76-4067-ba18-8487113cf68d)

In addition to increasing Tensor Core utilization, using AMP lowers the GPU memory utilization freeing up more space to increase the batch size. Although the GPU utilization has slightly decreased, our primary throughput metric has further increased by nearly 50%, from 1670 samples per second to 2477. We are on a roll!

Caution: Lowering the precision of portions of your model could have a meaningful effect on its convergence. As in the case of increasing the batch size (see above) the impact of using mixed precision will vary per model. In some cases, AMP will work with little to no effort. Other times you might need to work a bit harder to tune the autoscaler. Still other times you might need to set the precision types of different portions of the model explicitly (i.e., manual mixed precision).

![2_AMP_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/5b53abad-6d16-441a-b382-7fc5efb4a9f0)

![2_AMP_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/7cfdad00-7d89-4203-b2b3-fe00d33bd2a8)


## Optimization #2: Increase Batch Size
The chart shows that out of 16 GB of GPU memory, we are peaking at less than 1 GB of utilization. This is an extreme example of resource under-utilization that often (though not always) indicates an opportunity to boost performance. One way to control the memory utilization is to increase the batch size. In the image below we display the performance results when we increase the batch size to 512 (and the memory utilization to 11.3 GB).
```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, 
                                           shuffle=True)
```
Although the GPU utilization measure did not change much, our training speed has increased considerably, from 1200 samples per second (46 milliseconds for batch size 32) to 1584 samples per second (324 milliseconds for batch size 512).
![3_BatchSize_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/7a085e4a-e81e-4562-a976-2c1133675383)
![3_BatchSize_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/d9b6cdab-d991-4b59-8924-1271e7173242)

Caution: Contrary to our previous optimizations, increasing the batch size could have an impact on the behavior of your training application. Different models exhibit different levels of sensitivity to a change in batch size. 

## Optimization #3: Reduce Host to Device Copy

## Optimization #4: Multi-process Data Loading

## Optimization #5: Memory Pinning

## Conclusion
In this project, different optimized models have been compared on android device. Dynamic quantization plays remarkably well among these optimized models. This project can be extended on different datasets, models and hardware to see the performance of optimization techniques.

## References
1.	Pyimagesearch Website: Fire and smoke detection with Keras and deep learning link: https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/
2.	Youtube Website: tinyML Talks: A Practical guide to neural network quantization link: https://www.youtube.com/watch?v=KASuxB3XoYQ 


