# PyTorch Training Optimizations using Memory Analysis

Training optimization techniques are critical in machine learning because they enhance efficiency, speed up convergence, ensure stability, improve generalization, and enable scalability. These techniques are essential for developing effective models that perform well on various tasks and datasets while making efficient use of computational resources.

The focus of this tutorial will be on the optimization of the training stage on a single GPU using PyTorch framework. We will use PyTorch’s built-in performance analyzer, PyTorch Profiler, and the PyTorch Profiler TensorBoard plugin to analyze the performance of the training stage. The optimization techniques used in this tutorial are Automatic mixed precision, increased batch size, reduced H2D copy, multiprocessing, and pinned memory to improve training time and memory usage. 

## Usage:
The code is built using the NVIDIA container image of Pytorch, release 23.10, which is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). The code is built using the following libraries:

- Ubuntu 22.04 including Python 3.10
- NVIDIA cuDNN 8.9.5
- PyTorch 23.10
  
For Docker 19.03 or later, a typical command to launch the container is:
```
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:xx.xx-py3
```
For Docker 19.02 or earlier, a typical command to launch the container is:
```
nvidia-docker run -it --rm -v nvcr.io/nvidia/pytorch:xx.xx-py3
```
Where:
- xx.xx is the container version that is 23.10

## Baseline Model
Pytorch example "PyTorch Profiler With TensorBoard" is used as base code which is available [Link](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) accessed on February 2, 2024.

The tutorial has used a classification model (based on the Mobilenet_V2 architecture) that is trained on the popular CIFAR10 dataset. PyTorch Profiler and the PyTorch TensorBoard plugin are used to identify a bottleneck in the training step. 

![1_baseline_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3c871ec6-9aac-45c3-a306-7e43f5f65fe7)
![1_baseline_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3e1a7661-e364-4987-8b4a-1953b3081aa1)
As we can see, the step time is 117 msec with the GPU utilization is 73.28%. The average memory used in each training step can be found in the "Memory View" window. The base model uses almost 2.5GB in each training step. Some important terminologies regarding PyTorch GPU summary are explained below:

**GPU Utilization/ GPU busy time:** It is the time during “all steps time” when there is at least one GPU kernel running on this GPU. The higher, the better. However, It can’t tell how many SMs(Stream Multiprocessors) are in use. For example, a kernel with a single thread running continuously will get 100% GPU utilization.

**Est. SM Efficiency:** Estimated Stream Multiprocessor Efficiency. The "Estimated Stream Multiprocessor Efficiency" typically represents the ratio of the actual computational work being performed by the stream multiprocessors to the maximum possible work they could perform under ideal conditions. In other words, it measures how effectively the SMs are utilized during the execution of a parallel workload. A high SM efficiency indicates that the GPU is effectively utilizing its parallel processing resources. Monitoring and optimizing SM efficiency are essential for maximizing the performance and throughput of parallel computing tasks on GPUs.

**Est. Achieved Occupancy:** In parallel computing, occupancy refers to the ratio of active warps (threads) to the maximum possible number of warps that can be resident on a streaming multiprocessor (SM) at a given time. provides insight into how effectively the GPU kernel is utilizing the available hardware resources. A high achieved occupancy indicates that a large portion of the GPU's processing resources is actively utilized, which can lead to better performance. Monitoring and optimizing achieved occupancy are important for maximizing the performance of GPU-accelerated applications. Techniques such as optimizing thread block size, memory access patterns, and kernel execution configuration can help improve achieved occupancy and overall performance on GPU devices.

## Optimization #1: Automatic Mixed Precision
An interesting observation in the GPU kernel view is the minimal utilization of the GPU Tensor Cores. These components, found in more recent GPU designs, serve dual functions. Firstly, they act as specialized units for performing matrix multiplication, which greatly enhances the performance of AI applications. Secondly, Tensor Cores deliver significant performance improvements over conventional GPU cores through their use of mixed-precision arithmetic operations.
![1_baseline_kernel_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/074249c0-cd1d-4002-8a9f-6a5522bcc151)

In Automatic Mixed Precision (AMP) training, specific sections of the model are automatically converted to lower-precision 16-bit floats and executed on the GPU TensorCores. This approach delivers notable computational acceleration by performing operations in half-precision format while storing minimal data in single-precision to preserve essential information in critical parts of the network. With the introduction of Tensor Cores in the Volta and Turing architectures, significant speed enhancements have been achieved through mixed precision.

Implementing mixed precision training involves three key steps:

- Converting applicable parts of the model to utilize the float16 data type.
- Retaining float32 master weights to accumulate weight updates per iteration.
- Employing loss scaling to maintain small gradient values.


The modification to the training step for AMP is shown in the code block below.

```python
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
    #outputs = model(inputs)
    #loss = criterion(outputs, labels)
    # Note - torch.cuda.amp.GradScaler() may be required
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
The impact on the Tensor Core utilization is displayed in the image below. With just a few lines of code, the utilization jumped from 0% to 19%.
![2_AMP_Kernel_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/183527fb-ec76-4067-ba18-8487113cf68d)

In addition to increasing Tensor Core utilization, AMP has also lowered the GPU memory utilization freeing up more space to increase the batch size. The throughput metric in the training phase has also increased from 273 samples per second (117 milliseconds for batch size 32) to 395 (81 milliseconds for batch size 32).

![2_AMP_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/5b53abad-6d16-441a-b382-7fc5efb4a9f0)

![2_AMP_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/7cfdad00-7d89-4203-b2b3-fe00d33bd2a8)


## Optimization #2: Increase Batch Size
The previous optimization (Automatic mixed precision) has reduced step time significantly from 117 msec to 81 msec. This technique has not only made GPU memory almost half but also reduced the GPU busy time from 73.28% to 64.6%. It makes GPU under-utilized and allows us to increase the batch size. In the image below we display the performance results when we increase the batch size to 128.
```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
```
Although the GPU utilization value did not change much, the training speed has increased significantly, from 395 samples per second (81 milliseconds for batch size 32) to 551 samples per second (232 milliseconds for batch size 128).
![3_BatchSize_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/7a085e4a-e81e-4562-a976-2c1133675383)
![3_BatchSize_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/d9b6cdab-d991-4b59-8924-1271e7173242)

## Optimization #3: Reduce Host to Device Copy
Another optimization that can increase GPU utilization is to reduce host (CPU) operations and memory transfer from host to device. One way to address this kind of bottleneck is to reduce the amount of data and operations in each batch. This can be done by converting the data type from an 8-bit unsigned integer to a 32-bit float (due to normalization) after performing the data copy. In the code block below, data type conversion and normalization are performed once the data is on the GPU:
```python
transform = T.Compose(
    [T.Resize(224),
     T.ToTensor()#,
     ]) #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```
and the Train function will be updated as
```python
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    inputs = (inputs.to(torch.float32) / 255. - 0.5) / 0.5
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
    #outputs = model(inputs)
    #loss = criterion(outputs, labels)
    # Note - torch.cuda.amp.GradScaler() may be required
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
As a result of this change, the memory copy did not change but CPU execution and other time reduce significantly. It also increases the GPU Utilization:
![4_H2D_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3a7158de-859f-4915-bd6d-0e77f98d856a)

This optimization leaves us with 631 samples per second (202 milliseconds for batch size 128).

## Optimization #4: Multi-process Data Loading
A multiprocessing data loader is a component used in machine learning frameworks like PyTorch to load and process data in parallel during model training or evaluation. It utilizes multiple processor cores or threads to speed up data loading, improve efficiency, and support tasks like data augmentation. A multiprocessing data loader typically operates on the CPU. It leverages multiple CPU cores or threads to load and preprocess data in parallel, improving efficiency and speeding up the data-loading process.
To enable this optimization following change is made:
```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
```
The results of this optimization are displayed below:
![5_multiprocessor_overview_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/23d0a466-7512-40db-ae01-fa4efa97407f)

The change to a single line of code has reduced the training step time from 202.6 msec to 145.8 msec. 

Caution: Shared memory used for multiple processing by CPU cores is 64MB by default in docker. It is not sufficient and you will get an error such as
```
Unexpected bus error encountered. This might caused by insufficient shared memory (shm)
```
It can be solved by allocating more shared memory at the time of running the docker image. The following command is used to run the Docker image and allocation of the 2GB shared memory
```
docker run --gpus=all --rm -it --net=host --shm-size=2gb nvcr.io/nvidia/pytorch:23.10-py3
```

## Optimization #5: Memory Pinning
If we analyze the performance of the last optimization in the Overview Window, we can see that a significant amount of time is still spent on processing and loading the training data into the GPU. To tackle this concern, we will implement another PyTorch-recommended optimization aimed at streamlining the data input flow and utilizing memory pinning. Utilizing pinned memory can notably enhance the speed of data transfer from the host to the device, and importantly, enables asynchronous operations. This capability enables us to concurrently prepare the next training batch on the GPU while executing the training step on the current batch. To learn more about memory pinning check this link [Lecture 21 - Pinned Memory and Streams](https://www.youtube.com/watch?v=aNchuoFCgSs&t=103s)

This memory-pinning optimization requires changes to two lines of code. First, we set the pin_memory flag of the DataLoader to True.
```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```
Then we modify the host-to-device memory transfer (in the train function) to be non-blocking:
```python
inputs, labels = data[0].to(device=device, non_blocking=True), \
                 data[1].to(device=device, non_blocking=True)
```
The results of the memory pinning optimization are displayed below:
![6_pin_memory_overview_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/88e110f0-0c87-4265-8e47-10d7a0b38259)

Our GPU utilization has increased from 79.54% to 86.22% and our step time has further decreased to 92.6 msec.

## Result
The following results are collected on Quadro RTX 4000 GPU with 8GB memory. The performance of various optimization techniques is compared with the base model. GPU memory is in GB while Avg. step time is in microseconds. Samples per sec can be calculated by dividing the batch value by Avg. Step time. Percentage optimization value is the ratio of optimized samples per sec / base samples per sec.
| Optimization Technique        | Batch Size           | GPU Memory (GB)  | Avg. Step Time (ms)  | Samples per sec  | Optimization (rel. to base)  | 
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Base_Model      | 32 | 2.51 | 117.1      | 273.5 | 100% |
| Automatic Mixed Precision      | 32 | 1.32 | 81.1      | 395.1 | 144% |
| Increase Batch Size      | 128 | 4.95 | 232.1      | 551.5 | 201% |
| Reduce H2D Copy      | 128 | 4.95 | 202.6      | 631.63 | 230% |
| Multi-process Data Loading     | 128 | 4.95 | 145.8      | 877.84 | 320% |
| Memory Pinning     | 128 | 4.95 | 92.6      | 1381.6 | 505% |

## Conclusion
In this tutorial, different optimization techniques are used that improve the performance in the training stage 5 times. These techniques are useful for training large models such as Foundation models. With proper memory analysis and a few lines of code, significant results can be achieved in the training stage.

## References
1.	PyTorch Model Performance Analysis and Optimization link: https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-10c3c5822869
2.	PyTorch Profiler With TensorBoard link: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html


