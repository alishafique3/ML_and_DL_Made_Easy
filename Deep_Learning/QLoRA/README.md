# Finetuning of LLM (model: T5-3b) on single GPU using QLoRA for summarization task.

The recent emergence of fine-tuning methods such as QLoRA that can run on a single GPU has made this approach much more accessible. We will start by loading the model and quantize it using BitsAndBytes package from HuggingFace. Then we will use QLoRA, that help us fine-tune LoRA adaptater on top of frozen quantize model. Thanks to the use of 4bits model we will be able to run training on a single GPUs.

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

### Install following libraries inside the docker
```console
! pip install bitsandbytes transformers peft accelerate 
! pip install datasets trl ninja packaging
! pip install evaluate rouge_score
```

## Baseline Model
Pytorch example "PyTorch Profiler With TensorBoard" is used as base code which is available [Link](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) accessed on February 2, 2024.

The tutorial has used a classification model (based on the Mobilenet_V2 architecture) that is trained on the popular CIFAR10 dataset. PyTorch Profiler and the PyTorch TensorBoard plugin are used to identify a bottleneck in the training step. 

![1_baseline_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3c871ec6-9aac-45c3-a306-7e43f5f65fe7)
![1_baseline_memory_u](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/3e1a7661-e364-4987-8b4a-1953b3081aa1)
As we can see, the step time is 117 msec with the GPU utilization is 73.28%. The average memory used in each training step can be found in the "Memory View" window. The base model uses almost 2.5GB in each training step. Some important terminologies regarding PyTorch GPU summary are explained below:
