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
pip install bitsandbytes==0.43.0
pip install transformers==4.39.1
pip install peft==0.10.0
pip install accelerate==0.28.0 
pip install datasets==2.18.0
pip install trl==0.8.1
pip install ninja==1.11.1.1
pip install packaging
pip install evaluate==0.4.1
pip install rouge-score==0.1.2
```

## Baseline Model
Pytorch summarization task example is used as base code which is available at [Link](https://huggingface.co/docs/transformers/en/tasks/summarization), accessed on march 28, 2024.

The tutorial has used a encoder-decoder model (google-t5/t5-3b from huggingface) that is trained on the popular billsum dataset. Bitsandbytes and the PEFT libraries are used to implement QLoRA adapter in T5-3b model duing the training phase. 

