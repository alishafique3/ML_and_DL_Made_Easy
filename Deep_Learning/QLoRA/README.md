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
```python
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
## Baseline Model T5-3b
Pytorch summarization task example is used as base code which is available at [Link](https://huggingface.co/docs/transformers/en/tasks/summarization), accessed on march 28, 2024.
The tutorial has used a encoder-decoder model (google-t5/t5-3b from huggingface) that is trained on the popular billsum dataset. Bitsandbytes and the PEFT libraries are used to implement QLoRA adapter in T5-3b model duing the training phase. 
## Fine-tuning with QLoRA (Quantized Low-Rank Adaptation)
To achieve our goal, namely to fine-tune a model on a single GPU, we will need to quantize it. This means taking its weights, which are in a float32 format, and reducing them to a smaller format, here 4 bits. Then, for training, we will use QLORA, which is a quantized version of LoRA (see here). With QLoRA, we freeze the quantize weights of the base model and perform backpropagation only on the weights of a lower-rank matrix that overlays the base model.
![lora](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/4d490c99-86ca-4c09-86ce-bdf10a49ebc5)
The advantage is that the number of weights trained is much lower than the number of weights in the base model, while still maintaining a good level of accuracy. Moreover the Quantize model takes much less space on the RAM than the original one (google-t5/t5 3B model pass from ~11.4GB to just 4.29GB!) , meaning that you can run it on a powerful local machine or on a free google Colab instance.

For the model selection, you can opt for models that have up to about 20 billion parameters (see here) beyond that, you will have to get a better GPU. I have chosen as the base model the 7B model from MistralAI, which shows very good performance compared to other models of its size, and even manages to outperform larger language models like Llama 2 13B. (more details on the paper they release here).

```python
from transformers import AutoTokenizer

checkpoint = "t5-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Next, we create the Quantization parameters using the most optimal values: by loading the model in 4 bits, using the NF4 format (4-bit NormalFloat (NF4), a new data type that is optimal for normally distributed weight), and by using double quantization which allows for further memory savings. However, for computations, these can only be performed in float16 or bfloat16 depending on the GPU, so they will be converted during calculation and then reconverted into the compressed format.
```python
#Quantization as defined https://huggingface.co/docs/optimum/concept_guides/quantization will help us reduce the size of the model for it to fit on a single GPU 
#Quantization configuration
compute_dtype = getattr(torch, "float16")
print(compute_dtype)
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
```
Next, we load the model and quantize it on the fly using the previous configuration. If you have a GPU that is compatible with flash attention, set it to True. We force the device map to load the model on our GPU.
```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_q = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,quantization_config=bnb_config, device_map={"": 0}) #device_map="auto" will cause a problem in the training

model_q.get_memory_footprint()
```
We can then verify that our model has been successfully loaded and that the tensor format is indeed Linear4bit, and that the model is ready to be trained.
```python
print(model_q)
```

```python
T5ForConditionalGeneration(
  (shared): Embedding(32128, 1024)
  (encoder): T5Stack(
    (embed_tokens): Embedding(32128, 1024)
    (block): ModuleList(
      (0): T5Block(
        (layer): ModuleList(
          (0): T5LayerSelfAttention(
            (SelfAttention): T5Attention(
              (q): Linear4bit(in_features=1024, out_features=4096, bias=False)
              (k): Linear4bit(in_features=1024, out_features=4096, bias=False)
              (v): Linear4bit(in_features=1024, out_features=4096, bias=False)
              (o): Linear4bit(in_features=4096, out_features=1024, bias=False)
              (relative_attention_bias): Embedding(32, 32)
            )
            (layer_norm): FusedRMSNorm(torch.Size([1024]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): T5LayerFF(
            (DenseReluDense): T5DenseActDense(
              (wi): Linear4bit(in_features=1024, out_features=16384, bias=False)
              (wo): Linear(in_features=16384, out_features=1024, bias=False)
              (dropout): Dropout(p=0.1, inplace=False)
              (act): ReLU()
            )
            (layer_norm): FusedRMSNorm(torch.Size([1024]), eps=1e-06, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
...
```
We also notice the names of the different elements of the models (MistralDecoderLayer, MistralRotaryEmbedding, etc.). Next, we define the learning parameters of LoRA. We set the rank r, which is the rank each matrix should have. The higher this rank, the greater the number of weights in the lower-rank matrices. We set it to 16 for this example, but you can increase it if the performance is not satisfactory, or decrease it to reduce the number of trainable parameters. The dropout rate corresponds to the proportion of weights that should be set to 0 during training to make the network more robust and to prevent overfitting.

The target_modules corresponds to the names of modules that appear when we printed the model (q_proj, k_proj, v_proj, etc.). If you are using a different model, replace this line with the list of modules you want to target. The more modules you target, the more parameter you will have to train.
```python
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules= ['v', 'o'],
        modules_to_save=["lm_head"],
)
```
Add PEFT adapter to the 4bit model.
```python
model_q.add_adapter(peft_config, adapter_name="adapter_4")
model_q.set_adapter("adapter_4")
```

you can check the number of trainable parameters and the proportion they represent compared to the total number of parameters.
```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = model.num_parameters()
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model_q)
```

Finally, we define the training arguments.
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    evaluation_strategy="epoch",
    optim="paged_adamw_8bit", #used with QLoRA
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    learning_rate=2e-5,
    num_train_epochs=4,
    predict_with_generate=True,
    #fp16=True,
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model_q,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```
You can adjust the batch size depending on the size of the model and the GPU at your disposal (the resource tab on Colab will provide this information). Your goal here is to define batch sizes that maximize GPU usage without exceeding it.
For the optimizer, we use the Paged Optimizer provided by QLoRA. Paged optimizer is a feature provided by Nvidia to move paged memory of optimizer states between the CPU and GPU. It is mainly used here to manage memory spikes and avoid out-of-memory errors.
Set a low learning rate because we want to stay close to the original model.
Here we define the number of epoch to 1 but to obtain a pretty good result you should go for 3/4 epoch on your data.
