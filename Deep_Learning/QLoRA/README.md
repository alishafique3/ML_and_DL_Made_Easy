# Finetuning of LLM (model: T5-3b) on single GPU using QLoRA for summarization task.

Large Language Models (LLMs) have achieved a significant advancement in the field of natural language processing. Models such as Google’s Gemini and OpenAI’s GPT-4 have demonstrated human-like performance across a wide array of tasks involving text, images, and video. However, the training process for the LLMs demands extensive computing resources, limiting their development to a few tech giants and research groups. To mitigate this challenge, Quantized LoRA (Low-Rank Adaptation) provides an efficient method for fine-tuning the LLMs. This approach enables smaller organizations and individual developers to customize LLMs for specific tasks. 

QLoRA optimizes the memory usage of the models by learning a few quantized parameters. This process enhances the training speed and scalability while retaining adaptation flexibility. Initially, we load the model and apply quantization to reduce the memory footprint. Subsequently, we fine-tune the LoRA low-rank matrices (adapters) in the layers of the frozen quantized model. This configuration enables us to train the T5 model with three billion parameters on a single GPU.

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

### Install the following libraries inside the docker
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
Pytorch summarization task example is used as base code which is available at [Link](https://huggingface.co/docs/transformers/en/tasks/summarization), accessed on march 28, 2024. [This](https://medium.com/@hugo_fernandez/fine-tune-and-deploy-an-llm-on-google-colab-notebook-with-qlora-and-vertexai-58a838a63845) medium blog is also helpful for this tutorial.
Encoder decoder based model is used in this tutorial (google-t5/t5-3b from huggingface) which is finetuned on the popular billsum dataset. BitsAndBytes package is used to apply quantization to the model which will significantly reduce the memory footprint of the model. PEFT library is utilized to apply LoRA adapters inside the layers of the frozen quantized model.

## Fine-tuning with QLoRA (Quantized Low-Rank Adaptation)
In QLoRA, quantization is applied to reduce the memory footprint of the model. This technique involves converting the model's weights from a float32 format to a smaller one, typically 4 or 8 bits. Next, we freeze the quantized weights of the base model and perform backpropagation only on the weights of a lower-rank matrix that overlays the quantized base model.
![qlora](https://github.com/alishafique3/ML_and_DL_Made_Easy/assets/17300597/477536ee-d419-4d15-a7cc-04964a490cba)


The benefit lies in the significantly reduced number of trained weights compared to those in the base model, while maintaining the accuracy. Furthermore, the quantized model occupies much less RAM space than the original one (the google-t5/t5 3B model memory footprint reduces from approximately 11.4GB to just 4.29GB), allowing for development on a powerful local machine or a free Google Colab instance.

In this tutorial, the T5 model is used with three billion parameters. T5 is an encoder-decoder model and performs efficiently for Seq2Seq tasks.

```python
from transformers import AutoTokenizer

checkpoint = "t5-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Next, we have used quantization configurations to load the model in lower precision such as NF4 format (4-bit Normalized Float - NF4). It is a new data type ideal for normally distributed weights, and implementing double quantization to achieve additional memory conservation. However, during computations, these operations can only be executed in float16 or bfloat16, contingent upon the GPU's capabilities. As a result, they will be converted during calculation and later reverted to the compressed format.

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

- ```load_in_4bit=True``` to quantize the model to 4-bits when you load it
- ```bnb_4bit_quant_type="nf4"``` to use a special 4-bit data type for weights initialized from a normal distribution
- ```bnb_4bit_use_double_quant=True``` to use a nested quantization scheme to quantize the already quantized weights
- ```bnb_4bit_compute_dtype=torch.float16``` to use float16 for faster computation
  
Next, we load the full precision model to compare with the quantized model and to calculate the memory reduction ratio.
```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_q = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,quantization_config=bnb_config, device_map={"": 0})

model_q.get_memory_footprint()
```
```
11406393344
```
```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_q = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,quantization_config=bnb_config, device_map={"": 0}) #device_map="auto" will cause a problem in the training

model_q.get_memory_footprint()
```
```
4293910528
```
We also verify the quantized layers by printing the ```model_q```. The tensor format is Linear4bit, and the memory footprint has significantly reduced.
```python
print(model_q)
```

```
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
We also observe the names of the different layers/modules of the models (SelfAttention, DenseReluDense, etc.). we define the learning parameters of LoRA such as rank r, which is the rank of the adapter matrices. The higher this rank, the greater the number of weights in the lower-rank matrices. In our case, we set it to 32, but you can increase it if the performance is not desirable, or decrease it to reduce the number of trainable weights and memory footprint of optimizer parameters associated with each weight. The dropout rate corresponds to the proportion of weights that should be set to 0 during the training phase to make the network more robust and to prevent overfitting.

The target_modules corresponds to the names of modules that will be connected with low-rank matrices. The more modules you target, the more parameters you will have to train. modules_to_save defines which modules of the model will be trained(unfreezed) after connecting low-rank adapters.
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
- ```lora_alpha``` It specifies the value of the alpha parameter for LoRA, controlling the strength of low-rank approximation.
- ```lora_dropout``` This argument sets the dropout rate used in LoRA layers to prevent overfitting during training.
- ```r``` It determines the rank of the low-rank approximation used in LoRA. A higher value of r implies a higher rank and potentially more expressive power, but also more parameters.
- ```task_type``` Indicates the type of task the model is fine-tuned for. In this case, it's set to "SEQ_2_SEQ_LM", suggesting sequence-to-sequence language modeling.
- ```target_modules```specifies which layers/modules of the model will be adapted during fine-tuning.
- ```modules_to_save``` defines which modules of the model will be saved after adaptation.
  
Add PEFT low-rank matrices adapter to the 4-bit quantized model.
```python
model_q.add_adapter(peft_config, adapter_name="adapter_4")
model_q.set_adapter("adapter_4")
```
you can verify those layers ('v','o') connected with low-rank adapters using ```print``` command.
```
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
              (v): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=1024, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (adapter_4): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (adapter_4): Linear(in_features=1024, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (adapter_4): Linear(in_features=32, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (o): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=4096, out_features=1024, bias=False)
                (lora_dropout): ModuleDict(
                  (adapter_4): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (adapter_4): Linear(in_features=4096, out_features=32, bias=False)
                )
                (lora_B): ModuleDict(
                  (adapter_4): Linear(in_features=32, out_features=1024, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
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
you can check the number of trainable parameters and the trainable proportion as compared to the total number of parameters.
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
```trainable params: 56492032 || all params: 2908090368 || trainable%: 1.9425817237877527```

Finally, we define the training arguments and trainer configurations.
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
    fp16=True,
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
You have the flexibility to modify the batch size based on the model's size and the GPU's memory capacity. The objective is to set batch sizes that fully utilize the GPU's capabilities by avoiding CUDA out-of-memory issues.
As for the optimizer, we employ the Paged Optimizer offered by QLoRA. This optimizer utilizes a feature provided by Nvidia to transfer paged memory of optimizer states between the CPU and GPU. Its primary purpose in this context is to handle memory spikes and prevent out-of-memory errors.

```python
trainer.train()
```

Once the training is complete, you can conduct a few tests to see if the response meets your expectations and consider retraining if the result is not satisfactory.

```python
from transformers import AutoTokenizer

text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
inputs = tokenizer(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = model_q#AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Result
The following results are collected on a Tesla T4 GPU with 16 GB memory. The performance of various optimized models is compared with the base model. Memory reduction factor is also calculated by dividing the base model memory by optimized model memory.
| Model        | Memory (GB)           |Memory Reduction           | Total parameters (Billion)  | Trainable parameters (Billion)  | Trainable%  |
:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Base_Model      | 11.4 | 1 | 2.9 | 2.9      | 100% |
| Quantized Model      | 4.29 | 2.65 | 2.9 | 2.9      | 100% |
| QloRA Model      | 4.29 | 2.65 | 2.9 | 0.056      | 1.94% |


## Conclusion
In this tutorial, we utilized the QLoRA technique using BitsAndBytes and PEFT libraries to reduce memory usage during the training phase. Quantization led to a 2.65 times reduction in memory footprint, while LoRA froze the model and permitted 1.94% of parameters to be trained on the fine-tuning dataset. This setup allowed us to train large models like T5 with three billion parameters on a single GPU with 16GB memory. Such an approach will empower smaller organizations and individual developers to tailor LLMs for specific tasks.

## References
1.	Fine-tune and deploy an LLM on Google Colab Notebook with QLoRA and VertexAI Link: https://medium.com/@hugo_fernandez/fine-tune-and-deploy-an-llm-on-google-colab-notebook-with-qlora-and-vertexai-58a838a63845
2.	Huggingface summarization task: https://huggingface.co/docs/transformers/en/tasks/summarization
