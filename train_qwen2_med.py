#导入各种包，为代码调用做准备。
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from transformers import BitsAndBytesConfig
# 加载、处理数据和得到训练集、验证集。
train_jsonl_new_path = "data/cpmi_dev.json"
val_jsonl_new_path= "data/cpmi_val.json"

def process_func(example):
    """
    将数据集进行预处理, 处理成模型可以接受的格式
    """

    MAX_LENGTH = 512#此处要根据数据实际情况而定，大部分数据的长度是多少，就定多少。 
    input_ids, attention_mask, labels = [], [], []
    
    instruction = tokenizer(
        f"<|im_start|>system\n{example['instruction']}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   



model_id = "qwen/Qwen2-1.5B-Instruct"    
model_dir = "./qwen/Qwen2-1___5B-Instruct"

# 在modelscope上下载Qwen模型到本地目录下
model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

# Transformers加载预训练模型权重
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)

# 得到训练集、验证集
train_df = pd.read_json(train_jsonl_new_path)
val_df = pd.read_json(val_jsonl_new_path)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
val_ds = Dataset.from_pandas(val_df)
val_dataset = val_ds.map(process_func, remove_columns=val_ds.column_names)

#qlora量化设置
# 根据设备自动分配模型参数。
# 使用4位量化。
# 计算数据类型为torch.float16。
# 启用双量化及特定量化类型“nf4”。
# 模型张量数据类型为torch.bfloat16。
#llm_int8_threshold=6.0：设定权重转换为int8数据类型的阈值。数值大于此阈值的权重会被转换为int8格式，以减少模型大小和加速推理过程。
#llm_int8_has_fp16_weight=False：指示模型权重是否已经以fp16（半精度浮点数）存储。如果为False，则表示模型权重不是以fp16格式存储，加载时不会应用额#外的fp16优化。
# model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto",trust_remote_code=True,quantization_config=BitsAndBytesConfig(
#                                                  load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
#                                                  bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
#                                                  llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False),
#                                                 torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法，节约显存

#设置 LoRA 配置项
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # alpha值越大，LoRA调整基础模型的能力越强；反之，则调整能力较弱。例如，当alpha=32时，相比alpha=8，模型在训练过程中会对基础权重进行更大程度的调整，这意味着前者可能学习更快但也更容易过拟合。选择合适的alpha值有助于平衡模型性能和泛化能力。
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)#应用Lora配置

#定义各种超参
args = TrainingArguments(
    output_dir="./output/Qwen2-med",#训练结果保存的目录
    per_device_train_batch_size=4,#每个设备上的训练批次大小
    per_device_eval_batch_size=4,#每个设备上的验证批次大小
    eval_strategy="steps",# 评估策略，这里设置为按训练步骤评估
    eval_steps=100,#每100步评估一次
    gradient_accumulation_steps=4,#梯度累积步数，在进行反向传播前累积多少步
    logging_steps=10,#日志记录的频率，每10步记录一次训练状态。
    num_train_epochs=15,# 训练的总轮数
    save_steps=500,#每隔500步保存一次模型
    learning_rate=1e-4,#学习率
    gradient_checkpointing=True,# 是否启用梯度检查点，设置为True，可以节省显存。
)

#定义可视化监控
swanlab_callback = SwanLabCallback(
    project="Qwen2-med-fintune",
    experiment_name="Qwen2-1.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在中医对话数据集上微调，实现中医对话任务。",
    config={
        "model": model_id,
        "model_dir": model_dir,
        "dataset": "qgyd2021/chinese_med_sft",
    },
)
#定义 Trainer
#定义训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),#定义数据规整器：训练时自动将数据拆分成 Batch
    callbacks=[swanlab_callback],#回调函数，可视化训练效果。
)
#开始训练
trainer.train()
#进行测试
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
    return response

# 用测试集的随机20条，测试模型
# 得到测试集
test_jsonl_new_path= "data/cpmi_test.json"


# 得到训练集
test_df = pd.read_json(test_jsonl_new_path)

test_df = test_df[:int(len(test_df) * 0.1)].sample(n=20)

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))
    
swanlab.log({"Prediction": test_text_list})
swanlab.finish()