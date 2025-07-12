import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 1. 加载分词器和基础模型
model_path = "/home/ma-user/work/Qwen2.5-3B-Instruct"  # 按你自己的路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)  # NPU推荐bfloat16

# 2. LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "q_proj", "v_proj"],  # qwen用c_attn，有的模型用q_proj/v_proj/others，视模型而定
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 3. 加载并处理数据（jsonl文件必须有 instruction/output 字段）
data = load_dataset("json", data_files={"train": "/home/ma-user/work/data/train5000.jsonl"})["train"]

def preprocess(sample):
    text = sample["instruction"].strip() + "\n" + sample["output"].strip()
    return tokenizer(text, truncation=True, max_length=1024, padding="max_length")

tokenized_data = data.map(preprocess, remove_columns=data.column_names)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./output-lora-train5000",
    per_device_train_batch_size=1,     # NPU建议小batch，内存够可以加大
    num_train_epochs=1,
    save_steps=100,
    logging_steps=50,
    learning_rate=1e-4,
    fp16=False,                        # NPU建议关闭fp16，推荐bfloat16
    bf16=True,                         # NPU强烈建议bfloat16
    save_total_limit=2,
    report_to=[],                      # 不用wandb等外部监控
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 5. 开始微调
trainer.train()
# 训练完成后，保存LoRA adapter
model.save_pretrained("./output-lora-train5000")
print("LoRA Adapter 已保存到 ./output-lora-train5000/")

# 结果在 output-lora-qwen/ 目录，含LoRA adapter参数和config

# 推理/合成全量模型方法可单独补充

