本项目使用LLM进行NLP任务中的关系分类（以人物关系分类数据为例），训练框架使用[Firefly](https://github.com/yangjianxin1/Firefly).

### 数据集

人物关系分类数据集：[https://huggingface.co/datasets/jclian91/people_relation_classification](https://huggingface.co/datasets/jclian91/people_relation_classification)

### 模型训练

1. 运行`./script/preprocess.py`，将训练集转化为Firefly支持的格式
2. 运行`./script/model_train.sh`，进行模型训练；训练完后再合并模型权重
3. 运行`./script/single_chat_server.py`，部署训练好的模型
4. 运行`./script/model_eval.py`进行模型评估

### 使用模型

- Baichuan-7b
- Baichuan-13b

### 模型测评

- Qlora

#### 基础训练参数

```json
{
    "output_dir": "output/firefly-baichuan-7b-people",
    "model_name_or_path": "~/baichun_7b",
    "train_file": "./data/people.jsonl",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "max_seq_length": 256,
    "logging_steps": 100,
    "save_steps": 100,
    "save_total_limit": 1,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 100,
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,

    "gradient_checkpointing": true,
    "disable_tqdm": false,
    "optim": "paged_adamw_32bit",
    "seed": 42,
    "fp16": true,
    "report_to": "tensorboard",
    "dataloader_num_workers": 0,
    "save_strategy": "steps",
    "weight_decay": 0,
    "max_grad_norm": 0.3,
    "remove_unused_columns": false
}
```

#### 测评结果

| 基座模型 | 模型参数 | 评估指标 |
|------|------|------|
|      |      |      |


### 测试案例