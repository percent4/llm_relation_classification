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

1. 基础参数

```bash
              precision    recall  f1-score   support

         上下级     0.6000    0.8710    0.7105        31
         不知道     0.8989    0.8086    0.8514       209
          亲戚     0.7826    0.7500    0.7660        24
        兄弟姐妹     0.7500    0.9706    0.8462        34
       出生一个人     0.0000    0.0000    0.0000         0
          合作     0.9348    0.7288    0.8190        59
        同一个人     0.9744    0.9744    0.9744        39
          同学     0.9091    0.8333    0.8696        24
          同门     1.0000    0.6923    0.8182        26
          夫妻     0.8409    0.9367    0.8862        79
          好友     1.0000    0.8667    0.9286        30
          师生     0.8250    0.8919    0.8571        37
          情侣     0.8485    0.9032    0.8750        31
          父母     0.9130    0.9844    0.9474       128
          祖孙     0.9200    0.9200    0.9200        25

    accuracy                         0.8711       776
   macro avg     0.8131    0.8088    0.8046       776
weighted avg     0.8826    0.8711    0.8719       776
```
只有一条关系被识别为出生一个人，评估结果影响不大。

2. 13b模型 + 基础参数

learning_rate: 1e-4

```bash
              precision    recall  f1-score   support

         上下级     0.6842    0.8387    0.7536        31
         不知道     0.8905    0.8565    0.8732       209
          亲戚     0.9048    0.7917    0.8444        24
        兄弟姐妹     0.8649    0.9412    0.9014        34
         出生地     0.0000    0.0000    0.0000         0
          合作     0.8909    0.8305    0.8596        59
        同一个人     1.0000    0.9744    0.9870        39
          同学     0.9583    0.9583    0.9583        24
          同门     0.9615    0.9615    0.9615        26
          夫妻     0.9487    0.9367    0.9427        79
          好友     0.9600    0.8000    0.8727        30
          师生     0.9429    0.8919    0.9167        37
          情侣     0.8286    0.9355    0.8788        31
          父母     0.9343    1.0000    0.9660       128
          祖孙     0.9200    0.9200    0.9200        25

    accuracy                         0.9046       776
   macro avg     0.8460    0.8425    0.8424       776
weighted avg     0.9084    0.9046    0.9052       776
```

只有一条关系被识别为出生，评估结果影响不大。

| 基座模型 | 模型参数 | 评估指标 |
|------|------|------|
|      |      |      |


### 测试案例